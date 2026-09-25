"""Dataset writer for structural mutations, using official LeRobot v3 writing."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from lerobot_cleaner.transforms.vision.roi_crop import crop_image
from lerobot_cleaner.core.language import valid_index, text_error

IDENTITY = {"index", "frame_index", "episode_index", "timestamp", "task_index"}


from .images import image_array


class V3DatasetWriter:
    """Official add_frame/save_episode owns offsets, shards, tasks and stats."""
    def __init__(self, storage, root, plans, batch_frames=16):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.storage, self.root = storage, Path(root)
        self.batch_frames = batch_frames
        self.features = deepcopy(storage.info["features"])
        for key in IDENTITY:
            self.features.pop(key, None)
        # A feature must have one fixed image size across the output dataset.
        crops = {}
        for plan in plans:
            if plan.reject_episode:
                continue
            for key, settings in plan.crop.items():
                if key in crops and crops[key] != settings:
                    raise ValueError("Per-episode ROI changes require a common output shape")
                crops[key] = settings
        self.crops = crops
        for key, settings in crops.items():
            left, top, right, bottom = settings["box"]
            width, height = settings.get("resize") or (right-left, bottom-top)
            old_shape = list(self.features[key]["shape"])
            self.features[key]["shape"] = (height, width, *old_shape[2:])
        self.dataset = LeRobotDataset.create(
            repo_id=f"local/{self.root.name}", root=self.root,
            fps=storage.info["fps"], robot_type=storage.info.get("robot_type"),
            features=self.features, use_videos=bool(storage.meta.video_keys),
            image_writer_processes=0, image_writer_threads=4)
        self.source_root = storage.root
        self.episodes = self.frames = 0
        self.episode_map = []
        self.finalized = False

    def _task(self, task_index):
        if not valid_index(task_index):
            raise ValueError("Official writer requires an integer task index; no implicit repair")
        tasks = self.storage.tasks
        matches = tasks[tasks["task_index"] == task_index]
        if len(matches) != 1 or not valid_index(matches["task_index"].iloc[0]):
            raise ValueError("Unresolved source task")
        value = matches["task"].iloc[0] if "task" in matches else matches.index[0]
        if text_error(value):
            raise ValueError("Official writer requires nonempty task text")
        return value

    def write_episode(self, episode):
        if episode.dropped:
            raise ValueError("EpisodeFilter must reject dropped episodes before writing")
        source_times = np.asarray(episode.metadata["source_timestamps"], dtype=float)
        if len(source_times) != len(episode.df) or len(episode.keep_indices) != len(episode.df):
            raise ValueError("Frame selection does not cover every output modality")
        if not episode.metadata["transform_plan"]["reindex"]:
            if not np.allclose(episode.df.timestamp, np.arange(len(episode.df))/episode.fps, atol=1e-4, rtol=0):
                raise ValueError("Official writer needs a uniform clock; explicitly plan retime or frame filtering")
        actual_crops = episode.metadata.get("video_crop", {})
        if actual_crops != self.crops:
            raise ValueError("Every retained episode must use the declared output crop layout")
        for start in range(0, len(episode.df), self.batch_frames):
            stop = min(start+self.batch_frames, len(episode.df))
            video = {}
            for key in self.storage.meta.video_keys:
                tensors = self.storage.video_frames(episode.episode_index, key, source_times[start:stop])
                arrays = tensors.detach().cpu().numpy() if hasattr(tensors, "detach") else np.asarray(tensors)
                if len(arrays) != stop-start:
                    raise ValueError("Video decoder did not preserve the common frame selection")
                video[key] = arrays
            for position in range(start, stop):
                row = episode.df.iloc[position]
                output = {"task": self._task(episode.df["task_index"].iloc[position])}
                for key, spec in self.features.items():
                    if spec["dtype"] == "video":
                        pixels = np.moveaxis(video[key][position-start], 0, -1)
                        if np.issubdtype(pixels.dtype, np.floating):
                            pixels = np.clip(np.rint(pixels*255), 0, 255).astype(np.uint8)
                        value = crop_image(pixels, actual_crops[key]) if key in actual_crops else pixels
                    elif spec["dtype"] == "image":
                        value = image_array(row[key], self.storage.root)
                        if key in actual_crops:
                            value = crop_image(value, actual_crops[key])
                    elif spec["dtype"] in {"string", "str"}:
                        value = row[key]
                    else:
                        value = np.asarray(row[key], dtype=spec["dtype"]).reshape(tuple(spec["shape"]))
                        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                            raise ValueError(f"Refusing to publish nonfinite output: {key}")
                    if key == "task" and (not isinstance(value, str) or value != output["task"]):
                        raise ValueError("Sample task conflicts with the canonical task table; no implicit rewrite")
                    output[key] = value
                self.dataset.add_frame(output)
        self.dataset.save_episode()
        self.episode_map.append({"source_episode": episode.episode_index,
                                 "output_episode": self.episodes,
                                 "source_frames": episode.keep_indices})
        self.episodes += 1
        self.frames += len(episode.df)

    def finalize(self):
        if not self.finalized:
            self.dataset.finalize()
            self.finalized = True

    def close(self):
        # On failure leave only unpublished staging data; flush/stop official workers.
        if not self.finalized:
            self.finalize()
