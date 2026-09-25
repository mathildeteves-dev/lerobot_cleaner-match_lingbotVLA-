"""Write LeRobot metadata, then verify final video/parquet frame alignment."""
from .base import DatasetFinalizer


class LeRobotMetadataFinalizer(DatasetFinalizer):
    name = "lerobot_metadata"

    def __init__(self, config):
        self.config = config

    def finalize(self, results, writer, report):
        writer.finalize_metadata()
        if self.config.verify_alignment:
            for record in writer.episodes_out:
                error = writer.verify_episode_alignment(record["episode_index"], record["length"])
                if error:
                    report.add_alignment_error(error)
