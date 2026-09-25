"""LIBERO-specific v3 validation entry point (profile-driven review)."""

__all__ = ["validate_libero"]


def validate_libero(dataset, batch_rows=8192, profile=None):
    from lerobot_cleaner.v30.episode_review import DatasetReview
    from lerobot_cleaner.v30.v3 import V3Config
    from lerobot_cleaner.v30.v3_streaming import audit_streaming

    config = V3Config(engine="streaming", batch_rows=batch_rows, max_frames=30000000)
    observer = DatasetReview(dataset, profile)
    return audit_streaming(dataset, config, observer=observer)["dataset_review"]
