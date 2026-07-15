"""lerobot-cleaner: a configurable cleaning tool for GR00T-format LeRobot datasets."""

from lerobot_cleaner.config import CleaningConfig

__version__ = "0.1.0"


def clean(input_path, output_path, config=None, **kwargs):
    """Programmatic entry point mirroring the CLI.

    Args:
        input_path: path to the source GR00T LeRobot dataset.
        output_path: path for the cleaned dataset (must not be the input).
        config: a :class:`CleaningConfig`, a path to a yaml file, or None to use
            defaults. Extra kwargs override top-level config fields.

    Returns:
        The :class:`~lerobot_cleaner.report.CleaningReport` produced by the run.
    """
    from pathlib import Path

    from lerobot_cleaner.pipeline import Pipeline

    if config is None:
        cfg = CleaningConfig()
    elif isinstance(config, CleaningConfig):
        cfg = config
    else:
        cfg = CleaningConfig.from_yaml(config)

    cfg.input = Path(input_path)
    cfg.output = Path(output_path)
    for k, v in kwargs.items():
        setattr(cfg, k, v)

    return Pipeline(cfg).run()


__all__ = ["CleaningConfig", "clean", "__version__"]
