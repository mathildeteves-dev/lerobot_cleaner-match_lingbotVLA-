from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[2]

ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_configs" / "r1pro.yaml"

TRAIN_CONFIG = PROJECT_ROOT / "configs" / "vla" / "r1pro_load20000h.yaml"


def load_yaml(path: Path):
    assert path.is_file(), f"配置文件不存在：{path}"

    content = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(content, dict), f"配置文件为空或格式错误：{path}"
    return content


def test_robot_config_has_required_sections():
    config = load_yaml(ROBOT_CONFIG)

    assert "states" in config
    assert "actions" in config
    assert "images" in config

    assert isinstance(config["states"], list)
    assert isinstance(config["actions"], list)
    assert isinstance(config["images"], list)


def test_training_config_has_required_sections():
    config = load_yaml(TRAIN_CONFIG)

    assert "model" in config
    assert "data" in config
    assert "train" in config


def test_data_name_matches_robot_config_filename():
    config = load_yaml(TRAIN_CONFIG)

    assert config["data"]["data_name"] == ROBOT_CONFIG.stem


def test_robot_config_root_is_correct():
    config = load_yaml(TRAIN_CONFIG)

    assert config["data"]["robot_config_root"] == "configs/robot_configs"


def test_joint_names_are_consistent():
    robot = load_yaml(ROBOT_CONFIG)
    training = load_yaml(TRAIN_CONFIG)

    configured_joints = {next(iter(item)) for item in training["data"]["joints"]}

    state_joints = {next(iter(item)).removeprefix("observation.state.") for item in robot["states"]}

    action_joints = {next(iter(item)).removeprefix("action.") for item in robot["actions"]}

    assert state_joints <= configured_joints
    assert action_joints <= configured_joints


def test_camera_names_are_consistent():
    robot = load_yaml(ROBOT_CONFIG)
    training = load_yaml(TRAIN_CONFIG)

    configured_cameras = set(training["data"]["cameras"])

    robot_cameras = set()
    for item in robot["images"]:
        if isinstance(item, str):
            target_key = item
        else:
            target_key = next(iter(item))

        camera_name = target_key.removeprefix("observation.images.")
        robot_cameras.add(camera_name)

    assert robot_cameras <= configured_cameras
