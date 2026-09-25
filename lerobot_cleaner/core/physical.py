"""Optional, model-independent physical declarations and a single delta operator."""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class _StringEnum(str, Enum):
    def __str__(self):
        return self.value


class Unit(_StringEnum):
    RAD = "rad"
    DEG = "deg"
    M = "m"
    MM = "mm"
    M_S = "m/s"
    MM_S = "mm/s"
    RAD_S = "rad/s"
    DEG_S = "deg/s"
    N = "N"
    NM = "Nm"
    DIMENSIONLESS = "dimensionless"


class Representation(_StringEnum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    CARTESIAN_POSITION = "cartesian_position"
    POSE = "pose"
    DELTA_POSE = "delta_pose"
    QUATERNION = "quaternion"
    ROTATION_MATRIX = "rotation_matrix"
    GRIPPER_POSITION = "gripper_position"


class ControlType(_StringEnum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    DELTA_JOINT_POSITION = "delta_joint_position"
    EEF_POSE = "eef_pose"
    DELTA_POSE = "delta_pose"
    GRIPPER_POSITION = "gripper_position"


class CoordinateFrame(_StringEnum):
    WORLD = "world"
    ROBOT_BASE = "robot_base"
    EEF = "eef"
    CAMERA = "camera"


PHYSICAL_FIELDS = ("unit", "representation", "control_type", "periodic", "coordinate_frame")


@dataclass(frozen=True, kw_only=True)
class PhysicalSemantics:
    unit: Unit | None = None
    representation: Representation | None = None
    control_type: ControlType | None = None
    periodic: bool | None = None
    coordinate_frame: CoordinateFrame | None = None

    def __post_init__(self):
        for name, enum in (("unit", Unit), ("representation", Representation),
                           ("control_type", ControlType), ("coordinate_frame", CoordinateFrame)):
            value = getattr(self, name)
            if value is not None:
                try:
                    object.__setattr__(self, name, enum(value))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid physical {name}={value!r}; expected {[v.value for v in enum]}") from exc
        if self.periodic is not None and type(self.periodic) is not bool:
            raise ValueError("Physical periodic must be boolean or null")

    def physical_metadata(self):
        return {key: value.value if isinstance(value, Enum) else value
                for key in PHYSICAL_FIELDS for value in (getattr(self, key),)}

    def physical(self):
        return PhysicalSemantics(**self.physical_metadata())


class PhysicalDeltaResolver:
    """First-order displacement only; higher derivatives must never be wrapped."""
    @staticmethod
    def metadata(trajectory, source, columns=None):
        semantics = getattr(trajectory, source + "_semantics")
        return tuple(semantics[i] for i in columns) if columns is not None else semantics

    @staticmethod
    def unsupported(item):
        if item.representation in {"pose", "delta_pose", "quaternion", "rotation_matrix"} or item.control_type in {"eef_pose", "delta_pose"}:
            return "geometry-aware delta is not implemented for this representation/control_type"
        if item.periodic:
            if item.unit not in {"rad", "deg"}:
                return "periodic delta requires an explicit rad or deg unit"
            if item.representation not in {None, "joint_position"} or item.control_type not in {None, "joint_position"}:
                return "periodic wrapping is only supported for absolute angular positions"
        return None

    @classmethod
    def difference(cls, values, semantics):
        if len(semantics) != values.shape[1]:
            raise ValueError("Physical metadata width differs from selected data")
        for i, item in enumerate(semantics):
            reason = cls.unsupported(item)
            if reason:
                raise ValueError(f"Physical delta column {i}: {reason}; metadata={item.physical_metadata()}")
        delta = np.diff(values, axis=0)
        for i, item in enumerate(semantics):
            if item.periodic:
                period = 2*np.pi if item.unit == "rad" else 360.
                delta[:, i] = (delta[:, i] + period/2) % period - period/2
        return delta

    @classmethod
    def trajectory_difference(cls, trajectory, source="state", columns=None):
        values = getattr(trajectory, source)
        if columns is not None:
            values = values[:, columns]
        return cls.difference(values, cls.metadata(trajectory, source, columns))

    @classmethod
    def describe(cls, semantics, order):
        descriptions = []
        for item in semantics:
            base = item.control_type or item.representation
            if base == "joint_velocity":
                interpretation = {1:"joint_acceleration", 2:"joint_jerk"}.get(order, "higher_joint_velocity_derivative")
            elif base == "joint_torque":
                interpretation = "torque_time_derivative"
            elif base == "delta_joint_position":
                interpretation = "delta_command_time_derivative (not measured motion)"
            elif base in {"joint_position", "cartesian_position", "gripper_position"}:
                interpretation = {1:"position_velocity",2:"position_acceleration",3:"position_jerk"}[order]
            else:
                interpretation = "generic_numerical_derivative"
            descriptions.append({**item.physical_metadata(), "derivative_order": order,
                "interpretation": interpretation, "threshold_unit": f"({item.unit})/s^{order}" if item.unit else None,
                "delta_operator": "wrapped" if item.periodic else "euclidean"})
        return {"physical_semantics": descriptions,
            "physical_semantics_available": any(any(v is not None for v in item.physical_metadata().values()) for item in semantics),
            "note": "Thresholds are in native feature units per time power; no unit/frame conversion."}
