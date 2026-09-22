"""Physical metadata propagation and wrapped motion checks, without dataset I/O."""
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import pytest
from lerobot_cleaner.adapters.schema import FeatureSchema, FeatureSlice, CanonicalFeatureSchema
from lerobot_cleaner.adapters.mapping import parse_mapping
from lerobot_cleaner.adapters.resolver import FeatureResolver
from lerobot_cleaner.adapters.builder import EpisodeBuilder
from lerobot_cleaner.core.physical import PhysicalSemantics, PhysicalDeltaResolver, Unit
from lerobot_cleaner.core.trajectory import TrajectoryView
from lerobot_cleaner.core.quality._derivative import derivative_values
from lerobot_cleaner.core.quality import check_velocity, check_acceleration, check_jerk, check_static_ratio
from lerobot_cleaner.core.quality.motion.static_edges import check_static_edges
from lerobot_cleaner.v30.quality import TrajectoryQualityConfig, QualityGroup, audit_trajectory
from lerobot_cleaner.v30.quality_groups import QualityGroupResolver


def schema(unit="rad", periodic=True, **extra):
    return CanonicalFeatureSchema(states=(FeatureSchema("state.arm", "state", (FeatureSlice("q",0,1),),
        unit=unit, periodic=periodic, representation="joint_position", **extra),))


def frame(values):
    return pd.DataFrame({"q":[[v] for v in values],"timestamp":np.arange(len(values),dtype=float),
                         "episode_index":[0]*len(values)})


def test_optional_typed_fields_preserve_positional_constructor():
    old = FeatureSchema("state.arm","state",(FeatureSlice("q",0,1),))
    assert all(v is None for v in old.physical_metadata().values())
    new = FeatureSchema("state.arm","state",old.slices,unit="rad",representation="joint_position",periodic=True)
    assert new.unit is Unit.RAD
    assert json.loads(json.dumps(asdict(new)))["unit"] == "rad"


@pytest.mark.parametrize("kwargs", [{"unit":"radianz"},{"representation":"robot_joint"},
    {"control_type":"magic"},{"coordinate_frame":"unknown_robot"},{"periodic":"true"},{"periodic":1}])
def test_invalid_physical_declarations_are_not_arbitrary_strings(kwargs):
    with pytest.raises(ValueError,match="[Pp]hysical"):
        FeatureSchema("state.arm","state",**kwargs)


@pytest.mark.parametrize("sources", [{"source":"q"},{"sources":[{"source":"q"}],"dimension":1}])
def test_generic_mapping_to_resolved_group_preserves_all_metadata(sources):
    entry = {**sources,"unit":"rad","representation":"joint_position","control_type":"joint_position", "periodic":True,"coordinate_frame":"robot_base"}
    spec = parse_mapping({"canonical":{"action":{"arm":entry}}})
    feature = FeatureResolver().resolve_mapping(spec,{"q":{"dtype":"float32","shape":[1]}})[0]
    canonical = CanonicalFeatureSchema(actions=(feature,))
    group = QualityGroupResolver().resolve(QualityGroup(name="arm",feature="action.arm"),canonical,{"state":0,"action":1})
    resolved = group["resolved"]["feature_ranges"][0]
    assert resolved["source_key"] == "q" and resolved["canonical_path"] == "action.arm"
    for key,value in feature.physical_metadata().items():
        assert resolved[key] == value == group["resolved"]["physical_semantics"][0][key]


@pytest.mark.parametrize("unit,values,expected", [("rad",[3.13,-3.13],2*np.pi-6.26),("deg",[179.,-179.],2.)])
def test_wrapped_first_difference_native_units_and_readonly(unit,values,expected):
    episode = EpisodeBuilder(schema(unit),1).build(frame(values))
    view = episode.to_trajectory()
    np.testing.assert_allclose(PhysicalDeltaResolver.trajectory_difference(view),[[expected]])
    result = check_velocity(view,threshold=.1 if unit=="rad" else 3.)
    assert result.passed and result.metrics["physical_semantics"][0]["delta_operator"] == "wrapped"
    assert check_static_ratio(view,epsilon=expected+.001).metrics["static_ratio"] == 1.
    np.testing.assert_array_equal(view.state[:,0],values)
    assert not view.state.flags.writeable


def test_higher_derivatives_do_not_wrap_velocity_or_acceleration():
    # Differences +170, -170, +170 deg: the second derivative is -340, +340, not +20/-20.
    view = EpisodeBuilder(schema("deg"),1).build(frame([0,170,0,170])).to_trajectory()
    acceleration,_,_ = derivative_values(view,2)
    jerk,_,_ = derivative_values(view,3)
    np.testing.assert_allclose(acceleration[:,0],[-340,340])
    np.testing.assert_allclose(jerk[:,0],[680])


def test_nonuniform_timestamps_and_boundary_crossing_constant_velocity():
    data = frame([179,-179,-175,-169]);data.timestamp=[0.,1.,3.,6.]
    view = EpisodeBuilder(schema("deg"),1).build(data).to_trajectory()
    for order, expected in ((1,[2,2,2]),(2,[0,0]),(3,[0])):
        values,_,_ = derivative_values(view,order)
        np.testing.assert_allclose(values[:,0],expected,atol=1e-12)


@pytest.mark.parametrize("group", [{"name":"arm","feature":"state.arm"}, {"name":"arm","source":"state","columns":[0]}])
def test_semantic_and_legacy_group_use_physical_delta(group):
    config = TrajectoryQualityConfig(groups=[{**group,"velocity":.1,"acceleration":.1,"jerk":.1}])
    report = audit_trajectory(frame([3.13,-3.13,-3.10,-3.08]),1,config,schema=schema())
    arm = report["groups"]["arm"]
    assert arm["resolved"]["physical_semantics"][0]["periodic"]
    assert arm["checks"]["velocity"]["passed"]
    assert arm["checks"]["acceleration"]["passed"] and arm["checks"]["jerk"]["passed"]


def test_legacy_untagged_data_keeps_euclidean_behavior():
    view = TrajectoryView(np.array([[3.13],[-3.13]]),np.empty((2,0)),np.arange(2.),1)
    finding = check_velocity(view,1.)
    assert not finding.passed
    assert finding.metrics["max_velocity"] == pytest.approx(6.26)
    assert finding.metrics["physical_semantics_available"] is False


@pytest.mark.parametrize("metadata", [PhysicalSemantics(periodic=True),
    PhysicalSemantics(periodic=True,unit="m"),PhysicalSemantics(representation="quaternion"),
    PhysicalSemantics(representation="rotation_matrix"),PhysicalSemantics(control_type="delta_pose"),
    PhysicalSemantics(unit="rad",periodic=True,control_type="delta_joint_position")])
def test_unsupported_physics_is_unevaluated_not_false_euclidean_result(metadata):
    view = TrajectoryView(np.ones((5,1)),np.empty((5,0)),np.arange(5.),1,(metadata,))
    for check in (check_velocity,check_acceleration,check_jerk,check_static_ratio,check_static_edges):
        finding = check(view,source="state")
        assert not finding.metrics["evaluated"] and not finding.passed
        assert "Physical delta" in finding.message


@pytest.mark.parametrize("control,meaning", [("joint_position","position_velocity"),("joint_velocity","joint_acceleration"),
    ("joint_torque","torque_time_derivative"),("delta_joint_position","delta_command_time_derivative")])
def test_control_space_interpretation_is_reported(control,meaning):
    view = TrajectoryView(np.empty((4,0)),np.arange(4.).reshape(-1,1),np.arange(4.),1,
        action_semantics=(PhysicalSemantics(control_type=control),))
    found = check_velocity(view,source="action")
    assert meaning in found.metrics["physical_semantics"][0]["interpretation"]


def test_reordered_columns_bind_physics_and_raw_gaps_stay_unknown():
    raw = CanonicalFeatureSchema(raw_vectors=True,state_column="q",states=(
        FeatureSchema("state.angle","state",(FeatureSlice("q",2,3),),unit="deg",periodic=True),
        FeatureSchema("state.distance","state",(FeatureSlice("q",0,1),),unit="m")))
    meta = FeatureResolver().column_semantics(raw,"state",3)
    assert [m.unit for m in meta] == ["m",None,"deg"]
    view = TrajectoryView(np.array([[0,1,179],[10,1,-179]]),np.empty((2,0)),np.arange(2.),1,meta)
    delta,metrics,_ = derivative_values(view,1,columns=[2,0])
    np.testing.assert_allclose(delta,[[2,10]])
    assert [m["unit"] for m in metrics["physical_semantics"]] == ["deg","m"]


def test_raw_alias_conflicting_semantics_fail_with_context():
    raw = CanonicalFeatureSchema(raw_vectors=True,state_column="q",states=(
        FeatureSchema("state.a","state",(FeatureSlice("q",0,1),),unit="m"),
        FeatureSchema("state.b","state",(FeatureSlice("q",0,1),),unit="rad")))
    with pytest.raises(ValueError,match="Conflicting physical metadata.*column 0"):
        FeatureResolver().column_semantics(raw,"state",1)
