"""Environment factory: turns a validated ScenarioSpec into an OmniGibson env.

Environment variables that OmniGibson reads at import time are applied here,
before the first omnigibson import, so the rest of the package can just
import it.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import yaml

from rummagebench.core.errors import SimBackendError
from rummagebench.core.scenario import ScenarioSpec


def repo_root() -> Path:
    """Locate the RummageBench repo root (contains configs/ and scenarios/)."""
    override = os.environ.get("RUMMAGEBENCH_ROOT")
    if override:
        return Path(override)
    # src/rummagebench/sim/omnigibson/env_factory.py -> repo root is 5 levels up
    return Path(__file__).resolve().parents[4]


def apply_runtime_env(gpu_id: int | None = None) -> None:
    """Set OmniGibson import-time environment knobs.

    Must be called BEFORE `import omnigibson` anywhere in the process.
    """
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
    if gpu_id is not None:
        os.environ["OMNIGIBSON_GPU_ID"] = str(gpu_id)

NOTE_ON_SLUDGE_SEGFAULT = (
    "Known host-level instability (2026-09-28): after another user's IsaacLab "
    "workload saturates the GPUs, Beechwood scene import fails with "
    "'Failed to initialize sludge system' (a PhysX particle-system "
    "allocation) followed by a hard segfault during Kit cleanup — on every "
    "GPU, including previously-idle ones. The same code loads fine when the "
    "host is quiet. Not fixable benchmark-side; recover by re-running when "
    "the competing workload finishes (or a driver reset). Do NOT enable "
    "gm.USE_GPU_DYNAMICS as a workaround: it segfaults scene import too."
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_config() -> dict:
    return _load_yaml(repo_root() / "configs" / "simulator" / "omnigibson.yaml")


def load_robot_config(model: str) -> dict:
    path = repo_root() / "configs" / "robots" / f"{model.lower()}_symbolic.yaml"
    if not path.exists():
        raise SimBackendError(f"no robot config for {model!r} at {path}")
    return _load_yaml(path)


def build_env_config(scenario: ScenarioSpec) -> dict:
    """Merge the simulator template, the robot config, and the scenario."""
    config = deepcopy(load_env_config())
    if scenario.scene.source == "infinigen":
        from rummagebench.sim.omnigibson.infinigen_adapter import assembly_env_config

        manifest_path = Path(scenario.scene.assembly_manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
        config.update(assembly_env_config(manifest_path))
    elif scenario.scene.source == "behavior_layout":
        from rummagebench.sim.omnigibson.behavior_adapter import assembly_env_config

        manifest_path = Path(scenario.scene.assembly_manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
        config.update(assembly_env_config(manifest_path, config["scene"], scenario.scene.model))
    else:
        config["scene"]["scene_model"] = scenario.scene.model

    robot_cfg = deepcopy(load_robot_config(scenario.robot.model))
    robot_cfg["model"] = scenario.robot.model
    robot_cfg["name"] = scenario.robot.name
    # Renderer instance segmentation is supplied post-setup by the raw
    # instance_id_segmentation_fast capture (raw_instance.py) on the same
    # render product as RGB. OG's seg_instance obs chain depends on
    # seg_semantic, which crashes this host's renderer graph (native 139),
    # so those modalities must never reach the OG build config.
    from rummagebench.sim.omnigibson.raw_instance import GROUNDING_MODALITIES

    robot_cfg["obs_modalities"] = [
        m for m in scenario.robot.obs_modalities if m not in GROUNDING_MODALITIES
    ]
    robot_cfg["grasping_mode"] = scenario.robot.grasping_mode
    if scenario.robot.include_sensor_names is not None:
        robot_cfg["include_sensor_names"] = list(scenario.robot.include_sensor_names)
    if scenario.robot.exclude_sensor_names is not None:
        robot_cfg["exclude_sensor_names"] = list(scenario.robot.exclude_sensor_names)
    sensor_kwargs = robot_cfg.setdefault("sensor_config", {}).setdefault(
        "VisionSensor", {}
    ).setdefault("sensor_kwargs", {})
    sensor_kwargs.update(
        {
            "image_height": scenario.robot.image_height,
            "image_width": scenario.robot.image_width,
        }
    )
    if scenario.robot.focal_length_mm is not None:
        sensor_kwargs["focal_length"] = scenario.robot.focal_length_mm
    # Robot initial pose comes from the scenario's init anchor; the backend
    # teleports via the same anchor mechanism used by NAV to keep one code path.
    if getattr(scenario, "behavior_task", None) is None:
        anchor = scenario.anchors[scenario.robot.init_anchor]
        robot_cfg["position"] = list(anchor.position)
        robot_cfg["orientation"] = list(anchor.orientation)
        robot_cfg["pose_frame"] = "world"
    config["robots"] = [robot_cfg]
    if getattr(scenario, "behavior_task", None) is not None:
        from rummagebench.datasets.behavior import assert_rigid_task
        assert_rigid_task(scenario.behavior_task)
        config["task"] = {
            "type": "BehaviorTask",
            **scenario.behavior_task.model_dump(exclude={"bddl_root"}),
            "online_object_sampling": False,
            "use_presampled_robot_pose": True,
            "highlight_task_relevant_objects": False,
        }
        # The task loads the official cached scene and robot pose.
        for key in ("position", "orientation", "pose_frame"):
            robot_cfg.pop(key, None)
    return config


def build_env(scenario: ScenarioSpec):
    """Create the OmniGibson Environment (launches the simulator on first call)."""
    import omnigibson as og
    from omnigibson.envs import Environment
    from omnigibson.macros import gm

    # Execution is symbolic: objects must never transform on their own.
    # OmniGibson's default transition rules (slicing / mixing / cooking
    # recipes) fire on contact — e.g. a carving knife resting in a drawer next
    # to a bowl — and spawn a "sludge" particle system, which on this host
    # (no GPU dynamics) aborts the scene build.
    gm.ENABLE_TRANSITION_RULES = False
    if scenario.robot.panorama is not None:
        # The human UI and agents both use the head panorama. Do not create
        # an additional, unused spectator render product in headless runs.
        gm.RENDER_VIEWER_CAMERA = False

    if scenario.scene.source == 'behavior_layout':
        from omnigibson.utils.asset_utils import get_dataset_path
        from rummagebench.sim.omnigibson.behavior_adapter import load_manifest
        manifest_path = Path(scenario.scene.assembly_manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
        manifest = load_manifest(manifest_path)
        runtime_root = Path(get_dataset_path('behavior-1k-assets')).resolve()
        for asset in manifest['assets'].values():
            for source in asset['sources']:
                if not Path(source).resolve().is_relative_to(runtime_root):
                    raise SimBackendError('native layout and OmniGibson must use the same BEHAVIOR dataset')

    if scenario.scene.source == 'infinigen' and og.sim is None:
        # In Isaac's environment pxr is provided by the running Kit app.
        # Reading assembly USD dependencies before Environment construction
        # otherwise fails before the first simulator launch.
        og.launch()

    config = build_env_config(scenario)
    if getattr(scenario, "behavior_task", None) is not None:
        import bddl
        from omnigibson.utils.asset_utils import get_dataset_path
        spec = scenario.behavior_task
        definition = Path("activity_definitions") / spec.activity_name / f"problem{spec.activity_definition_id}.bddl"
        if (Path(bddl.__file__).parent / definition).read_bytes() != (Path(spec.bddl_root) / definition).read_bytes():
            raise SimBackendError("installed BDDL definition differs from audited source")
        from omnigibson.utils.asset_utils import get_task_instance_path
        instance = f"{scenario.scene.model}_task_{spec.activity_name}_{spec.activity_definition_id}_{spec.activity_instance_id}_template"
        template = get_task_instance_path(scenario.scene.model, instance)
        if template is None or not Path(template).is_file():
            raise SimBackendError(f"official pre-sampled task instance missing: {instance}")
    if scenario.scene.source == 'infinigen':
        from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, restore_source_static_colliders
        manifest_path = Path(scenario.scene.assembly_manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
        manifest = load_assembly_manifest(manifest_path)
        expected = {item['name'] for item in manifest['objects'] if item['role']=='static_background'}
        evidence = []
        def source_collider_import(obj):
            result = restore_source_static_colliders(obj, expected)
            if result is not None: evidence.append(result)
        hook = 'findingbench_infinigen_source_triangle_colliders'
        og.sim.add_callback_on_add_obj(hook,source_collider_import)
        try:
            env = Environment(configs=config)
        finally:
            og.sim.remove_callback_on_add_obj(hook)
        if {e['entity'] for e in evidence} != expected:
            raise ValueError('SOURCE_STATIC_COLLIDER_IMPORT_COVERAGE_INCOMPLETE')
        env._findingbench_source_collider_import = evidence
    else:
        env = Environment(configs=config)
    if scenario.scene.source == 'behavior_layout' and manifest.get('living_context',{}).get('geometry'):
        from rummagebench.authoring.layout.indoor_geometry import restore_native_state
        env._findingbench_geometry_restore = restore_native_state(env,manifest)
    if scenario.scene.source == "infinigen":
        from rummagebench.sim.omnigibson.infinigen_adapter import install_runtime_lighting

        install_runtime_lighting(env, scenario.scene.assembly_manifest)
    return env, og
