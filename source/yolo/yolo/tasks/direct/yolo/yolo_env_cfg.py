import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from yolo.robots.jetbot import JETBOT_CONFIG
from isaaclab.sensors import CameraCfg

@configclass
class yoloEnvCfg(DirectRLEnvCfg):
    """Jetbot YOLO 프로젝트 환경 설정 클래스 (수정된 버전)."""

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, #한번에 생성 될 훈련 에이전트 수
        env_spacing=4.0,
        replicate_physics=True,
    )

    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target", # 타겟이 생성될 경로
        spawn=sim_utils.SphereCfg(
            radius=0.25, # 반지름 25cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0) # RGB
            ),
        ),
    )
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/camera_sensor",
        update_period=0.02,
        height=224,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.1), 
            rot=(0.0, 0.0, 0.0, 1.0)
        )
    )

    action_space: int = 2
    observation_space: int = 4
    state_space: int = 0
    dof_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]
    
    # 에피소드 설정
    decimation: int = 2
    episode_length_s: float = 10.0