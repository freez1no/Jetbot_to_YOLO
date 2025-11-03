from __future__ import annotations
from collections.abc import Sequence
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim.spawners import spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera 
from ultralytics import YOLO
from .yolo_env_cfg import yoloEnvCfg

class yoloEnv(DirectRLEnv):
    cfg: yoloEnvCfg
    def __init__(self, cfg: yoloEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(f"Loading YOLOv11 model 'yolo11n.pt' on device {self.device}...")
        self.yolo_model = YOLO("yolo11n.pt") 
        self.yolo_model.to(self.device)
        print("YOLO model loaded successfully.")
        self.bbox_obs = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.target = RigidObject(self.cfg.target_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())

        self.cfg.camera_cfg.num_envs = self.cfg.scene.num_envs
        camera_sensor = Camera(self.cfg.camera_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target 
        self.scene.sensors["camera"] = camera_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # 1. 카메라 이미지 가져오기 [B, H, W, 4] (RGBA, uint8)
        image_data_rgba = self.scene.sensors["camera"].data.output["rgb"]

        # 2. 이미지 전처리 [B, H, W, 3] (BHWC, uint8)
        images_rgb_nhwc = image_data_rgba[..., :3]

        # 3. 이미지 포맷 변환 및 정규화
        #    (BHWC -> BCHW)
        images_rgb_bchw_uint8 = images_rgb_nhwc.permute(0, 3, 1, 2)

        #    (수정) uint8 (0-255) -> float32 (0.0-1.0)
        #    YOLO 모델은 0.0~1.0 범위의 float 텐서를 입력으로 받습니다.
        images_rgb_bchw_float = images_rgb_bchw_uint8.float() / 255.0
        # --- (수정 끝) ---

        # 4. YOLO V11 추론 (Inference)
        #    (입력을 float 텐서로 변경)
        results = self.yolo_model(
            images_rgb_bchw_float, # 👈 수정됨
            classes=[32],         # sports ball
            verbose=False, 
            device=self.device
        )

        # 5. 결과 후처리 (변경 없음)
        obs_tensor = torch.zeros((self.num_envs, 4), device=self.device)
        for i, res in enumerate(results):
            if res.boxes.shape[0] > 0:
                obs_tensor[i] = res.boxes.xywhn[0]

        # 6. 결과 저장 및 반환 (변경 없음)
        self.bbox_obs = obs_tensor
        return {"policy": self.bbox_obs}
    def _get_rewards(self) -> torch.Tensor:
        x_norm, _, w_norm, h_norm = torch.split(self.bbox_obs, 1, dim=1)
        x_norm = x_norm.squeeze()
        w_norm = w_norm.squeeze()
        h_norm = h_norm.squeeze()
        found_target = (w_norm > 0).float()
        center_error = torch.abs(x_norm - 0.5)
        centering_reward = (0.5 - center_error) * 2.0 * found_target
        distance_reward = h_norm * found_target
        time_penalty = -0.01
        total_reward = (centering_reward * 1.0) + \
                       (distance_reward * 0.5) + \
                       time_penalty
        return total_reward.reshape(-1, 1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        dones = torch.zeros_like(time_out)
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids) 
        if env_ids is None:
            reset_env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            reset_env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            reset_env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_resets = len(reset_env_ids_tensor)
        robot_default_state = self.robot.data.default_root_state[reset_env_ids_tensor]
        robot_default_state[:, :3] += self.scene.env_origins[reset_env_ids_tensor]
        self.robot.write_root_state_to_sim(robot_default_state, reset_env_ids_tensor)
        default_joint_pos = self.robot.data.default_joint_pos[reset_env_ids_tensor]
        default_joint_vel = self.robot.data.default_joint_vel[reset_env_ids_tensor]
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, reset_env_ids_tensor)
        target_new_state = self.target.data.default_root_state[reset_env_ids_tensor].clone()
        target_new_state[:, :3] += self.scene.env_origins[reset_env_ids_tensor]

        rand_x = torch.rand(num_resets, device=self.device) * 2.0 + 1.0
        rand_y = (torch.rand(num_resets, device=self.device) * 4.0 - 2.0)
        
        target_new_state[:, 0] = self.scene.env_origins[reset_env_ids_tensor, 0] + rand_x
        target_new_state[:, 1] = self.scene.env_origins[reset_env_ids_tensor, 1] + rand_y
        target_new_state[:, 2] = self.scene.env_origins[reset_env_ids_tensor, 2] + 0.25 # 공 반지름만큼 살짝 띄우기

        target_new_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        target_new_state[:, 7:13] = 0.0 # 속도

        self.target.write_root_state_to_sim(target_new_state, reset_env_ids_tensor) 

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        scaled_actions = self.actions * 2.5 
        self.robot.set_joint_velocity_target(scaled_actions, joint_ids=self.dof_idx)