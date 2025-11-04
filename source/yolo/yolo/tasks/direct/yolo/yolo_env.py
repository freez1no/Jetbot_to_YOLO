# 파이썬 기본 라이브러리
from __future__ import annotations
from collections.abc import Sequence

# Torch (파이토치)
import torch

# Isaac Lab 핵심 라이브러리
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim.spawners import spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera 

# YOLO v11 라이브러리
from ultralytics import YOLO

# 우리 환경 설정 파일
from .yolo_env_cfg import yoloEnvCfg


class yoloEnv(DirectRLEnv):
    """Jetbot YOLO RL 환경 클래스 (겹침 방지 및 보상 로직 수정)."""

    cfg: yoloEnvCfg

    def __init__(self, cfg: yoloEnvCfg, **kwargs):
        """
        환경을 초기화하고 YOLO 모델을 로드합니다.
        """
        super().__init__(cfg, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        
        print(f"Loading YOLOv11 model 'yolo11n.pt' on device {self.device}...")
        self.yolo_model = YOLO("yolo11n.pt") 
        self.yolo_model.to(self.device)
        print("YOLO model loaded successfully.")
        
        self.bbox_obs = torch.zeros((self.num_envs, 4), device=self.device)
        
        # (신규) 오차(Error)가 커지는 것을 감지하기 위해 이전 BBox 높이를 저장할 텐서
        self.prev_h_norm = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """
        시뮬레이션 씬(Scene)을 설정합니다.
        """
        self.robot = Articulation(self.cfg.robot_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())

        # (중요) 카메라의 num_envs를 환경의 num_envs와 동기화
        self.cfg.camera_cfg.num_envs = self.cfg.scene.num_envs
        camera_sensor = Camera(self.cfg.camera_cfg)

        self.scene.clone_environments(copy_from_source=False)

        # 씬(Scene)에 에셋 등록
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target 
        self.scene.sensors["camera"] = camera_sensor # 카메라 등록

        # 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        """
        카메라 이미지를 YOLO로 추론 (BCHW, float32 포맷)
        """
        image_data_rgba = self.scene.sensors["camera"].data.output["rgb"]
        images_rgb_nhwc = image_data_rgba[..., :3]
        images_rgb_bchw_float = images_rgb_nhwc.permute(0, 3, 1, 2).float() / 255.0

        results = self.yolo_model(
            images_rgb_bchw_float, 
            classes=[32], # sports ball
            verbose=False, 
            device=self.device
        )

        obs_tensor = torch.zeros((self.num_envs, 4), device=self.device)
        for i, res in enumerate(results):
            if res.boxes.shape[0] > 0:
                obs_tensor[i] = res.boxes.xywhn[0]
        
        self.bbox_obs = obs_tensor
        return {"policy": self.bbox_obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        (수정됨) 보상 함수:
        - "오차가 커지면" (타겟이 멀어지면) 패널티
        - "너무 가까우면" (겹치면) 패널티
        - "적당히 가까이 가면" 보상
        """
        x_norm, _, w_norm, h_norm = torch.split(self.bbox_obs, 1, dim=-1)

        x_norm = x_norm.squeeze(-1) # [B]
        w_norm = w_norm.squeeze(-1) # [B]
        h_norm = h_norm.squeeze(-1) # [B]

        # --- 보상 항목 계산 ---
        found_target = (w_norm > 0).float()

        # 1. 중앙 정렬 보상 (0.0 ~ 1.0)
        center_error = torch.abs(x_norm - 0.5)
        centering_reward = (0.5 - center_error) * 2.0 * found_target

        # 2. 거리 보상 (최적 거리: 0.5, 0.0 ~ 1.0)
        optimal_h = 0.5
        distance_error_sq = (h_norm - optimal_h) ** 2
        distance_reward = torch.exp(-20.0 * distance_error_sq) * found_target

        # 3. (신규) 오차가 커지는 것(멀어지는 것)에 대한 패널티
        #    이전 스텝보다 BBox 높이(h_norm)가 작아졌는지 확인
        delta_h = h_norm - self.prev_h_norm
        #    멀어졌을 때(delta_h < 0)만 패널티를 줌 (예: -1.0 * delta)
        worsening_penalty = torch.clamp(delta_h, max=0.0) * -1.0 * found_target
        
        # 4. 근접 패널티 (겹침 방지)
        #    BBox 높이가 80% (0.8)를 초과하면 -2.0 패널티
        too_close_penalty = (h_norm > 0.8).float() * -2.0

        # --- 총 보상 계산 ---
        total_reward = (centering_reward * 1.0) + \
                       (distance_reward * 1.5) + \
                       (worsening_penalty * 0.5) + \
                       too_close_penalty
        
        # --- 현재 상태 저장 ---
        # 다음 스텝에서 비교하기 위해 현재 BBox 높이를 저장 (detach()로 그래디언트 흐름 차단)
        self.prev_h_norm = h_norm.detach()
        
        return total_reward.reshape(-1, 1) # [B, 1] 형태로 반환

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (수정됨) 에피소드 종료 조건: 3초 타임아웃만 적용
        """
        # 1. 타임아웃: 3초가 지나면 타임아웃
        time_out = self.episode_length_buf >= self.max_episode_length

        # 2. (수정) "오차가 커지면" 리셋하는 대신, 
        #    보상 함수가 패널티를 주도록 하고 에피소드는 3초간 계속 진행합니다.
        dones = torch.zeros_like(time_out)
        
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 1. 부모 클래스의 reset을 *가장 먼저* 호출
        super()._reset_idx(env_ids) 

        # 2. 리셋할 텐서 ID 생성
        if env_ids is None:
            reset_env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            reset_env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            reset_env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            
        num_resets = len(reset_env_ids_tensor)
        
        # --- 3. Jetbot 리셋: 환경 원점 [0, 0, 0.05] 위치에 고정 ---
        robot_new_state = torch.zeros((num_resets, self.robot.data.default_root_state.shape[1]), device=self.device)
        robot_new_state[:, 2] = 0.05 # 지면에서 5cm 띄우기
        robot_new_state[:, 3] = 1.0  # 기본 방향 (W=1)
        # 각 Jetbot을 고유한 환경 원점(origin)으로 이동
        robot_new_state[:, :3] += self.scene.env_origins[reset_env_ids_tensor]
        
        # 시뮬레이션에 상태 쓰기
        self.robot.write_root_state_to_sim(robot_new_state, reset_env_ids_tensor)
        
        # Jetbot 관절은 0으로 리셋
        default_joint_pos = self.robot.data.default_joint_pos[reset_env_ids_tensor] * 0.0
        default_joint_vel = self.robot.data.default_joint_vel[reset_env_ids_tensor] * 0.0
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, reset_env_ids_tensor)

        # --- 4. (수정) Target 리셋: Jetbot 중심 20cm 원 둘레 랜덤 위치 ---
        target_new_state = torch.zeros((num_resets, self.target.data.default_root_state.shape[1]), device=self.device)
        
        # 0 ~ 2*pi (360도) 사이의 랜덤한 각도(라디안) 생성 [num_resets]
        random_angle = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        
        # Jetbot으로부터의 상대 위치 계산
        distance = 0.20 # 20cm
        rel_x = torch.cos(random_angle) * distance
        rel_y = torch.sin(random_angle) * distance
        
        # Jetbot의 위치(robot_new_state)에 상대 위치를 더함
        target_new_state[:, 0] = robot_new_state[:, 0] + rel_x
        target_new_state[:, 1] = robot_new_state[:, 1] + rel_y
        target_new_state[:, 2] = robot_new_state[:, 2] + 0.20 # Jetbot보다 20cm 위 (공 반지름 0.15 + Jetbot 높이 0.05)

        # 기본 방향 (W=1)
        target_new_state[:, 3] = 1.0 
        # 속도는 0
        target_new_state[:, 7:13] = 0.0 

        # 시뮬레이션에 상태 쓰기
        self.target.write_root_state_to_sim(target_new_state, reset_env_ids_tensor)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        scaled_actions = self.actions * 2.5 
        self.robot.set_joint_velocity_target(scaled_actions, joint_ids=self.dof_idx)