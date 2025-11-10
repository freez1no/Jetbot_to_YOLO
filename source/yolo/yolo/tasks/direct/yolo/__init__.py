import gymnasium as gym
from . import agents 
from .yolo_env import yoloEnv 
from .yolo_env_cfg import yoloEnvCfg
gym.register(
    id="jbtestv0", #name of the environment
    
    entry_point=f"{__name__}.yolo_env:yoloEnv",
    
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yolo_env_cfg:yoloEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml", 
    },
    disable_env_checker=True,
)