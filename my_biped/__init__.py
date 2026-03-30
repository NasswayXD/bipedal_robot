import gymnasium as gym

gym.register(
    id="MyBiped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # Pass the paths as strings: "module.path:ClassName"
        "env_cfg_entry_point": "my_biped.my_biped_env_cfg:MyBipedEnvCfg",
        "rsl_rl_cfg_entry_point": "my_biped.agents.rsl_rl_ppo_cfg:MyBipedFlatPPORunnerCfg",
    },
    disable_env_checker=True,
)