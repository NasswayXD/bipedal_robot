from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from my_biped.my_biped_robot_cfg import MY_BIPED_CFG


@configclass
class MyBipedRewards(RewardsCfg): # revards and penalties
   
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )

   
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)  
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)  
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-07)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02) 

   
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle.*")}
    )

    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_one_.*"])}
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "hip_leg_.*", "knee_leg_.*"  
        ])}
    )
    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ankle_leg_.*"])}
    )

    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,   
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_leg.*"),
            "threshold": 0.3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_leg.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_leg.*"),
        },
    )
    undesired_contacts = None
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class MyBipedEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: MyBipedRewards = MyBipedRewards()

    def __post_init__(self):
        super().__post_init__()

        
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        self.scene.robot = MY_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

       
        self.observations.policy.enable_corruption = True
        self.observations.policy.height_scan = None

        
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (-0.1, 0.1), 
                "operation": "add",
            },
        )
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)  
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={"velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
            }},
        )

    
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)

       # kill the robot if it falls :(
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "hip_one_right", "hip_two_right", "hip_leg_right", "knee_leg_right",
            "hip_one_left",  "hip_two_left",  "hip_leg_left",  "knee_leg_left",
        ]