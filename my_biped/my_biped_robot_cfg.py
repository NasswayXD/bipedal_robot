import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

MY_BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/nassway/Documents/final_test/Assem2/urdf/Assem2.urdf", #model of the robot
        fix_base=False,
        force_usd_conversion=True,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg( # becasue we use force and position 
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,   #the gains og the joints 
                damping=2.0,      
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, # terminate the robot if colides with itself, need to avoid leg crossing 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg( # initial pose and how hight does it spawn
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            "hip_one_right_joint":   0.0,
            "hip_two_right_joint":   0.0,
            "hip_leg_right_joint":   -0.4,
            "knee_leg_right_joint":  0.95,
            "ankle_leg_right_joint": -0.55,
            "hip_one_left_joint":    0.0,
            "hip_two_left_joint":    0.0,
            "hip_leg_left_joint":    0.4,
            "knee_leg_left_joint":   -0.95,
            "ankle_leg_left_joint":  0.55,
        },
    ),
    actuators={ #again the ankles are softer for smoother walk, all the stiffness and damping values are estimated 
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_one_right_joint", "hip_two_right_joint",
                "hip_leg_right_joint", "knee_leg_right_joint",
                "hip_one_left_joint",  "hip_two_left_joint",
                "hip_leg_left_joint",  "knee_leg_left_joint",
            ],
            stiffness=40.0,
            damping=2.0,
            effort_limit=2.0,
            velocity_limit=6.28,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_leg_right_joint",
                "ankle_leg_left_joint",
            ],
            stiffness=20.0,   
            damping=2.0,
            effort_limit=2.0,
            velocity_limit=6.28,
        ),
    },
)