# this file rns policy that we trained 

from typing import Optional

import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController


class BipedFlatTerrainPolicy(PolicyController):

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "biped",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        policy_path: str = None,
        env_path: str = None,
    ) -> None:
      
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        if policy_path is None:
            raise ValueError("policy_path must be provided")
        if env_path is None:
            raise ValueError("env_path must be provided")

        self.load_policy(policy_path, env_path)

        self._action_scale = 0.5
        self._previous_action = np.zeros(10)  # 10 joints
        self._policy_counter = 0

       
        self.command = np.zeros(3)

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0): # send signals to the robot 
        self.command = np.array([vx, vy, wz])

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        """
        Observation structure (42 dims)
            [0:3]   base linear velocity in body frame
            [3:6]   base angular velocity in body frame  
            [6:9]   gravity direction in body frame
            [9:12]  velocity command [vx, vy, wz]
            [12:22] joint positions relative to default
            [22:32] joint velocities
            [32:42] previous actions
        Inputs
            command: velocity command [vx, vy, wz]

        What robot returns for calculations (will need some sensors)
            np.ndarray: 42-dimensional observation vector
        """
        # get velocities from simulator
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        # convert to body frame
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # build 42-dim observation
        obs = np.zeros(42, dtype=np.float32)
        obs[0:3]  = lin_vel_b
        obs[3:6]  = ang_vel_b
        obs[6:9]  = gravity_b
        obs[9:12] = command

       
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()

        obs[12:22] = current_joint_pos - self.default_pos
        obs[22:32] = current_joint_vel
        obs[32:42] = self._previous_action

        return obs

    def forward(self, dt: float, command: Optional[np.ndarray] = None):

        if command is None:
            command = self.command

        # 200 Hz
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # apply joint position targets
        target_pos = self.default_pos + self.action * self._action_scale
        action = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action)

        self._policy_counter += 1

    def initialize(self):
  
        return super().initialize(set_articulation_props=False)
