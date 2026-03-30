
import carb
import numpy as np
import omni
import omni.appwindow
import omni.timeline
import asyncio
import sys

sys.path.insert(0, "/home/nassway/Documents/biped_policy/")
from biped_policy import BipedFlatTerrainPolicy
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api import World

# all the files that we need 
POLICY_PATH   = "/home/nassway/IsaacLab/logs/rsl_rl/my_biped_flat/2026-03-29_20-19-01/exported/policy.pt" # what we trained 
ENV_PATH      = "/home/nassway/Documents/biped_policy/biped_env.yaml" # env for the sim 
URDF_USD_PATH = "/home/nassway/Documents/final_test/Assem2/urdf/Assem2/Assem2.usd" # model path



class BipedExample(BaseSample):
    def __init__(self) -> None: # initial conditions 
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0
        self._world_settings["rendering_dt"] = 8.0 / 200.0
        self._base_command = np.array([0.0, 0.0, 0.0])
        self._input_keyboard_mapping = {
            "NUMPAD_8": [0.75, 0.0, 0.0],
            "UP":       [0.75, 0.0, 0.0],
            "NUMPAD_2": [-0.75, 0.0, 0.0],
            "DOWN":     [-0.75, 0.0, 0.0],
            "NUMPAD_4": [0.0, 0.0, 0.75],
            "LEFT":     [0.0, 0.0, 0.75],
            "NUMPAD_6": [0.0, 0.0, -0.75],
            "RIGHT":    [0.0, 0.0, -0.75],
        }
        self._init_steps = 0  # count steps before starting policy

    def setup_scene(self) -> None: # how the scene looks like and what we have in it 
        self.get_world().scene.add_default_ground_plane( # ground to stand on
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ) 
        self.biped = BipedFlatTerrainPolicy( # the robt 
            prim_path="/World/Biped",
            name="Biped",
            usd_path=URDF_USD_PATH,
            position=np.array([0.0, 0.0, 0.45]),
            policy_path=POLICY_PATH,
            env_path=ENV_PATH,
        )
        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events( # that is how it is controled 
            self._keyboard, self._sub_keyboard_event
        )
        self._physics_ready = False
        self._init_steps = 0
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        self._init_steps = 0
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    def on_physics_step(self, step_size) -> None:
        if not self._physics_ready:
            self._init_steps += 1
            if self._init_steps == 1:
                # First step: initialize the robot
                try:
                    self.biped.initialize()
                    self.biped.post_reset()
                    self.biped.robot.set_joints_default_state(self.biped.default_pos)
                except Exception as e:
                    print(f"Init error: {e}")
                    return
            elif self._init_steps >= 3:
                # Wait 3 steps for physics to settle, then start policy
                self._physics_ready = True
                print("Policy running! Use arrow keys to control.")
        else:
            self.biped.forward(step_size, self._base_command)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.biped:
            self._physics_ready = False
            self._init_steps = 0
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")


async def main():
    example = BipedExample()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/200.0,
        rendering_dt=8.0/200.0
    )
    example._world = world
    example.setup_scene()
    await world.initialize_simulation_context_async()
    await example.setup_post_load()
    print("=" * 50)
    print("Biped loaded! Arrow keys to control.")
    print("UP=forward DOWN=back LEFT=turn left RIGHT=turn right")
    print("=" * 50)

asyncio.ensure_future(main())