import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDrawerWindowEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        
        obj_low = (-0.1, 0.85, 0.115)
        obj_high = (0.1, 0.9, 0.115)
        self.obj_low = np.array(obj_low)
        self.obj_high = np.array(obj_high)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0., 0.9, 0.115], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.78, 0.12])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_add_drawer_window.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('btnGeom')

    def _get_pos_objects(self):
        drawer_pos = self.get_body_com('drawer_link') + np.array([.0, -.16, .0])
        window_pos = self._get_site_pos('handleOpenStart')
        if not hasattr(self, 'is_button'):
            return np.concatenate((drawer_pos,window_pos))
        pos_all = np.concatenate((drawer_pos,window_pos)) if (self.is_button) else np.concatenate((window_pos, drawer_pos))
        return pos_all

    def _get_quat_objects(self):
        if not hasattr(self, 'is_button'):
            return np.concatenate((self.sim.data.get_body_xquat('drawer_link'),np.zeros(4)))
        quat_all = np.concatenate((self.sim.data.get_body_xquat('drawer_link'),np.zeros(4))) if (self.is_button) else np.concatenate((np.zeros(4),self.sim.data.get_body_xquat('window')))
        return quat_all

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)
    
    def _get_state_rand_vec(self):
        self._freeze_rand_vec = False
        button_left = 1 if np.random.uniform() < 0.5 else -1
        self.is_button = True if np.random.uniform() < 0.5 else False
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else:
            # drawer
            obj_low1 = np.array([-0.1, 0.9, 0.0])+np.array([0.25, 0, 0.0])*button_left
            obj_high1 = np.array([0.1, 0.9, 0.0])+np.array([0.25, 0, 0.0])*button_left
            # drawer
            obj_low2 = np.array([-0.1, 0.7, 0.16])+np.array([-0.25, 0, 0.0])*button_left
            obj_high2 = np.array([0.1, 0.9, 0.16])+np.array([-0.25, 0, 0.0])*button_left
            random_reset_space1 = Box(obj_low1, obj_high1)
            random_reset_space2 = Box(obj_low2,obj_high2)
            rand_vec1 = np.random.uniform(
                random_reset_space1.low,
                random_reset_space1.high,
                size=self._random_reset_space.low.size)
            rand_vec2 = np.random.uniform(
                random_reset_space2.low,
                random_reset_space2.high,
                size=self._random_reset_space.low.size)
            rand_vec = [rand_vec1,rand_vec2]
            self._last_rand_vec = rand_vec
            return rand_vec

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos[0]
            self.obj_init_pos2 = goal_pos[1]

        self.sim.model.body_pos[
            self.model.body_name2id('drawer')] = self.obj_init_pos
        self.sim.model.body_pos[
            self.model.body_name2id('window')] = self.obj_init_pos2
        self._set_obj_xyz(0)

        if self.is_button:
            self.maxDist = 0.2
            self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])
            self.window_handle_pos_init = self._get_site_pos('handleOpenStart')
        else:
            self._target_pos = self.obj_init_pos2 + np.array([.2, .0, .0])

            self.sim.model.body_pos[self.model.body_name2id(
                'window'
            )] = self.obj_init_pos2
            self.window_handle_pos_init = self._get_site_pos('handleOpenStart')
            self.data.set_joint_qpos('window_slide', 0.0)

        return self._get_obs()

    def get_instruction(self):
        if self.is_button:
            return 'open the drawer'
        else:
            return 'open the window'

    def compute_reward(self, actions, obs):
        del actions
        obj = self._get_site_pos('handleOpenStart')
        tcp = self.tcp_center
        target = self._target_pos.copy()
        
        target_to_obj = (obj[0] - target[0])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos[0] - target[0])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.window_handle_pos_init - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        reward = 10 * reward_utils.hamacher_product(reach, in_place)
        return (reward,
               tcp_to_obj,
               tcp_opened,
               target_to_obj,
               object_grasped,
               in_place)
