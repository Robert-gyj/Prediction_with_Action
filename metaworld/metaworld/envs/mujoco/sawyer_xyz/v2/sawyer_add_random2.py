import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
import random

class SawyerRandom2ObjectEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        
        obj_low = (-0.1, 0.85, 0.115)
        obj_high = (0.1, 0.9, 0.115)
        self.obj_low = np.array(obj_low)
        self.obj_high = np.array(obj_high)

        ######################################## all possible objects
        self.object_list = ['button','drawer','window']
        self.height_offset = {'button':0.115, 'drawer':0.0, 'window':0.16, 'button2':0.115, 'drawer2':0.0, 'window2':0.16}
        #############################################################
        self.selected_idx = np.random.choice(range(len(self.object_list)), 2, replace=True)
        self.selected_idx = [0,1]
        self.selected_objs = [self.object_list[i] for i in self.selected_idx] 
        print("selected objs", self.selected_idx, self.selected_objs)
        if self.selected_objs[0] == self.selected_objs[1]:
            self.operate_idx = 0
            self.operate_obj = self.object_list[self.selected_idx[0]]
            self.selected_objs = [self.selected_objs[0], self.selected_objs[0]+'2']
        # operated object
        else:
            self.operate_idx = np.random.choice(range(len(self.selected_idx)), 1)[0]
            self.operate_obj = self.object_list[self.selected_idx[self.operate_idx]]
        # print("operate objs", self.operate_idx, self.operate_obj)
        # for these two objects envs
        self.selected_objs_in_order = self.selected_objs.copy()
        self.selected_objs_in_order.sort()
        self.model_name = full_v2_path_for('sawyer_xyz/sawyer_add_{}_{}.xml'.format(self.selected_objs_in_order[0],self.selected_objs_in_order[1]))

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

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def _get_pos_objects(self):
        if self.operate_obj == 'button':
            pos = self.get_body_com('button') + np.array([.0, -.193, .0])
        elif self.operate_obj == 'drawer':
            pos = self.get_body_com('drawer_link') + np.array([.0, -.16, .0])
        elif self.operate_obj == 'window':
            pos = self._get_site_pos('handleOpenStart')
        else:
            raise NotImplementedError
        return pos

    def _get_quat_objects(self):
        if self.operate_obj == 'button':
            return self.sim.data.get_body_xquat('button')
        elif self.operate_obj == 'drawer':
            return self.sim.data.get_body_xquat('drawer_link')
        elif self.operate_obj == 'window':
            return np.zeros(4)
        else:
            raise NotImplementedError
    
    def _get_state_rand_vec(self):
        self._freeze_rand_vec = False
        # button_left = 1 if np.random.uniform() < 0.5 else -1
        # self.is_button = True if np.random.uniform() < 0.5 else False
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else:
            bias = [np.array([0.25, 0, 0.0]), np.array([-0.25, 0, 0.0])]
            bias = [np.array([0.0, 0, 0.0]), np.array([-0.25, 0, 0.0])] #stop shuffle
            # random.shuffle(bias) 
            # reselect the operated object
            if self.selected_objs[0]+'2' == self.selected_objs[1]:
                self.operate_idx = 0
                self.operate_obj = self.object_list[self.selected_idx[0]]
            else:
                self.operate_idx = np.random.choice(range(len(self.selected_idx)), 1)[0]
                self.operate_obj = self.object_list[self.selected_idx[self.operate_idx]]
            print("operate objs", self.operate_idx, self.operate_obj)

            rand_vec_all = []
            for i, name in enumerate(self.selected_objs):
                obj_low = np.array([-0.1, 0.85, 0])+bias[i]+np.array([0.0, 0.0, self.height_offset[name]])
                obj_high = np.array([0.1, 0.9, 0])+bias[i]+np.array([0.0, 0.0, self.height_offset[name]])
                random_reset_space = Box(obj_low, obj_high)
                rand_vec = np.random.uniform(
                    random_reset_space.low,
                    random_reset_space.high,
                    size=self._random_reset_space.low.size)
                rand_vec_all.append(rand_vec)

            self._last_rand_vec = rand_vec_all
            return rand_vec_all

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if self.random_init:
            self.init_pos = self._get_state_rand_vec()
        # print("init pos", self.init_pos)
        name_map = {'button':'box', 'drawer':'drawer', 'window':'window', 'button2':'box2', 'drawer2':'drawer2', 'window2':'window2'}
        for i, name in enumerate(self.selected_objs):
            self.sim.model.body_pos[
                self.model.body_name2id(name_map[name])] = self.init_pos[i]
            
        # set goals according to the operated obj
        self.operated_init_pos = self.init_pos[self.operate_idx]
        if self.operate_obj == 'button':
            self._set_obj_xyz(0)
            self._target_pos = self._get_site_pos('hole')
            self._obj_to_target_init = abs(
                self._target_pos[1] - self._get_site_pos('buttonStart')[1]
            )
        elif self.operate_obj == 'drawer':
            self.maxDist = 0.2
            self._target_pos = self.operated_init_pos + np.array([.0, -.16 - self.maxDist, .09])
        elif self.operate_obj == 'window':
            self._target_pos = self.operated_init_pos + np.array([.2, .0, .0])

            self.sim.model.body_pos[self.model.body_name2id(
                'window'
            )] = self.operated_init_pos
            self.window_handle_pos_init = self._get_site_pos('handleOpenStart')
            self.data.set_joint_qpos('window_slide', 0.0)
        else:
            raise NotImplementedError

        return self._get_obs()

    def get_instruction(self):
        if self.operate_obj == 'button':
            return 'press the button'
        elif self.operate_obj == 'window':
            return 'open the window'
        elif self.operate_obj == 'drawer':
            return 'open the drawer'
        else:
            raise NotImplementedError

    def compute_reward(self, actions, obs):
        if self.operate_obj == 'window':
            del actions
            obj = self._get_pos_objects()
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
        elif self.operate_obj == 'button':
            del actions
            obj = obs[4:7]
            tcp = self.tcp_center

            tcp_to_obj = np.linalg.norm(obj - tcp)
            tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
            obj_to_target = abs(self._target_pos[1] - obj[1])

            tcp_closed = max(obs[3], 0.0)
            near_button = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.05),
                margin=tcp_to_obj_init,
                sigmoid='long_tail',
            )
            button_pressed = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.005),
                margin=self._obj_to_target_init,
                sigmoid='long_tail',
            )

            reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
            if tcp_to_obj <= 0.05:
                reward += 8 * button_pressed

            return (
                reward,
                tcp_to_obj,
                obs[3],
                obj_to_target,
                near_button,
                button_pressed
            )
        elif self.operate_obj == 'drawer':
            gripper = obs[:3]
            handle = obs[4:7]

            handle_error = np.linalg.norm(handle - self._target_pos)

            reward_for_opening = reward_utils.tolerance(
                handle_error,
                bounds=(0, 0.02),
                margin=self.maxDist,
                sigmoid='long_tail'
            )

            handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
            # Emphasize XY error so that gripper is able to drop down and cage
            # handle without running into it. By doing this, we are assuming
            # that the reward in the Z direction is small enough that the agent
            # will be willing to explore raising a finger above the handle, hook it,
            # and drop back down to re-gain Z reward
            scale = np.array([3., 3., 1.])
            gripper_error = (handle - gripper) * scale
            gripper_error_init = (handle_pos_init - self.init_tcp) * scale

            reward_for_caging = reward_utils.tolerance(
                np.linalg.norm(gripper_error),
                bounds=(0, 0.01),
                margin=np.linalg.norm(gripper_error_init),
                sigmoid='long_tail'
            )

            reward = reward_for_caging + reward_for_opening
            # reward *= 5.0
            reward *= 2.0

            return (
                reward,
                np.linalg.norm(handle - gripper),
                obs[3],
                handle_error,
                reward_for_caging,
                reward_for_opening
            )
        else:
            raise NotImplementedError

