
import mediapy
import numpy as np
from PIL import Image
import json
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2

os.environ["MUJOCO_GL"] = "egl"
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from evaluation.agent import DiffusionAgent
from evaluation.run_cfg import INSTRUCTIONS, CONFIG 

def add_bound(rgb,color="red"):
    width=10
    c = 0 if color=="red" else 1
    rgb[:width,:,1:3]=100
    rgb[:width,:,c]=255
    
    rgb[-width:,:,1:3]=100
    rgb[-width:,:,c]=255
    
    rgb[:,:width,1:3]=100
    rgb[:,:width,c]=255
    rgb[:,-width:,1:3]=100
    rgb[:,-width:,c]=255
    return rgb

def merge_img(obs, predict,img_word):
    img_1 = cv2.resize(obs, (256,256), interpolation=cv2.INTER_AREA)
    image = np.concatenate((add_bound(img_1,color="green"), predict),axis=1)
    image = np.concatenate((img_word,image),axis=0)
    return image

def plot_word():
    from PIL import Image, ImageDraw, ImageFont
    fnt_titile = ImageFont.truetype("evaluation/TIMES.ttf", int(600/20))
    img_word = Image.new('RGB', (256*4,40), color = 'white')
    draw = ImageDraw.Draw(img_word)
    task = "Observations"
    task2 = "Predictions"
    draw.text((50,10), task, font=fnt_titile, fill='green')
    draw.text((572,10), task2, font=fnt_titile, fill='red')

    return img_word

# motion planner for metaworld tasks
def motion_planner(target_xyz, target_gripper, curr_xyz, curr_gripper, env, image_3, thirdview, predict_img=None, img_word=None):
    # a simple motion planner to reach the target pose, starting from the current pose
    # stage (0) Move to the target pose with a constant velocity (0.6 or 0.3)
    # stage (1) If grasp, then close the gripper
    stage = 0 
    grasp_moment = False

    # check whether the gripper should closed
    if np.abs(target_gripper - curr_gripper) > 0.2:
        # target_xyz[2] -=0.04 if target_xyz[2]>0.1 else 0.01
        grasp_moment = True
        print("prepare grasp!!")
    
    # start motion planner with max 50 steps
    motion_steps = 50
    for i in range(motion_steps):                
        a = -np.ones(4)
        if stage == 0: # moving to target pose with a constant velocity
            if target_gripper < 0.75 and curr_gripper < 0.75:
                a[3] = 0.7
            velocity = 0.6 if np.linalg.norm(target_xyz-curr_xyz) > 0.03 else 0.3
            a[:3] = (target_xyz-curr_xyz)/np.linalg.norm(target_xyz-curr_xyz)*velocity

            # step the env
            obs, r, done, info = env.step(a)
            img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224],depth=False)
            if predict_img is not None:
                img_all= merge_img(img,predict_img,img_word)
            image_3.append(img_all)
            curr_xyz,curr_gripper = env._get_obs()[:3], env._get_obs()[3]

            # check if the target pose is reached
            if stage==0 and (np.linalg.norm(target_xyz-curr_xyz) < 0.005 or curr_xyz[2]<0.05):
                stage += 1 if grasp_moment else motion_steps

        elif stage < 20: # grasping stage
            if target_gripper < 0.82:
                a = np.array([0,0,0,0.7]) # close the gripper
                obs, r, done, info = env.step(a)
                img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224],depth=False)
                if predict_img is not None:
                    img_all= merge_img(img,predict_img,img_word)
                image_3.append(img_all)
                stage += 1
            else:
                break
        else:
            break
    return info,img


# rollout tasks
task_list = [key for key in INSTRUCTIONS.keys()]
task_list = CONFIG['task_list']
success_num = np.zeros(len(task_list))
thirdview = CONFIG['thirdview_camera']
firstview = CONFIG['firstview_camera']
ckpt_path = CONFIG['ckpt_path']
use_depth = CONFIG['use_depth']

# build agent
agent = DiffusionAgent(ckpt_path=ckpt_path,vae_path=CONFIG['vae_path'], clip_path=CONFIG['clip_path'], denoise_steps=CONFIG['denoise_steps'])
if CONFIG['visualize_prediction']:
    img_word = plot_word()
else:
    img_word = None
    predict_img = None

# start rollout
for selected_id, task in enumerate(task_list):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+"-goal-observable"]
    env = env_cls(seed=selected_id+100)

    for traj_idx in range(CONFIG['rollout_num']):
        print("task name", task, 'traj_idx', traj_idx)
        image_3 = []

        obs = env.reset()
        img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224],depth=False)
        # image_3.append(img)
        for plan_step in tqdm(range(CONFIG['max_steps'])):
            # prepare input data
            grasp_moment = False
            state = obs
            rgb = img
            depth =  depth if use_depth else None
            text = INSTRUCTIONS[task]
            # state = env._get_obs()[:4]
            state = env._get_obs()[:4]

            # plan next target with PAD agent
            samples,sample_a,sample_depth = agent.action(text, rgb, depth, state)

            if CONFIG['visualize_prediction']:
                predict_img = agent.decode_rgb(rgb, samples) # np.array shape (256,256*3)
                predict_img = add_bound(predict_img)

            target = sample_a/agent.args.action_scale
            target_xyz, target_gripper = target[0,0,:3], target[0,0,3] # target pose
            curr_xyz, curr_gripper = state[:3], state[3] # current pose
            
            # motion planner to reach the target pose, starting from the current pose
            info, img = motion_planner(target_xyz, target_gripper, curr_xyz, curr_gripper, env, image_3, thirdview, predict_img=predict_img, img_word=img_word)
            print(info)

            if info['success']:
                print(task, traj_idx, 'success')
                success_num[selected_id] += 1
                break
        
        # save video
        # ckpt_path = ckpt_path.split('.')[0]
        video_dir = CONFIG['video_dir']
        os.makedirs(f'{video_dir}/rollout_metaworld', exist_ok=True)
        mediapy.write_video(f'{video_dir}/rollout_metaworld/{task}_{traj_idx}.mp4', image_3, fps=20)
    

for i in range(len(task_list)):
    print(task_list[i], success_num[i])

# running command:
# CUDA_VISIBLE_DEVICES=1 python sample_pose.py