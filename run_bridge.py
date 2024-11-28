import torch
import torch.distributed as dist
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
from evaluation.agent import DiffusionAgent
from evaluation.run_cfg import INSTRUCTIONS, CONFIG, BRIDGE_CONFIG 
from decord import VideoReader, cpu

def _load_video(video_path, frame_ids):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    assert (np.array(frame_ids) < len(vr)).all()
    assert (np.array(frame_ids) >= 0).all()
    vr.seek(0)
    try:
        frame_data = vr.get_batch(frame_ids).asnumpy() #(frame, h, w, c)
    except:
        frame_data = vr.get_batch(frame_ids).numpy()
    # central crop
    h, w = frame_data.shape[1], frame_data.shape[2]
    if h > w:
        margin = (h - w) // 2
        frame_data = frame_data[:, margin:margin + w]
    elif w > h:
        margin = (w - h) // 2
        frame_data = frame_data[:, :, margin:margin + h]
    return frame_data

INS = {
    'sample_0':"put potato in pot cardboard fence",
    'sample_1':"put carrot on plate",
    'sample_2':"put pear on plate",
    'sample_3':"Put the mushroom into the pot",
    'sample_4':"open fridge",
}

if __name__ == "__main__":
    # first-view
    prefix = "bridge"
    # build agent
    ckpt_path = BRIDGE_CONFIG['ckpt_path']
    agent = DiffusionAgent(ckpt_path=ckpt_path,vae_path=BRIDGE_CONFIG['vae_path'], clip_path=BRIDGE_CONFIG['clip_path'])

    name = BRIDGE_CONFIG['sample_name']
    path = f'gallery/bridge/{name}.mp4'
    frames = _load_video(path, [0, 4, 8, 12])
    rgb = frames[0]
    rgb = np.array(rgb)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    text = INS[name]

    depth = None
    state = None
    samples,sample_a,sample_depth = agent.action(text, rgb, depth, state)
    if sample_a is not None:
        print(sample_a)
    img = agent.decode(rgb, samples, prefix=prefix, save=False)

    uuid = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs("output/bridge_prediction", exist_ok=True)
    img = Image.fromarray(img).save(f"output/bridge_prediction/{name}_{uuid}.jpg")

# scp -P 8576 -r "/cephfs/cjyyj/dit_ckpt/128-DiT-XL-2-2024-05-13-15-31-58/checkpoints/0040000.pt" guoyanjiang@101.6.96.209:"/home/guoyanjiang/ckpt/dit/128"

