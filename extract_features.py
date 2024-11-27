# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        # os.makedirs(os.path.join(args.features_path, 'imagenet256_features'), exist_ok=True)
        # os.makedirs(os.path.join(args.features_path, 'imagenet256_labels'), exist_ok=True)

    # Create img encoder model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained("/cephfs/shared/llm/sd-vae-ft-mse/").to(device)
    # Create text encoder model:
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    model = CLIPTextModelWithProjection.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32/")

    # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # /home/zhangjianke/CrossEmbodimentData/jaco_play_processed/
    import json
    json_all = "{}/dataset_info.json".format(args.data_path)
    with open(json_all, "r") as f:
        json_all = json.load(f)
    dataset_info = []

    dataset = ImageFolder(args.data_path, transform=transform)
    print(len(dataset.imgs))

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    episode_steps = 0
    img_idx = 0
    traj_idx_pre = -1
    dataset_list = dataset.imgs
    print(dataset_list[:100])
    for x, y in loader:
        if '_1' in dataset_list[img_idx][0] or '.npy' in dataset_list[img_idx][0]:
            img_idx+=1
            continue
        y = y.to(device)
        y = y.detach().cpu().numpy()    # (1,)
        traj_id = y[0]
        os.makedirs(os.path.join(args.features_path, f"episode{traj_id:07}"), exist_ok=True)
        
        # detect new traj 

        if traj_id != traj_idx_pre:
            episode_steps = 0
            instruction = json_all[int(traj_id)]["instruction"]
            print(f"traj_id: {traj_id}")

            # save text features
            with torch.no_grad():
                inputs = tokenizer([instruction], padding=True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                text_embeds = outputs.text_embeds
            np.save(f'{args.features_path}/episode{traj_id:07}/text_clip.npy', text_embeds.cpu().numpy())
        
        # save image features
        x = x.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x = x.detach().cpu().numpy()    # (1, 4, 32, 32)
        np.save(f'{args.features_path}/episode{traj_id:07}/color_wrist_1_{episode_steps:04}.npy', x)
        # img = Image.open(dataset_list[img_idx][0])
        # img.save(f'{args.features_path}/episode{traj_id:07}/img{episode_steps:04}.jpg')
        # copy depth.npy
        color_img = dataset_list[img_idx][0]
        depth_img = color_img.replace('color_wrist_2', 'depth_wrist_1')
        depth_img = depth_img.replace('png', 'npy')
        os.system(f'cp {depth_img} {args.features_path}/episode{traj_id:07}/depth_wrist_1_{episode_steps:04}.npy')
         
        

        # save json info
        instruction = json_all[int(traj_id)]["instruction"]
        action = json_all[int(traj_id)]["action"][episode_steps]
        state = json_all[int(traj_id)]["features"][episode_steps]
        dataset_info.append({"idx":str(train_steps), "episode": str(traj_id), "frame": str(episode_steps), 
                             "wrist_1": f'episode{traj_id:07}/color_wrist_1_{episode_steps:04}.npy', 
                             "depth_1": f'episode{traj_id:07}/depth_wrist_1_{episode_steps:04}.npy',
                             "label": str(traj_id), 
                             "instruction": instruction,  
                             "ins_emb_path": f'episode{traj_id:07}/text_clip.npy', 
                             "action":action,
                             "state":state
                             })

        traj_idx_pre = traj_id
        train_steps += 1
        episode_steps += 1
        img_idx += 1
    
    import json
    with open(os.path.join(args.features_path, "dataset_rgb_s_d.json"), "w") as f:
        json.dump(dataset_info, f,indent=2)

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
