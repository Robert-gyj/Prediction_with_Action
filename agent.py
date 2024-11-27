import torch
import torch.distributed as dist
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DiffusionAgent():
    def __init__(self, ckpt_path, vae_path="/cephfs/shared/llm/sd-vae-ft-mse", clip_path="/cephfs/shared/llm/clip-vit-base-patch32",denoise_steps=200):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        args = checkpoint["args"]
        self.args = args

        new_attr = ["action_condition", "action_dim", "action_steps", "d_hidden_size", "use_depth", 'ckpt_wrapper']
        for attr in new_attr:
            if not hasattr(self.args, attr):
                setattr(self.args, attr, False)
        # if not hasattr(self.args, "action_condition"):
        #     self.args.action_condition = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("load dit")
        self.latent_size = args.image_size // 8
        self.model = DiT_models[args.model](
            input_size=self.latent_size,
            num_classes=args.num_classes,
            args = args
        )
        state_dict = checkpoint["model"]
        model_dict = self.model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print("load diffusion")
        self.diffusion = create_diffusion(str(denoise_steps))
        
        print("load vae and clip")
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device)
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.clip_model = CLIPTextModelWithProjection.from_pretrained(clip_path).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path)
    
    def encode_text(self, text):
        with torch.no_grad():
            inputs = self.clip_tokenizer([text], padding=True, return_tensors="pt").to(self.device)
            outputs = self.clip_model(**inputs)
            text_embeds = outputs.text_embeds
        return text_embeds

    def encode_image(self, image):
        image = np.array(image)
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
        image = torch.tensor(image).float().to(self.device).permute(2,0,1).unsqueeze(0)
        # print("image shape:", image.shape)
        image = torch.clamp((image-128.0)/127.5, -1, 1).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample().mul_(0.18215)
        return latent

    def filter(self, depth):
        depth = cv2.resize(depth, (32,32), interpolation=cv2.INTER_NEAREST)
        depth = np.clip(depth/10000,0,1)
        return depth

    def filter2(self, depth):
        depth = np.clip(depth,1000,5000)/5000
        depth = np.array(depth*256,dtype=np.uint8)
        depth = cv2.medianBlur(depth, 15)
        depth = cv2.resize(depth,(32,32),interpolation=cv2.INTER_NEAREST)/256
        return depth

    
    def action(self, text, rgb=None, depth=None,state=None):
        y = self.encode_text(text)
        x_cond = self.encode_image(rgb)
        if depth is not None:
            depth_cond = self.filter(depth) if not self.args.depth_filter else self.filter2(depth)
            depth_cond = depth_cond[np.newaxis][np.newaxis]
            depth_cond = torch.tensor(depth_cond).float().to(self.device)
        else:
            depth_cond = None
        # print("y shape:", y.shape, "x_cond shape:", x_cond.shape, "depth_cond shape:", depth_cond.shape)
        
        sample_fn = self.model.forward

        t = self.args.predict_horizon
        latent_size = self.args.image_size // 8
        z = torch.randn(1, self.model.in_channels*t, latent_size, latent_size).to(self.device)
        # if self.args has arribute action_condition and self.args.action_condition:
        if hasattr(self.args, "action_condition") and self.args.action_condition:
            action_cond = torch.tensor(state*self.args.action_scale).float().to(self.device).unsqueeze(0).unsqueeze(0)
        else:
            action_cond = None
        z_a_dim = 1 if self.args.action_condition else self.args.action_steps
        length = self.args.action_dim if not self.args.action_condition else int(self.args.action_dim*self.args.action_steps)
        z_a = torch.randn(1, z_a_dim, length, device=self.device) if self.args.action_steps>0 else None
        z_d = torch.randn(1, t, self.args.d_hidden_size, self.args.d_hidden_size, device=self.device) if self.args.use_depth else None

        model_kwargs = {
            "y": y,
            "x_cond": x_cond,
            "noised_action": z_a,
            "depth_cond": depth_cond,
            "noised_depth": z_d,
            "action_cond": action_cond
        }
        if self.args.action_steps ==0 and not self.args.use_depth:
            samples = self.diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
        )
            samples_a = None
            sample_depth = None
        else:
            samples, samples_a, sample_depth = self.diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
        )
            if samples_a is not None:
                samples_a = samples_a.detach().cpu().numpy()
            if sample_depth is not None:
                sample_depth = sample_depth.detach().cpu().numpy()

        return samples, samples_a, sample_depth
    
    def decode(self, x, x_pred, prefix="", save=False):
        x = cv2.resize(x, (256,256), interpolation=cv2.INTER_AREA)
        B,C,H,W = x_pred.shape
        t = self.args.predict_horizon
        x_pred = x_pred.view(B*t,int(C/t),H,W)
        rec_pred = self.vae.decode(x_pred / 0.18215).sample
        rec_pred = torch.clamp(127.5 * rec_pred + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() # 3, 256,256,3
        img = np.concatenate([x, rec_pred[0], rec_pred[1], rec_pred[2]], axis=1)
        if not save:
            return img
        # save img
        import time
        uuid = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        img = Image.fromarray(img).save(f"output/predict_{prefix}_{uuid}.jpg")
    
    def decode_depth(self, depth, depth_pred, prefix="",save=False):
        # depth = cv2.resize(depth, (32,32), interpolation=cv2.INTER_NEAREST)
        depth_cond = self.filter(depth) if not self.args.depth_filter else self.filter2(depth)
        B,C,H,W = depth_pred.shape
        t = self.args.predict_horizon
        depth_pred = depth_pred.reshape(B*t,int(C/t),H,W)
        print("depth_pred shape:", depth_pred.shape)
        print("depth shape:", depth_cond.shape)
        depth_img = np.concatenate([depth_cond, depth_pred[0][0], depth_pred[1][0], depth_pred[2][0]], axis=1)
        if not save:
            return depth_img
        # save img
        import time
        uuid = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        plt.imsave(f"output/predict_{prefix}_{uuid}_depth.png", depth_img)
        # Image.fromarray(depth_img).save(f"diffusion_deploy/predict_{prefix}_{uuid}_depth2.png")

    def decode_rgb(self, x, x_pred, prefix=""):
        x = cv2.resize(x, (256,256), interpolation=cv2.INTER_AREA)
        B,C,H,W = x_pred.shape
        t = self.args.predict_horizon
        x_pred = x_pred.view(B*t,int(C/t),H,W)
        rec_pred = self.vae.decode(x_pred / 0.18215).sample
        rec_pred = torch.clamp(127.5 * rec_pred + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() # 3, 256,256,3
        img = np.concatenate([rec_pred[0], rec_pred[1], rec_pred[2]], axis=1)
        return img




if __name__ == "__main__":
    # depth image
    agent = DiffusionAgent(ckpt_path="/cephfs/cjyyj/dit_ckpt/063-DiT-XL-2-2024-04-26-21-58-30/checkpoints/0120000.pt")

    # load image at "/cephfs/shared/panda_real_data_processed/2024-04-26-pick_random/episode0000009/color_wrist_1_0000.jpg"
    rgb = Image.open("/cephfs/shared/panda_real_data_processed/2024-04-26-pick_random/episode0000409/color_wrist_1_0000.jpg").convert("RGB")
    rgb = np.array(rgb)
    
    depth = np.load("/cephfs/shared/panda_real_data_processed/2024-04-26-pick_random/episode0000409/depth_wrist_1_0000.npy")
    text = "pick the blue block"

    samples,sample_a,sample_depth = agent.action(text, rgb, depth)
    if sample_a is not None:
        print(sample_a)
    agent.decode(rgb, samples)
    
        


