# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_mask = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if self.attn_mask is not None:
                self.attn_mask = self.attn_mask.to(x.device)
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=self.attn_mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class LanguageEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size,args):
        super().__init__()
        # from transformers import AutoTokenizer, CLIPTextModelWithProjection
        # self.model = CLIPTextModelWithProjection.from_pretrained(args.clip_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.clip_path)

        self.embedding_table = nn.Linear(512, hidden_size) # 512 is clip embedding size

    def forward(self, labels, train, force_drop_ids=None):
        # with torch.no_grad():
        #     inputs = self.tokenizer(labels, padding=True, return_tensors="pt")
        #     outputs = self.model(**inputs) 
        #     text_embeds = outputs.text_embeds # (B,512)
        embeddings = self.embedding_table(labels) # (B, hidden_size)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels,args=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.args = args
        self.action_steps = args.action_steps
        self.use_action = args.action_steps>0
        if args.action_steps>0:
            self.a_norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.a_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            if not args.action_condition:
                output_size = args.action_dim*2
            else:
                output_size = args.action_dim*(args.action_steps)*2
            self.a_linear = nn.Linear(hidden_size,output_size,bias=True)
            # self.a_head = nn.Sequential(
            #     nn.Linear(hidden_size, hidden_size, bias=True),
            #     nn.SiLU(),
            #     nn.Linear(hidden_size, args.action_dim*2, bias=True)
            # )

        self.use_depth = args.use_depth
        if args.use_depth:
            self.d_norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.d_linear = nn.Linear(hidden_size, args.d_patch_size ** 2 * args.predict_horizon *2 , bias=True)
            self.d_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        # rgb
        start,end = self.args.start_idx[0],self.args.end_idx[0]
        rgb = x[:, start:end]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        rgb = modulate(self.norm_final(rgb), shift, scale)
        rgb = self.linear(rgb)
        if not (self.use_action or self.use_depth):
            return rgb

        a, d = None, None
        if self.use_action:
            start,end = self.args.start_idx[1],self.args.end_idx[1]
            a = x[:, start:end]
            shift, scale = self.a_adaLN_modulation(c).chunk(2, dim=1)
            a = modulate(self.a_norm_final(x[:, start:end]), shift, scale)
            a = self.a_linear(a)
            # a = self.a_head(x[:, start:end])
        if self.use_depth:
            start,end = self.args.start_idx[2],self.args.end_idx[2]
            d = x[:, start:end]
            d_shift, d_scale = self.d_adaLN_modulation(c).chunk(2, dim=1)
            d = modulate(self.d_norm_final(d), d_shift, d_scale)
            d = self.d_linear(d)
        return (rgb, a, d)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        args=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = in_channels * 2 if not args.dynamics else in_channels * 2 * args.predict_horizon
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.args = args
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        x_embedder_channels = in_channels if not args.dynamics else in_channels+in_channels*args.predict_horizon

        self.x_embedder = PatchEmbed(input_size, patch_size, x_embedder_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if args.action_steps>0:
            action_input_shape = args.action_dim*(args.action_steps+1) if args.action_condition else args.action_dim
            self.a_embedder = nn.Linear(action_input_shape, hidden_size)
            a_token = args.action_steps if not args.action_condition else 1
            if args.learnable_action_pos:
                self.a_pos_embed = nn.Parameter(torch.zeros(1, a_token, hidden_size), requires_grad=True)
            else:
                self.a_pos_embed = nn.Parameter(torch.zeros(1, a_token, hidden_size), requires_grad=False)
        if args.use_depth:
            d_embedder_channels = 1 if not args.dynamics else 1+args.predict_horizon
            d_patch_size = args.d_patch_size
            d_input_size = args.d_hidden_size
            self.d_embedder = PatchEmbed(d_input_size, d_patch_size, d_embedder_channels, hidden_size, bias=True)
            self.d_in_channels = 1
            self.d_out_channels = args.predict_horizon
            self.d_num_patches = (d_input_size // d_patch_size) ** 2
            self.d_patch_size = d_patch_size

            self.d_pos_embed = nn.Parameter(torch.zeros(1, self.d_num_patches, hidden_size), requires_grad=False)

        if args.text_cond:
            self.y_embedder = LanguageEmbedder(hidden_size,args)
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        rgb_l = num_patches
        a_l = self.args.action_steps if not args.action_condition else 1
        depth_l = 0 if not args.use_depth else self.d_num_patches

        self.args.start_idx = [0, rgb_l, rgb_l+a_l]
        self.args.end_idx = [rgb_l, rgb_l+a_l, rgb_l+a_l+depth_l]

        attn_mask = None
        if args.attn_mask:
            token_num = rgb_l + a_l + depth_l
            attn_mask = torch.ones((token_num, token_num), dtype=torch.bool)
            attn_mask[:rgb_l,rgb_l:] = False
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_mask=attn_mask) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.args)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.args.use_depth:
            # pos_embed = get_2d_sincos_pos_embed(self.d_pos_embed.shape[-1], int(self.d_num_patches ** 0.5))
            ratio = self.d_patch_size // self.patch_size
            l = int(self.x_embedder.num_patches ** 0.5)
            pos_embed = pos_embed.reshape([l,l,self.pos_embed.shape[-1]])[::ratio,::ratio,:]
            pos_embed = pos_embed.reshape([-1,self.pos_embed.shape[-1]])
            self.d_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize a_embedder
        if self.args.action_steps>0:
            nn.init.xavier_uniform_(self.a_embedder.weight)
            nn.init.constant_(self.a_embedder.bias, 0)
            _, pos, h = self.a_pos_embed.shape
            if self.args.learnable_action_pos:
                nn.init.xavier_uniform_(self.a_pos_embed)
            else:
                a_pos_embed = get_1d_sincos_pos_embed_from_grid(h,np.arange(pos, dtype=np.float32))
                self.a_pos_embed.data.copy_(torch.from_numpy(a_pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, in_channels=None, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        # c = self.in_channels * 2 * self.args.predict_horizon
        # p = self.x_embedder.patch_size[0]
        in_channels = self.in_channels if in_channels is None else in_channels
        patch_size = self.patch_size if patch_size is None else patch_size

        c = in_channels * 2 * self.args.predict_horizon
        p = patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    # def unpatchify(self, x):
    #     """
    #     x: (N, T, patch_size**2 * C* t)
    #     imgs: (N, H, W, C)
    #     """
    #     # split x to t part in the last dim
    #     d = self.patch_size ** 2*self.in_channels*2
    #     xs = torch.split(x, d, dim=-1)
    #     img_mean, img_std = [], []
    #     for x in xs:
    #         # c = self.out_channels
    #         c = self.in_channels*2
    #         p = self.x_embedder.patch_size[0]
    #         h = w = int(x.shape[1] ** 0.5)
    #         assert h * w == x.shape[1]

    #         x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #         x = torch.einsum('nhwpqc->nchpwq', x)
    #         img = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    #         img_mean.append(img[:,:self.in_channels])
    #         img_std.append(img[:,self.in_channels:])
    #     imgs = torch.cat(img_mean+img_std, dim=1)
    #     return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y, x_cond=None, action_cond=None, noised_action=None, depth_cond=None, noised_depth=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        x_cond: (N, C, H, W) conditioned image
        action: (N, n_step, a_dim)
        output_type: 'image' or 'action'
        """
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1) #(N, 2C, H, W) C with noise, C ture value
        
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if self.args.action_steps>0:
            # TODO: action_pos_embed current is just learnable parameters
            if self.args.action_condition:
                noised_action = torch.cat([noised_action,action_cond],dim=-1)
            a = self.a_embedder(noised_action)+self.a_pos_embed # (N, action_steps, D)
            x = torch.cat([x,a],dim=1) # (N, T + action_step, D)
        if self.args.use_depth:
            noised_depth = torch.cat([noised_depth,depth_cond],dim=1)
            d = self.d_embedder(noised_depth) + self.d_pos_embed
            x = torch.cat([x,d],dim=1)
            # print("x_shape",x.shape)

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c) if self.args.ckpt_wrapper else block(x, c)   # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        # print("x_shape", x.shape)
        if self.args.action_steps>0 or self.args.use_depth:
            x, a, d = x
            x = self.unpatchify(x)
            if d is not None:
                d = self.unpatchify(d, in_channels=self.d_in_channels, patch_size=self.d_patch_size)
            # print("x_shape", x.shape, d.shape)
            # print(a)
            return (x,a,d)                 # (N, out_channels, H, W), (N, action_steps, action_dim)
        else:
            x = self.unpatchify(x)                   # (N, out_channels, H, W)
            return x

    def forward_with_cfg(self, x, t, y, cfg_scale, x_cond=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, x_cond=x_cond)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
