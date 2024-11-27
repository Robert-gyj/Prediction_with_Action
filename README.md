<div align="center">
<h2><center>👉 Prediction with Action: Visual Policy Learning via Joint Denoising Process</h2>

[Yanjiang Guo*](), [Yucheng Hu*](), [Jianke Zhang](), [Yen-Jen Wang](), [Xiaoyu Chen](), [Chaochao Lu](), [Jianyu Chen]()


<a href='https://arxiv.org/abs/2311.12886'><img src='https://img.shields.io/badge/ArXiv-2311.12886-red'></a> 
<a href='https://animationai.github.io/AnimateAnything/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

![DiT samples](gallery/PAD_method.png)

This repo is the official PyTorch implementation for the NeurIPS 2024 paper [**Prediction with Action**](https://www.wpeebles.com/DiT).

## Friendship Link 🔥

🔥🔥🔥We are excited to announce the open-source release of our latest work [**Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations**](https://github.com/alibaba/Tora) which is stronger and faster. Video Prediction Policy trains a video predcition model for manipulation domain with large-scale internet maniplation datasets to guide action learning.


##  Installation 🛠️
First, download and set up the repo.

```bash
git clone https://github.com/Robert-gyj/prediction_with_action.git
conda env create -f environment.yml
conda activate PAD
```


If you want to perform experiments in [Metaworld](https://github.com/Farama-Foundation/Metaworld), you need to first install the `Mujoco`. Metaworld is in active developing with different versions of `Mujoco`, we include a older version of Metaworld in codebase based on `mujoco-py==2.0`. 
```bash
cd metaworld
pip install -e .
```


## CheckPoints 📷
First you need to download the [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) and [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) checkpoints from the huggingface. These models are freeze during trainning.

Then download the PAD ckpt. For convenience, we provide two types of PAD models to download based on your demands:

| Ckpt name     | Training type | Parameter Num |
|---------------|------------------|---------|
| [bridge-pre](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | Bridge Dataset Pretrain         | 673M    |
| [bridge-pre-mw-ft](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) |    Bridge Dataset pretrain + Metaworld Finetuned      | 677M    |


**📊 Try Predictions on Bridge:** If you want to make predictions on bridge datasets, download the bridge-pre model.

**📊 Try Rollouts in Metaworld:** If you want to rollout on Metaworld benchmark, download the bridge-pre-mw-ft model.

**🛸 Train PAD in new environments**: If you want to run PAD algorithm in other environments, download the bridge-pre model and initializa your model with it.




## Evaluation 📊
### 📊 Make future predictions on the Bridge datasets
For your convenience, we put some bridge video samples in the folder `gallery/bridge`. You can visualize predictions by running inference on ckpt [bridge-pre](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt). Remember to reset the `ckpt_path, vae_path, clip_path, sample_name` in `BRIDGE_CONFIG` of `run_cfg.py` to your model download locations.

```bash
python run_bridge.py
```

You should find some outputs in the folder `output/bridge_prediction` wchich is simialr to that in `gallery/bridge_prediction`


### 📊 Rollout on Metaworld benchmarks
You should install Metaworld as described in installation section and also download the [bridge-pre-mw-ft](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) ckpt.  Remember to reset the `ckpt_path, vae_path, clip_path` in `CONFIG` of `run_cfg.py` to where your model located. After these setup finishes, you can rollout with:

```bash
python run_meatworld.py
```

You should find some rollout videos in the folder `output/rollout_metaworld` wchich visualize both the current observations and future predictions. Some examples are provided in `gallery/rollout_metaworld`.



## Trainning PAD 🛸 


### 🛸 Prepare your datasets
(1) We highly recommand to pre-encode your rgb images and save the corresponding latents. This process will save GPU memory cost and reduce training time. We provide an example script in [`extract_features.py`](extract_features.py).

(2) Also you need to reimplement the `RobotDataset` class in [`train_robot.py`](train_robot.py) to fit your datasets.

### 🛸 Training requirements
Our experiments are run with batch size 64*4=256 on 4 A800 80G cards. Under this setting, the training process takes ~70G GPU memory and the training speed is ~1.15 ite/s. 

If you have limited GPU memory, you can use `torch.utils.checkpoint.checkpoint` to save GPU memory which sacrifice some training speed. You just need to set the `--ckpt_wrapper` flag in [`exp.sh`](exp.sh) to enable it. On the same A800 GPUs, the training process takes ~18G GPU memory and the training speed is ~0.90 ite/s. 

### 🛸 Training script
We provide two training script for PAD in [`train_robot.py`](train_robot.py) and [`train_cotrain.py`](train_cotrain.py). You can initailiza the model with the pretrained ckpt [bridge-pre](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) by setting `--rgb_init xxx`.

You can train directly on custemer robot datasets with [`train_robot.py`](train_robot.py). Also, you can cotrain on robot datasets and internet video datasets with [`train_cotrain.py`](train_cotrain.py).


To launch PAD training with multiple GPUs on one node:
```bash
bash exp.sh
```



## Bibtex 
🌟 If you find our work helpful, please leave us a star and cite our paper.
```
@misc{dai2023animateanything,
      title={AnimateAnything: Fine-Grained Open Domain Image Animation with Motion Guidance}, 
      author={Zuozhuo Dai and Zhenghao Zhang and Yao Yao and Bingxue Qiu and Siyu Zhu and Long Qin and Weizhi Wang},
      year={2023},
      eprint={2311.12886},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}