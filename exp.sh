accelerate launch  --multi_gpu --gpu_ids 0,1,2,3 --num_processes 4  train_metaworld.py \
    --model DiT-XL/2 \
    --global-batch-size 256  \
    --dynamics \
    --text_cond  \
    --without_ema \
    --predict_horizon 3 \
    --skip_step 4 \
    --ckpt-every 10000 \
    --epochs 10000  \
    --action_scale 1 \
    --action_steps 3 \
    --action_dim 4 \
    --attn_mask \
    --absolute_action \
    --action_condition \
    --feature-path /cephfs/shared/metaworld/mt50_2_npy \
    --results-dir /cephfs/cjyyj/dit_ckpt \
    --rgb_init /cephfs/cjyyj/dit_ckpt/110-DiT-XL-2-2024-05-10-15-50-30/checkpoints/0080000.pt \


# sample ddp
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --master_port=25601 sample_ddp.py --model DiT-XL/2 --num-fid-samples 48 --feature-path "/home/gyj/dataset/bridge_features_consist" --dynamics --text_cond --ckpt /cephfs/cjyyj/dit_ckpt/000-DiT-XL-2-2024-04-11-17-58-00/checkpoints/0350000.pt --cfg-scale 0 --global-seed 2 --num-sampling-steps 150 --action steps 0