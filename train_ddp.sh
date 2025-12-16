#!/bin/bash

# ============================================================
# LiftFeat 分布式训练启动脚本 (Multi-GPU Training with DDP)
# ============================================================
#
# 使用方法:
#   1. 修改下方的配置参数
#   2. 运行: bash train_ddp.sh
#
# 注意事项:
#   - 确保 CUDA_VISIBLE_DEVICES 中的 GPU 数量与 NPROC_PER_NODE 一致
#   - 多卡训练时，每卡的 batch_size 会乘以 GPU 数量得到总 batch_size
#   - 建议根据 GPU 数量调整 batch_size 和学习率
# ============================================================

# ==================== 基础配置 ====================
# 设置可见的 GPU（例如：0,1 表示使用第0和第1块GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 每个节点的 GPU 数量（与上面的 GPU 数量一致）
NPROC_PER_NODE=6

# 主节点端口（避免与其他程序冲突，可以改成其他值如 29501, 29502 等）
MASTER_PORT=29500

# ==================== 训练配置 ====================
# 实验名称
NAME="LiftFeat_DDP"

# Checkpoint 保存路径
CKPT_PATH="trained_weights/ddp_test"

# MegaDepth 数据集路径
MEGADEPTH_PATH="data"

# COCO 数据集路径
COCO_PATH="data/coco_20k"

# ==================== 训练超参数 ====================
# 每卡的 MegaDepth batch size（总batch = 这个值 × GPU数量）
MEGADEPTH_BATCH_SIZE=4

# 每卡的 COCO batch size（总batch = 这个值 × GPU数量）
COCO_BATCH_SIZE=2

# 学习率（多卡训练时可以适当调大，例如：lr * sqrt(num_gpus)）
LR=3e-4

# 训练步数
N_STEPS=160000

# 每多少步保存一次 checkpoint
SAVE_EVERY=500

# ==================== 开始训练 ====================
echo "=============================================="
echo "Starting Distributed Training with ${NPROC_PER_NODE} GPUs"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# 使用 torchrun 启动分布式训练（推荐，PyTorch 1.10+）
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    train.py \
    --name ${NAME} \
    --distributed \
    --world_size ${NPROC_PER_NODE} \
    --ckpt_save_path ${CKPT_PATH} \
    --megadepth_root_path ${MEGADEPTH_PATH} \
    --use_megadepth \
    --megadepth_batch_size ${MEGADEPTH_BATCH_SIZE} \
    --coco_root_path ${COCO_PATH} \
    --use_coco \
    --coco_batch_size ${COCO_BATCH_SIZE} \
    --lr ${LR} \
    --n_steps ${N_STEPS} \
    --save_ckpt_every ${SAVE_EVERY}

# ==================== 备选方案 ====================
# 如果你的 PyTorch 版本较低（< 1.10），可以使用下面的命令代替 torchrun：
#
# python -m torch.distributed.launch \
#     --nproc_per_node=${NPROC_PER_NODE} \
#     --master_port=${MASTER_PORT} \
#     train.py \
#     --name ${NAME} \
#     --distributed \
#     --world_size ${NPROC_PER_NODE} \
#     --ckpt_save_path ${CKPT_PATH} \
#     --megadepth_root_path ${MEGADEPTH_PATH} \
#     --use_megadepth \
#     --megadepth_batch_size ${MEGADEPTH_BATCH_SIZE} \
#     --coco_root_path ${COCO_PATH} \
#     --use_coco \
#     --coco_batch_size ${COCO_BATCH_SIZE} \
#     --lr ${LR} \
#     --n_steps ${N_STEPS} \
#     --save_ckpt_every ${SAVE_EVERY}
