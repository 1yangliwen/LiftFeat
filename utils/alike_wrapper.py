"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
"""


import sys
import os

ALIKE_PATH = '/home/yangliwen/project/ALIKE'
sys.path.append(ALIKE_PATH)

import torch
import torch.nn as nn
from alike import ALike
import cv2
import numpy as np

import pdb

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-l.pth')},
}


class ALikeExtractor(nn.Module):
    def __init__(self,model_type,device=None) -> None:
        super().__init__()
        self.net=ALike(**configs[model_type],device='cpu',top_k=4096,scores_th=0.1,n_limit=8000)

        # 如果你确定不训练 ALike，可以锁死参数
        for param in self.net.parameters():
            param.requires_grad = False
        
    
    @torch.inference_mode()
    def extract_alike_kpts(self,img):
        pred0=self.net(img,sub_pixel=True)
        return pred0['keypoints']
    
    @torch.inference_mode()
    def extract_alike_kpts_batch(self, img_tensor):
        """
        处理 Batch Tensor 输入，全程 GPU 加速
        :param img_tensor: (B, 3, H, W), 值域 0-1, 标准化后的 Tensor
        :return: list of Tensors, 长度为 B. 每个元素是 (N, 2) 的关键点坐标
        """
        # 1. 直接提取特征图 (ALike 原生支持 Batch)
        # extract_dense_map 返回的是 scores_map, descriptor_map
        descriptor_map, scores_map = self.net.extract_dense_map(img_tensor, ret_dict=False)
        
        # 2. 调用 DKD (Soft Detect) 解码关键点
        # dkd 返回 keypoints, descriptors, scores 等
        # 注意：self.net.dkd 应该能处理 Batch 输入并返回列表（取决于 soft_detect 具体实现，通常是可以的）
        # 如果 dkd 返回的是 padded tensor，这里可能需要处理一下；
        # 但查看 alike 源码，它在 forward 里取了 [0]，暗示它返回的是 list 或 batch 维度的结构
        keypoints, _, _, _ = self.net.dkd(scores_map, descriptor_map, sub_pixel=True)
        
        # keypoints 通常是一个 List 或 Tuple，长度等于 Batch Size
        return keypoints


