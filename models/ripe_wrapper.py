# 文件: wrappers/ripe_wrapper.py

import torch
import numpy as np
import cv2
# 假设 FeatureExtractor 抽象类在你的项目中可以这样导入
from utils.feature_extractor import FeatureExtractor

import sys, os
# 假设你把 ripe 源码放在项目根的 third_party/ripe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', 'third_party', 'ripe')))

# 确保 RIPE 的代码在你的 Python 环境路径中
from ripe import vgg_hyper
from ripe.utils.utils import resize_image

class RIPEWrapper(FeatureExtractor):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RIPE model is running on: {self.device}")

        self.model = vgg_hyper().to(self.device)
        self.model.eval()

        self.threshold = config.get('threshold', 0.5)
        self.top_k = config.get('top_k', 4096)
    
    def _preprocess(self, image_np_bgr):
        """
        将输入的 BGR NumPy 图像预处理成 RIPE 模型需要的 Tensor 格式。
        """
        # 1. BGR -> Grayscale
        image_np_gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
            
        # 2. NumPy -> PyTorch Tensor
        image_tensor = torch.from_numpy(image_np_gray.astype(np.float32))

        # 3. 数值归一化到 [0, 1]
        image_tensor = image_tensor / 255.0

        # 4. 增加 Batch 和 Channel 维度 (H, W) -> (1, 1, H, W)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        # 5. 调用 RIPE 自带的 resize 工具函数
        # image_tensor = resize_image(image_tensor)
        
        # 6. 将数据移动到指定设备
        return image_tensor.to(self.device)

    # 这个方法签名现在和你的 VisualOdometry 类完美匹配！
    @torch.no_grad()
    def extract(self, image):
        """
        接收一张图像，返回包含关键点和描述子Tensors的字典。
        
        Args:
            image (np.array): 从 KITTILoader 传来的 BGR 图像。
        
        Returns:
            dict: {'keypoints': torch.Tensor, 'descriptors': torch.Tensor}
        """
        # ==================== 新增代码：获取原始尺寸 ====================
        # image 是原始的 numpy 图像
        H_orig, W_orig = image.shape[:2]
        # ==========================================================

        # 1. 预处理图像 (包括了 resize)
        image_tensor = self._preprocess(image)
        
        # ================== 新增代码：获取预处理后的尺寸 ==================
        # image_tensor 的形状是 (B, C, H, W)，所以 H 在第2维, W 在第3维
        _B, _C, H_new, W_new = image_tensor.shape
        # ==========================================================
        
        # 2. 模型推理
        kpts_tensor, desc_tensor, score_tensor = self.model.detectAndCompute(
            image_tensor, 
            threshold=self.threshold, 
            top_k=self.top_k
        )

        # ================== 新增代码：坐标点按比例放大 ==================
        # kpts_tensor 的形状是 (N, 2)，其中第一列是 x (对应宽度 W)，第二列是 y (对应高度 H)
        # 计算缩放比例
        scale_w = W_orig / W_new
        scale_h = H_orig / H_new
        
        # 创建一个缩放向量
        scale_tensor = torch.tensor([scale_w, scale_h], device=kpts_tensor.device, dtype=kpts_tensor.dtype)
        
        # 将所有关键点坐标进行等比例放大
        kpts_scaled_tensor = kpts_tensor * scale_tensor
        # ==========================================================

        # 3. 直接返回包含 Tensors 的字典
        #    不需要再转换为 cv2.KeyPoint 或 numpy array
        return {
            "keypoints": kpts_scaled_tensor,
            "descriptors": desc_tensor
        }
    
    # 我们仍然可以保留这个方法，以防其他地方需要OpenCV格式的输出
    # 但对于 VisualOdometry 来说，它是不被调用的
    def detectAndCompute(self, image):
        predict_data = self.extract(image)
        kpts_tensor = predict_data['keypoints']
        desc_tensor = predict_data['descriptors']
        
        # 这里的 to_cv_kpts 和 .cpu().numpy() 只在这个方法里用
        # 需要 from ripe.utils.utils import to_cv_kpts
        # keypoints_cv = to_cv_kpts(kpts_tensor, score_tensor) # score_tensor 作用域问题
        # descriptors_np = desc_tensor.cpu().numpy()
        # return keypoints_cv, descriptors_np
        pass