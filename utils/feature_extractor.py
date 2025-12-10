from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """所有特征提取器 Wrapper 都应继承这个类。"""
    
    @abstractmethod
    def extract(self, image):
        """
        接收一张图像，返回包含关键点和描述子Tensors的字典。
        
        Args:
            image (np.array): 输入图像。
        
        Returns:
            dict: 必须包含 'keypoints' 和 'descriptors' 两个键，
                  其值为 PyTorch Tensors 或 NumPy Arrays。
        """
        pass
# =================================================