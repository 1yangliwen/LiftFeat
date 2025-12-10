# 文件路径: demo.py
# (这是你的主程序入口文件)

import numpy as np
import cv2
import argparse
import yaml
import logging
import os
import sys

# 假设你的项目结构如下:
# project/
# |-- demo.py
# |-- utils/
# |   |-- VisualOdometry.py
# |-- wrappers/
# |   |-- ripe_wrapper.py
# |-- models/
# |   |-- liftfeat_wrapper.py
# 所以我们将项目根目录 project/ 加入到 python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.VisualOdometry import VisualOdometry, AbosluteScaleComputer, create_dataloader, \
    plot_keypoints, create_detector, create_matcher

# ==================== 核心修改部分 ====================
# 将 vo_config 指向 RIPE 模型
vo_config = {
    'dataset': {
        'name': 'KITTILoader',
        'root_path': '/home/yepeng_liu/code_python/dataset/visual_odometry/kitty/gray',
        'sequence': '10',
        'start': 0
    },
    'detector': {
        'name': 'RIPE',               # <-- 1. 修改名称以匹配工厂函数
        'threshold': 0.5,             # <-- 2. RIPE 的特有参数
        'top_k': 4096                 # <-- 3. RIPE 的特有参数
    },
    'matcher': {
        'name': 'FrameByFrameMatcher',
        'type': 'FLANN',              # RIPE描述子是浮点型，FLANN适用
        'FLANN': {
            'kdTrees': 5,
            'searchChecks': 50
        },
        'distance_ratio': 0.75
    }
}
# =======================================================


# 可视化当前frame的关键点
def keypoints_plot(img, vo, img_id, path2):
    img_ = cv2.imread(path2+str(img_id-1).zfill(6)+".png")
 
    if not vo.match_kps:
        # 注意: vo.kptdescs["cur"]["keypoints"] 现在可能是 numpy array
        # plot_keypoints 函数需要能处理它
        img_ = plot_keypoints(img_, vo.kptdescs["cur"]["keypoints"])
    else:
        for index in range(vo.match_kps["ref"].shape[0]):
            ref_point = tuple(map(int, vo.match_kps['ref'][index,:]))
            cur_point = tuple(map(int, vo.match_kps['cur'][index,:]))
            cv2.line(img_, ref_point, cur_point, (0, 255, 0), 2)
            cv2.circle(img_, cur_point, 3, (0, 0, 255), -1)

    return img_

# 负责绘制相机的轨迹并计算估计轨迹与真实轨迹的误差。
class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((800, 800, 3), dtype=np.uint8)

    def update(self, est_xyz, gt_xyz):
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]
        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)
        error = np.linalg.norm(est - gt)
        self.errors.append(error)
        avg_error = np.mean(np.array(self.errors))
        
        draw_x, draw_y = int(x) + 80, int(z) + 230
        true_x, true_y = int(gt_x) + 80, int(gt_z) + 230

        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 0, 255), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 255, 0), 2)
        cv2.rectangle(self.traj, (10, 5), (450, 120), (0, 0, 0), -1)

        text = "[AvgError] %2.4fm" % (avg_error)
        print(text)
        cv2.putText(self.traj, text, (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        note = "Green: GT, Red: Predict"
        cv2.putText(self.traj, note, (20, 80),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return self.traj

# 这个函数内的逻辑代码完全不需要修改
def run_video(args):
    # create dataloader
    vo_config["dataset"]['root_path'] = args.path1
    vo_config["dataset"]['sequence'] = args.id
    loader = create_dataloader(vo_config["dataset"])
    
    # create detector using factory
    detector = create_detector(vo_config["detector"])
    
    # create matcher
    matcher = create_matcher(vo_config["matcher"])

    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter()

    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    # 文件名可以根据 detector 动态生成
    fname = f"kitti_{vo_config['detector']['name']}_{vo_config['matcher']['type']}match"
    log_fopen = open("output/" + fname + ".txt", mode='w') # 使用 'w' 模式覆盖旧日志

    # 将 detector 传入 VisualOdometry
    vo = VisualOdometry(detector, matcher, loader.cam)

    keypoints_video_path = "output/" + fname + "_keypoints.avi"
    trajectory_video_path = "output/" + fname + "_trajectory.avi"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    
    # 获取第一帧图像以确定视频尺寸
    first_img_for_size = cv2.imread(args.path2 + str(vo_config['dataset']['start']).zfill(6) + ".png")
    h, w, _ = first_img_for_size.shape
    keypoints_writer = cv2.VideoWriter(keypoints_video_path, fourcc, fps, (w, h)) # 使用原始图像尺寸
    trajectory_writer = cv2.VideoWriter(trajectory_video_path, fourcc, fps, (800, 800))
    
    for i, img in enumerate(loader):
        img_id = loader.img_id
        gt_pose = loader.get_cur_pose()
        
        R, t = vo.update(img, absscale.update(gt_pose))
        
        print(f"Frame {i}: est_t = {t.flatten()}, gt_t = {gt_pose[:, 3]}", file=log_fopen)

        img1 = keypoints_plot(img, vo, img_id, args.path2)
        # img1 = cv2.resize(img1, (1200, 400)) # 建议不 resize 或使用原始图像尺寸
        img2 = traj_plotter.update(t, gt_pose[:, 3])

        keypoints_writer.write(img1)
        trajectory_writer.write(img2)
    
    log_fopen.close()
    keypoints_writer.release()
    trajectory_writer.release()
    print(f"Videos saved as {keypoints_video_path} and {trajectory_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--path1', type=str, default='/home/yangliwen/project/LiftFeat/dataset/visual_odometry/kitty/gray',
                          help='Path to the gray image sequences root directory')
    parser.add_argument('--path2', type=str, default="/home/yangliwen/project/LiftFeat/dataset/visual_odometry/kitty/color/sequences/03/image_2/",
                          help='Path to the color image sequence for visualization')
    parser.add_argument('--id', type=str, default="03",
                          help='Sequence ID to process')
    
    args = parser.parse_args()
    
    run_video(args)