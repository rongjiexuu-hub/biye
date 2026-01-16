"""
2D人体姿态估计模块
使用MediaPipe进行2D关键点检测
"""

import cv2
import numpy as np
import mediapipe as mp
# 兼容不同版本的 mediapipe 导入方式
try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    # 尝试备用导入
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Keypoint2D:
    """2D关键点数据类"""
    x: float  # 归一化x坐标 (0-1)
    y: float  # 归一化y坐标 (0-1)
    z: float  # 深度估计
    visibility: float  # 可见度置信度
    name: str  # 关键点名称


@dataclass
class PoseResult2D:
    """2D姿态估计结果"""
    keypoints: List[Keypoint2D]  # 所有关键点
    landmarks_pixel: np.ndarray  # 像素坐标 (N, 2)
    landmarks_normalized: np.ndarray  # 归一化坐标 (N, 3)
    confidence: float  # 整体置信度
    image_width: int
    image_height: int


class Pose2DEstimator:
    """
    2D人体姿态估计器
    使用MediaPipe Pose进行人体关键点检测
    """
    
    # MediaPipe关键点名称映射
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # 骨架连接定义 (用于可视化)
    SKELETON_CONNECTIONS = [
        # 躯干
        (11, 12),  # 左肩-右肩
        (11, 23),  # 左肩-左髋
        (12, 24),  # 右肩-右髋
        (23, 24),  # 左髋-右髋
        # 左臂
        (11, 13),  # 左肩-左肘
        (13, 15),  # 左肘-左腕
        # 右臂
        (12, 14),  # 右肩-右肘
        (14, 16),  # 右肘-右腕
        # 左腿
        (23, 25),  # 左髋-左膝
        (25, 27),  # 左膝-左踝
        # 右腿
        (24, 26),  # 右髋-右膝
        (26, 28),  # 右膝-右踝
        # 面部
        (0, 1), (1, 2), (2, 3),  # 左眼
        (0, 4), (4, 5), (5, 6),  # 右眼
        (9, 10),  # 嘴巴
    ]
    
    # 主要关键点索引 (用于SMPL对齐)
    MAIN_KEYPOINT_INDICES = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 2,
        static_image_mode: bool = True
    ):
        """
        初始化2D姿态估计器
        
        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            model_complexity: 模型复杂度 (0, 1, 2)
            static_image_mode: 静态图像模式（对于单张图片设为True）
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=True  # 启用人体分割
        )
        
    def estimate(self, image: np.ndarray) -> Optional[PoseResult2D]:
        """
        对输入图像进行2D姿态估计
        
        Args:
            image: BGR格式的输入图像 (numpy数组)
            
        Returns:
            PoseResult2D对象，如果检测失败则返回None
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 进行姿态估计
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks is None:
            return None
        
        # 提取关键点
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        landmarks_pixel = []
        landmarks_normalized = []
        
        for i, landmark in enumerate(landmarks):
            name = self.KEYPOINT_NAMES[i] if i < len(self.KEYPOINT_NAMES) else f'point_{i}'
            
            keypoint = Keypoint2D(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
                name=name
            )
            keypoints.append(keypoint)
            
            # 像素坐标
            landmarks_pixel.append([landmark.x * w, landmark.y * h])
            # 归一化坐标 (包含深度)
            landmarks_normalized.append([landmark.x, landmark.y, landmark.z])
        
        # 计算整体置信度
        visibilities = [kp.visibility for kp in keypoints]
        avg_confidence = np.mean(visibilities)
        
        return PoseResult2D(
            keypoints=keypoints,
            landmarks_pixel=np.array(landmarks_pixel),
            landmarks_normalized=np.array(landmarks_normalized),
            confidence=avg_confidence,
            image_width=w,
            image_height=h
        )
    
    def get_main_keypoints(self, result: PoseResult2D) -> Dict[str, np.ndarray]:
        """
        获取主要关键点（用于SMPL对齐）
        
        Args:
            result: 2D姿态估计结果
            
        Returns:
            主要关键点字典
        """
        main_kpts = {}
        for name, idx in self.MAIN_KEYPOINT_INDICES.items():
            main_kpts[name] = result.landmarks_pixel[idx]
        return main_kpts
    
    def draw_pose(
        self,
        image: np.ndarray,
        result: PoseResult2D,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),
        skeleton_color: Tuple[int, int, int] = (255, 0, 0),
        keypoint_radius: int = 5,
        skeleton_thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制姿态
        
        Args:
            image: 输入图像
            result: 2D姿态估计结果
            draw_keypoints: 是否绘制关键点
            draw_skeleton: 是否绘制骨架
            keypoint_color: 关键点颜色 (BGR)
            skeleton_color: 骨架颜色 (BGR)
            keypoint_radius: 关键点半径
            skeleton_thickness: 骨架线条粗细
            
        Returns:
            绘制后的图像
        """
        output_image = image.copy()
        
        # 绘制骨架
        if draw_skeleton:
            for start_idx, end_idx in self.SKELETON_CONNECTIONS:
                if start_idx < len(result.landmarks_pixel) and end_idx < len(result.landmarks_pixel):
                    start_point = tuple(result.landmarks_pixel[start_idx].astype(int))
                    end_point = tuple(result.landmarks_pixel[end_idx].astype(int))
                    
                    # 根据可见度调整透明度
                    visibility = min(
                        result.keypoints[start_idx].visibility,
                        result.keypoints[end_idx].visibility
                    )
                    if visibility > 0.5:
                        cv2.line(output_image, start_point, end_point, 
                                skeleton_color, skeleton_thickness)
        
        # 绘制关键点
        if draw_keypoints:
            for i, kp in enumerate(result.keypoints):
                if kp.visibility > 0.5:
                    point = tuple(result.landmarks_pixel[i].astype(int))
                    cv2.circle(output_image, point, keypoint_radius, keypoint_color, -1)
                    cv2.circle(output_image, point, keypoint_radius, (255, 255, 255), 1)
        
        return output_image
    
    def get_bounding_box(self, result: PoseResult2D, padding: float = 0.1) -> Tuple[int, int, int, int]:
        """
        根据关键点计算人体边界框
        
        Args:
            result: 2D姿态估计结果
            padding: 边界框填充比例
            
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        visible_points = result.landmarks_pixel[
            [kp.visibility > 0.5 for kp in result.keypoints]
        ]
        
        if len(visible_points) == 0:
            return (0, 0, result.image_width, result.image_height)
        
        x_min, y_min = visible_points.min(axis=0)
        x_max, y_max = visible_points.max(axis=0)
        
        # 添加填充
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - width * padding)
        y_min = max(0, y_min - height * padding)
        x_max = min(result.image_width, x_max + width * padding)
        y_max = min(result.image_height, y_max + height * padding)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def to_dict(self, result: PoseResult2D) -> Dict:
        """
        将结果转换为字典格式（用于JSON序列化）
        """
        return {
            'keypoints': [
                {
                    'name': kp.name,
                    'x': float(kp.x),
                    'y': float(kp.y),
                    'z': float(kp.z),
                    'visibility': float(kp.visibility)
                }
                for kp in result.keypoints
            ],
            'confidence': float(result.confidence),
            'image_size': {
                'width': result.image_width,
                'height': result.image_height
            }
        }
    
    def close(self):
        """释放资源"""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def process_image(image_path: str) -> Optional[Dict]:
    """
    处理单张图片的便捷函数
    
    Args:
        image_path: 图片路径
        
    Returns:
        包含2D姿态信息的字典
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    with Pose2DEstimator() as estimator:
        result = estimator.estimate(image)
        if result is None:
            return None
        return estimator.to_dict(result)


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = process_image(image_path)
        if result:
            print(f"检测到 {len(result['keypoints'])} 个关键点")
            print(f"整体置信度: {result['confidence']:.2f}")
        else:
            print("未检测到人体")
    else:
        print("用法: python pose_2d.py <图片路径>")

