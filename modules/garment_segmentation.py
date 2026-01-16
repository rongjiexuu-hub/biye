"""
服装分割与检测模块
从图像中提取服装区域
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GarmentType(Enum):
    """服装类型枚举"""
    UPPER_BODY = "upper_body"      # 上装
    LOWER_BODY = "lower_body"      # 下装
    FULL_BODY = "full_body"        # 连体装
    OUTERWEAR = "outerwear"        # 外套
    DRESS = "dress"                # 连衣裙
    UNKNOWN = "unknown"


@dataclass
class GarmentRegion:
    """服装区域数据类"""
    garment_type: GarmentType
    mask: np.ndarray              # 二值掩码
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    contour: np.ndarray           # 轮廓点
    area: float                   # 面积
    confidence: float             # 置信度
    cropped_image: np.ndarray     # 裁剪的服装图像
    

@dataclass 
class SegmentationResult:
    """分割结果"""
    garments: List[GarmentRegion]
    full_mask: np.ndarray         # 完整分割掩码
    visualization: np.ndarray     # 可视化图像
    person_mask: np.ndarray       # 人体掩码


class GarmentSegmenter:
    """
    服装分割器
    使用颜色空间分析和边缘检测进行服装分割
    """
    
    # 常见服装颜色范围 (HSV)
    COLOR_RANGES = {
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50]),
        'red': ([0, 100, 100], [10, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255]),
        'yellow': ([20, 100, 100], [40, 255, 255]),
        'gray': ([0, 0, 50], [180, 50, 200]),
    }
    
    def __init__(self, use_mediapipe: bool = True):
        """
        初始化服装分割器
        
        Args:
            use_mediapipe: 是否使用MediaPipe进行人体分割
        """
        self.use_mediapipe = use_mediapipe
        
        if use_mediapipe:
            import mediapipe as mp
            # 兼容不同版本的 mediapipe 导入方式
            self.mp_selfie = mp.solutions.selfie_segmentation
            self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)
    
    def segment(self, image: np.ndarray, person_keypoints: Optional[np.ndarray] = None) -> SegmentationResult:
        """
        分割服装区域
        
        Args:
            image: BGR格式输入图像
            person_keypoints: 可选的人体关键点 (用于精确分割)
            
        Returns:
            SegmentationResult对象
        """
        h, w = image.shape[:2]
        
        # 获取人体掩码
        person_mask = self._get_person_mask(image)
        
        # 基于人体关键点划分上下身区域
        if person_keypoints is not None:
            upper_mask, lower_mask = self._split_body_regions(person_mask, person_keypoints, h, w)
        else:
            # 简单的上下半分
            upper_mask = person_mask.copy()
            lower_mask = person_mask.copy()
            mid_y = h // 2
            upper_mask[mid_y:, :] = 0
            lower_mask[:mid_y, :] = 0
        
        garments = []
        
        # 提取上装
        upper_garment = self._extract_garment_region(image, upper_mask, GarmentType.UPPER_BODY)
        if upper_garment is not None:
            garments.append(upper_garment)
        
        # 提取下装
        lower_garment = self._extract_garment_region(image, lower_mask, GarmentType.LOWER_BODY)
        if lower_garment is not None:
            garments.append(lower_garment)
        
        # 创建完整掩码
        full_mask = np.zeros((h, w), dtype=np.uint8)
        for g in garments:
            full_mask = cv2.bitwise_or(full_mask, g.mask)
        
        # 创建可视化
        visualization = self._create_visualization(image, garments)
        
        return SegmentationResult(
            garments=garments,
            full_mask=full_mask,
            visualization=visualization,
            person_mask=person_mask
        )
    
    def _get_person_mask(self, image: np.ndarray) -> np.ndarray:
        """获取人体分割掩码"""
        if self.use_mediapipe:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.segmenter.process(image_rgb)
            
            if results.segmentation_mask is not None:
                mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                return mask
        
        # 备用方法：使用GrabCut
        return self._grabcut_segmentation(image)
    
    def _grabcut_segmentation(self, image: np.ndarray) -> np.ndarray:
        """使用GrabCut进行前景分割"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        
        # 初始化矩形（假设人在中间）
        rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9))
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            return mask2 * 255
        except:
            return np.ones((h, w), dtype=np.uint8) * 255
    
    def _split_body_regions(
        self, 
        person_mask: np.ndarray, 
        keypoints: np.ndarray,
        h: int, w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据关键点划分上下身区域"""
        upper_mask = person_mask.copy()
        lower_mask = person_mask.copy()
        
        # MediaPipe关键点索引
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        
        # 计算分界线（髋部位置）
        if len(keypoints) > max(LEFT_HIP, RIGHT_HIP):
            hip_y = int((keypoints[LEFT_HIP, 1] + keypoints[RIGHT_HIP, 1]) / 2)
        else:
            hip_y = h // 2
        
        # 划分区域
        upper_mask[hip_y:, :] = 0
        lower_mask[:hip_y, :] = 0
        
        return upper_mask, lower_mask
    
    def _extract_garment_region(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        garment_type: GarmentType
    ) -> Optional[GarmentRegion]:
        """提取服装区域"""
        # 形态学操作清理掩码
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 选择最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 1000:  # 面积太小则忽略
            return None
        
        # 计算边界框
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        bbox = (x, y, x + bw, y + bh)
        
        # 创建精确掩码
        precise_mask = np.zeros_like(mask)
        cv2.drawContours(precise_mask, [largest_contour], -1, 255, -1)
        
        # 裁剪服装图像
        cropped = image[y:y+bh, x:x+bw].copy()
        cropped_mask = precise_mask[y:y+bh, x:x+bw]
        
        # 应用掩码（透明背景）
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = cropped_mask
        
        return GarmentRegion(
            garment_type=garment_type,
            mask=precise_mask,
            bbox=bbox,
            contour=largest_contour,
            area=area,
            confidence=0.8,
            cropped_image=cropped_rgba
        )
    
    def _create_visualization(self, image: np.ndarray, garments: List[GarmentRegion]) -> np.ndarray:
        """创建分割可视化"""
        vis = image.copy()
        
        colors = {
            GarmentType.UPPER_BODY: (0, 255, 0),    # 绿色
            GarmentType.LOWER_BODY: (255, 0, 0),    # 蓝色
            GarmentType.FULL_BODY: (0, 255, 255),   # 黄色
            GarmentType.OUTERWEAR: (255, 0, 255),   # 紫色
            GarmentType.DRESS: (0, 165, 255),       # 橙色
        }
        
        for garment in garments:
            color = colors.get(garment.garment_type, (128, 128, 128))
            
            # 绘制半透明掩码
            overlay = vis.copy()
            overlay[garment.mask > 0] = color
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            
            # 绘制轮廓
            cv2.drawContours(vis, [garment.contour], -1, color, 2)
            
            # 绘制边界框和标签
            x1, y1, x2, y2 = garment.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{garment.garment_type.value}"
            cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis
    
    def segment_standalone_garment(self, image: np.ndarray) -> Optional[GarmentRegion]:
        """
        分割单独的服装图片（不在人体上）
        
        Args:
            image: 服装图片
            
        Returns:
            GarmentRegion对象
        """
        h, w = image.shape[:2]
        
        # 转换到LAB颜色空间进行更好的分割
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 使用Otsu阈值分割
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 如果背景是白色，反转掩码
        if np.mean(image) > 200:
            mask = cv2.bitwise_not(mask)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 找最大轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        bbox = (x, y, x + bw, y + bh)
        
        precise_mask = np.zeros_like(mask)
        cv2.drawContours(precise_mask, [largest_contour], -1, 255, -1)
        
        # 裁剪
        cropped = image[y:y+bh, x:x+bw].copy()
        cropped_mask = precise_mask[y:y+bh, x:x+bw]
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = cropped_mask
        
        # 根据宽高比判断类型
        aspect_ratio = bh / (bw + 1e-6)
        if aspect_ratio > 1.5:
            garment_type = GarmentType.FULL_BODY
        elif aspect_ratio > 0.8:
            garment_type = GarmentType.UPPER_BODY
        else:
            garment_type = GarmentType.LOWER_BODY
        
        return GarmentRegion(
            garment_type=garment_type,
            mask=precise_mask,
            bbox=bbox,
            contour=largest_contour,
            area=area,
            confidence=0.7,
            cropped_image=cropped_rgba
        )
    
    def visualize_segmentation(self, image: np.ndarray, garment: GarmentRegion) -> np.ndarray:
        """
        生成单个服装的分割可视化
        
        Args:
            image: 原始图像
            garment: 服装区域
            
        Returns:
            可视化图像
        """
        vis = image.copy()
        
        # 创建彩色遮罩
        color_mask = np.zeros_like(image)
        color_mask[garment.mask > 0] = (147, 112, 219)  # 紫色
        
        # 混合
        vis = cv2.addWeighted(vis, 0.6, color_mask, 0.4, 0)
        
        # 绘制轮廓
        cv2.drawContours(vis, [garment.contour], -1, (0, 255, 0), 2)
        
        # 绘制边界框
        x1, y1, x2, y2 = garment.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # 添加标签
        label = f"{garment.garment_type.value}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(vis, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 255), -1)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return vis
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'segmenter'):
            self.segmenter.close()


def extract_garment_from_person(
    image: np.ndarray, 
    person_keypoints: np.ndarray
) -> SegmentationResult:
    """
    从人体图像中提取服装
    
    Args:
        image: 人体图像
        person_keypoints: 人体关键点
        
    Returns:
        分割结果
    """
    segmenter = GarmentSegmenter()
    result = segmenter.segment(image, person_keypoints)
    segmenter.close()
    return result


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        if image is not None:
            segmenter = GarmentSegmenter()
            result = segmenter.segment(image)
            
            print(f"检测到 {len(result.garments)} 个服装区域")
            for g in result.garments:
                print(f"  - {g.garment_type.value}: 面积={g.area:.0f}, 置信度={g.confidence:.2f}")
            
            cv2.imwrite("segmentation_result.jpg", result.visualization)
            print("可视化结果已保存到 segmentation_result.jpg")
            
            segmenter.close()

