"""
服装语义理解模块
从服装图片中识别款式、颜色、材质等语义信息
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image
import colorsys


@dataclass
class ClothSemanticResult:
    """服装语义分析结果"""
    # 服装类型
    cloth_type: str  # 如: 上衣、裤子、裙子、连衣裙等
    cloth_type_confidence: float
    
    # 款式属性
    style_attributes: Dict[str, str]  # 如: {袖长: 长袖, 领型: 圆领, ...}
    
    # 主要颜色
    dominant_colors: List[Dict]  # [{color: "#FFFFFF", name: "白色", ratio: 0.6}, ...]
    
    # 材质预测
    material: str  # 如: 棉、涤纶、丝绸等
    material_confidence: float
    
    # 图案/纹理
    pattern: str  # 如: 纯色、条纹、格子、印花等
    pattern_confidence: float
    
    # 分割掩码
    segmentation_mask: Optional[np.ndarray] = None
    
    # 边界框
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


# 服装类型定义
CLOTH_TYPES = [
    'T恤', '衬衫', '毛衣', '卫衣', '外套', '夹克', '西装', '羽绒服',
    '裤子', '牛仔裤', '短裤', '运动裤',
    '裙子', '短裙', '长裙', '连衣裙',
    '其他'
]

# 款式属性定义
STYLE_ATTRIBUTES = {
    '袖长': ['无袖', '短袖', '七分袖', '长袖'],
    '领型': ['圆领', 'V领', '翻领', '立领', '连帽', '高领', '一字领'],
    '版型': ['修身', '常规', '宽松', '超宽松'],
    '长度': ['短款', '常规', '中长款', '长款'],
    '开襟': ['套头', '拉链', '纽扣', '系带'],
}

# 材质类型
MATERIALS = ['棉', '涤纶', '丝绸', '羊毛', '麻', '牛仔', '皮革', '尼龙', '雪纺', '针织', '混纺']

# 图案类型
PATTERNS = ['纯色', '条纹', '格子', '印花', '碎花', '几何', '字母/数字', '卡通', '扎染', '渐变']

# 颜色名称映射
COLOR_NAMES = {
    (255, 255, 255): '白色',
    (0, 0, 0): '黑色',
    (128, 128, 128): '灰色',
    (255, 0, 0): '红色',
    (0, 255, 0): '绿色',
    (0, 0, 255): '蓝色',
    (255, 255, 0): '黄色',
    (255, 165, 0): '橙色',
    (128, 0, 128): '紫色',
    (255, 192, 203): '粉色',
    (165, 42, 42): '棕色',
    (0, 128, 128): '青色',
    (245, 245, 220): '米色',
    (0, 0, 128): '藏青色',
    (128, 0, 0): '酒红色',
}


class ClothClassifier(nn.Module):
    """
    服装分类器
    基于ResNet50进行多任务分类
    """
    
    def __init__(self, num_cloth_types: int = len(CLOTH_TYPES)):
        super(ClothClassifier, self).__init__()
        
        # 使用预训练的ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 提取特征层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 多任务分类头
        self.fc_cloth_type = nn.Linear(2048, num_cloth_types)
        self.fc_material = nn.Linear(2048, len(MATERIALS))
        self.fc_pattern = nn.Linear(2048, len(PATTERNS))
        
        # 款式属性分类头
        self.style_heads = nn.ModuleDict()
        for attr_name, attr_values in STYLE_ATTRIBUTES.items():
            self.style_heads[attr_name] = nn.Linear(2048, len(attr_values))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, 224, 224)
            
        Returns:
            各分类任务的logits
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        outputs = {
            'cloth_type': self.fc_cloth_type(features),
            'material': self.fc_material(features),
            'pattern': self.fc_pattern(features),
        }
        
        for attr_name, head in self.style_heads.items():
            outputs[f'style_{attr_name}'] = head(features)
        
        return outputs


class ColorAnalyzer:
    """颜色分析器"""
    
    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors
    
    def analyze(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        分析图像的主要颜色
        
        Args:
            image: BGR图像
            mask: 可选的掩码，只分析掩码区域
            
        Returns:
            主要颜色列表
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用掩码
        if mask is not None:
            mask_bool = mask > 0
            pixels = image_rgb[mask_bool]
        else:
            pixels = image_rgb.reshape(-1, 3)
        
        if len(pixels) == 0:
            return []
        
        # 使用K-means聚类提取主要颜色
        from sklearn.cluster import KMeans
        
        n_clusters = min(self.n_colors, len(pixels) // 10 + 1)
        if n_clusters < 1:
            n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 计算每个聚类的比例
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        total = counts.sum()
        
        colors = []
        for label, count in zip(labels, counts):
            center = kmeans.cluster_centers_[label].astype(int)
            ratio = count / total
            
            # 转换为十六进制
            hex_color = '#{:02x}{:02x}{:02x}'.format(center[0], center[1], center[2])
            
            # 获取颜色名称
            color_name = self._get_color_name(center)
            
            colors.append({
                'color': hex_color,
                'rgb': center.tolist(),
                'name': color_name,
                'ratio': float(ratio)
            })
        
        # 按比例排序
        colors.sort(key=lambda x: x['ratio'], reverse=True)
        
        return colors
    
    def _get_color_name(self, rgb: np.ndarray) -> str:
        """获取最接近的颜色名称"""
        min_dist = float('inf')
        closest_name = '其他'
        
        for ref_rgb, name in COLOR_NAMES.items():
            dist = np.sqrt(np.sum((np.array(ref_rgb) - rgb) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        
        return closest_name


class ClothSegmenter:
    """
    服装分割器
    使用简单的GrabCut或深度学习方法分割服装区域
    """
    
    def __init__(self):
        pass
    
    def segment(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        分割服装区域
        
        Args:
            image: BGR图像
            bbox: 可选的边界框 (x, y, w, h)
            
        Returns:
            二值掩码
        """
        h, w = image.shape[:2]
        
        if bbox is None:
            # 自动估计服装区域（假设服装在图像中央）
            margin = 0.1
            bbox = (
                int(w * margin),
                int(h * margin),
                int(w * (1 - 2 * margin)),
                int(h * (1 - 2 * margin))
            )
        
        # 使用GrabCut进行分割
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (bbox[0], bbox[1], bbox[2], bbox[3])
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        except:
            # 如果GrabCut失败，使用简单的阈值方法
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = mask // 255
        
        # 形态学操作清理掩码
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask * 255


class ClothSemanticAnalyzer:
    """
    服装语义分析器
    整合分类、颜色分析和分割功能
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # 初始化分类器
        self.classifier = ClothClassifier().to(self.device)
        self.classifier.eval()
        
        # 初始化颜色分析器
        self.color_analyzer = ColorAnalyzer(n_colors=5)
        
        # 初始化分割器
        self.segmenter = ClothSegmenter()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def analyze(self, image: np.ndarray) -> ClothSemanticResult:
        """
        分析服装图像
        
        Args:
            image: BGR格式的输入图像
            
        Returns:
            ClothSemanticResult对象
        """
        h, w = image.shape[:2]
        
        # 1. 服装分割
        mask = self.segmenter.segment(image)
        
        # 2. 颜色分析
        dominant_colors = self.color_analyzer.analyze(image, mask)
        
        # 3. 服装分类
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
        
        # 解析分类结果
        cloth_type_idx = outputs['cloth_type'].argmax(dim=1).item()
        cloth_type = CLOTH_TYPES[cloth_type_idx]
        cloth_type_conf = F.softmax(outputs['cloth_type'], dim=1).max().item()
        
        material_idx = outputs['material'].argmax(dim=1).item()
        material = MATERIALS[material_idx]
        material_conf = F.softmax(outputs['material'], dim=1).max().item()
        
        pattern_idx = outputs['pattern'].argmax(dim=1).item()
        pattern = PATTERNS[pattern_idx]
        pattern_conf = F.softmax(outputs['pattern'], dim=1).max().item()
        
        # 款式属性
        style_attributes = {}
        for attr_name, attr_values in STYLE_ATTRIBUTES.items():
            key = f'style_{attr_name}'
            if key in outputs:
                idx = outputs[key].argmax(dim=1).item()
                style_attributes[attr_name] = attr_values[idx]
        
        # 计算边界框
        if mask.max() > 0:
            coords = np.where(mask > 0)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        else:
            bbox = (0, 0, w, h)
        
        return ClothSemanticResult(
            cloth_type=cloth_type,
            cloth_type_confidence=cloth_type_conf,
            style_attributes=style_attributes,
            dominant_colors=dominant_colors,
            material=material,
            material_confidence=material_conf,
            pattern=pattern,
            pattern_confidence=pattern_conf,
            segmentation_mask=mask,
            bounding_box=bbox
        )
    
    def to_dict(self, result: ClothSemanticResult) -> Dict:
        """将结果转换为字典格式"""
        return {
            'cloth_type': result.cloth_type,
            'cloth_type_confidence': result.cloth_type_confidence,
            'style_attributes': result.style_attributes,
            'dominant_colors': result.dominant_colors,
            'material': result.material,
            'material_confidence': result.material_confidence,
            'pattern': result.pattern,
            'pattern_confidence': result.pattern_confidence,
            'bounding_box': result.bounding_box,
        }
    
    def visualize(self, image: np.ndarray, result: ClothSemanticResult) -> np.ndarray:
        """可视化分析结果"""
        output = image.copy()
        
        # 绘制分割掩码边界
        if result.segmentation_mask is not None:
            contours, _ = cv2.findContours(
                result.segmentation_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        
        # 绘制边界框
        if result.bounding_box:
            x, y, w, h = result.bounding_box
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 添加文字标签
        label = f"{result.cloth_type} ({result.cloth_type_confidence:.0%})"
        cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output


def process_cloth_image(image_path: str) -> Dict:
    """
    处理服装图像的便捷函数
    
    Args:
        image_path: 图片路径
        
    Returns:
        包含语义分析信息的字典
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    analyzer = ClothSemanticAnalyzer()
    result = analyzer.analyze(image)
    
    return analyzer.to_dict(result)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = process_cloth_image(image_path)
        print("服装语义分析结果:")
        print(f"  类型: {result['cloth_type']} ({result['cloth_type_confidence']:.0%})")
        print(f"  材质: {result['material']}")
        print(f"  图案: {result['pattern']}")
        print(f"  款式: {result['style_attributes']}")
        print(f"  主要颜色: {[c['name'] for c in result['dominant_colors'][:3]]}")
    else:
        print("用法: python cloth_semantic.py <服装图片路径>")

