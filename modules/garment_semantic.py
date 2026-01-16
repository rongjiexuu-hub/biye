"""
服装语义理解模块
识别服装的款式、颜色、材质等属性
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import colorsys


class GarmentCategory(Enum):
    """服装大类"""
    TOP = "上装"
    BOTTOM = "下装"
    DRESS = "连衣裙"
    OUTERWEAR = "外套"
    SUIT = "套装"
    UNKNOWN = "未知"


class TopStyle(Enum):
    """上装款式"""
    T_SHIRT = "T恤"
    SHIRT = "衬衫"
    BLOUSE = "女衬衫"
    SWEATER = "毛衣"
    HOODIE = "卫衣"
    TANK_TOP = "背心"
    POLO = "Polo衫"
    UNKNOWN = "未知"


class BottomStyle(Enum):
    """下装款式"""
    JEANS = "牛仔裤"
    TROUSERS = "西裤"
    SHORTS = "短裤"
    SKIRT = "裙子"
    LEGGINGS = "打底裤"
    SWEATPANTS = "运动裤"
    UNKNOWN = "未知"


class Material(Enum):
    """材质类型"""
    COTTON = "棉"
    DENIM = "牛仔布"
    SILK = "丝绸"
    WOOL = "羊毛"
    POLYESTER = "涤纶"
    LEATHER = "皮革"
    LINEN = "亚麻"
    CHIFFON = "雪纺"
    UNKNOWN = "未知"


class Pattern(Enum):
    """图案类型"""
    SOLID = "纯色"
    STRIPED = "条纹"
    PLAID = "格子"
    FLORAL = "花卉"
    POLKA_DOT = "波点"
    PRINTED = "印花"
    UNKNOWN = "未知"


class SleeveLength(Enum):
    """袖长"""
    SLEEVELESS = "无袖"
    SHORT = "短袖"
    THREE_QUARTER = "七分袖"
    LONG = "长袖"
    UNKNOWN = "未知"


class NecklineType(Enum):
    """领型"""
    CREW = "圆领"
    V_NECK = "V领"
    COLLAR = "翻领"
    TURTLENECK = "高领"
    SCOOP = "船领"
    OFF_SHOULDER = "一字领"
    UNKNOWN = "未知"


@dataclass
class ColorInfo:
    """颜色信息"""
    name: str                     # 颜色名称
    hex_code: str                 # 十六进制颜色码
    rgb: Tuple[int, int, int]     # RGB值
    percentage: float             # 占比
    is_primary: bool = False      # 是否主色


@dataclass
class GarmentSemantics:
    """服装语义信息"""
    category: GarmentCategory                    # 服装大类
    style: str                                   # 具体款式
    colors: List[ColorInfo]                      # 颜色列表
    primary_color: str                           # 主色调名称
    material: Material                           # 材质
    pattern: Pattern                             # 图案
    sleeve_length: Optional[SleeveLength] = None # 袖长
    neckline: Optional[NecklineType] = None      # 领型
    attributes: Dict[str, str] = field(default_factory=dict)  # 其他属性
    confidence: float = 0.0                      # 整体置信度


class ColorAnalyzer:
    """颜色分析器"""
    
    # 基础颜色名称映射
    COLOR_NAMES = {
        'red': '红色',
        'orange': '橙色',
        'yellow': '黄色',
        'green': '绿色',
        'cyan': '青色',
        'blue': '蓝色',
        'purple': '紫色',
        'pink': '粉色',
        'brown': '棕色',
        'white': '白色',
        'gray': '灰色',
        'black': '黑色',
        'beige': '米色',
        'navy': '藏青色',
        'burgundy': '酒红色',
        'olive': '橄榄绿',
        'coral': '珊瑚色',
        'turquoise': '青绿色',
    }
    
    def __init__(self, n_colors: int = 5):
        """
        初始化颜色分析器
        
        Args:
            n_colors: 提取的颜色数量
        """
        self.n_colors = n_colors
    
    def analyze(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[ColorInfo]:
        """
        分析图像中的主要颜色
        
        Args:
            image: BGR图像
            mask: 可选的掩码
            
        Returns:
            颜色信息列表
        """
        # 应用掩码
        if mask is not None:
            # 只保留掩码区域的像素
            pixels = image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            return []
        
        # 使用K-means聚类提取主要颜色
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        k = min(self.n_colors, len(pixels))
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 计算每个颜色的占比
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        colors = []
        for i, center in enumerate(centers):
            if i in unique:
                idx = np.where(unique == i)[0][0]
                percentage = counts[idx] / total
                
                # BGR转RGB
                rgb = tuple(int(c) for c in center[::-1])
                hex_code = '#{:02x}{:02x}{:02x}'.format(*rgb)
                name = self._get_color_name(rgb)
                
                colors.append(ColorInfo(
                    name=name,
                    hex_code=hex_code,
                    rgb=rgb,
                    percentage=percentage,
                    is_primary=(percentage > 0.3)
                ))
        
        # 按占比排序
        colors.sort(key=lambda x: x.percentage, reverse=True)
        
        # 标记主色
        if colors:
            colors[0].is_primary = True
        
        return colors
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """获取颜色名称"""
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h *= 360
        
        # 根据HSV值判断颜色
        if v < 0.15:
            return '黑色'
        elif v > 0.9 and s < 0.1:
            return '白色'
        elif s < 0.15:
            if v < 0.5:
                return '深灰色'
            else:
                return '浅灰色'
        else:
            # 根据色相判断
            if h < 15 or h >= 345:
                if s > 0.5 and v > 0.5:
                    return '红色'
                else:
                    return '酒红色'
            elif h < 45:
                if v > 0.8:
                    return '橙色'
                else:
                    return '棕色'
            elif h < 70:
                return '黄色'
            elif h < 150:
                if v < 0.5:
                    return '橄榄绿'
                else:
                    return '绿色'
            elif h < 200:
                return '青色'
            elif h < 260:
                if v < 0.4:
                    return '藏青色'
                else:
                    return '蓝色'
            elif h < 290:
                return '紫色'
            else:
                return '粉色'


class TextureAnalyzer:
    """纹理分析器"""
    
    def analyze(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Pattern, float]:
        """
        分析图像纹理/图案
        
        Args:
            image: BGR图像
            mask: 可选掩码
            
        Returns:
            (图案类型, 置信度)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # 计算边缘
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 计算方差（纹理复杂度）
        variance = np.var(gray[gray > 0]) if np.any(gray > 0) else 0
        
        # 检测条纹（使用霍夫变换）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        has_stripes = lines is not None and len(lines) > 10
        
        # 检测圆形（波点）
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=30)
        has_dots = circles is not None and len(circles[0]) > 5
        
        # 判断图案类型
        if edge_density < 0.05 and variance < 1000:
            return Pattern.SOLID, 0.9
        elif has_stripes:
            return Pattern.STRIPED, 0.7
        elif has_dots:
            return Pattern.POLKA_DOT, 0.7
        elif edge_density > 0.15:
            return Pattern.PRINTED, 0.6
        else:
            return Pattern.SOLID, 0.5


class MaterialClassifier:
    """材质分类器"""
    
    def __init__(self):
        """初始化材质分类器"""
        # 材质特征描述
        self.material_features = {
            Material.DENIM: {'color': ['蓝色', '藏青色'], 'texture': 'rough'},
            Material.COTTON: {'color': None, 'texture': 'medium'},
            Material.SILK: {'color': None, 'texture': 'smooth', 'shiny': True},
            Material.WOOL: {'color': None, 'texture': 'fuzzy'},
            Material.LEATHER: {'color': ['黑色', '棕色'], 'texture': 'smooth', 'shiny': True},
        }
    
    def classify(
        self, 
        image: np.ndarray, 
        colors: List[ColorInfo],
        pattern: Pattern
    ) -> Tuple[Material, float]:
        """
        分类材质
        
        Args:
            image: 服装图像
            colors: 颜色信息
            pattern: 图案类型
            
        Returns:
            (材质类型, 置信度)
        """
        primary_color = colors[0].name if colors else '未知'
        
        # 简单规则判断
        if primary_color in ['蓝色', '藏青色'] and pattern == Pattern.SOLID:
            # 可能是牛仔布
            return Material.DENIM, 0.6
        
        # 分析光泽度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        highlights = np.sum(gray > 240) / gray.size
        
        if highlights > 0.05:
            # 有光泽，可能是丝绸或皮革
            if primary_color in ['黑色', '棕色']:
                return Material.LEATHER, 0.5
            else:
                return Material.SILK, 0.5
        
        # 默认棉质
        return Material.COTTON, 0.4


class GarmentSemanticAnalyzer:
    """
    服装语义分析器
    综合分析服装的各种属性
    """
    
    def __init__(self):
        """初始化语义分析器"""
        self.color_analyzer = ColorAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.material_classifier = MaterialClassifier()
    
    def analyze(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        garment_type: str = "unknown"
    ) -> GarmentSemantics:
        """
        分析服装语义
        
        Args:
            image: 服装图像
            mask: 服装掩码
            garment_type: 服装类型提示
            
        Returns:
            GarmentSemantics对象
        """
        # 分析颜色
        colors = self.color_analyzer.analyze(image, mask)
        primary_color = colors[0].name if colors else "未知"
        
        # 分析图案
        pattern, pattern_conf = self.texture_analyzer.analyze(image, mask)
        
        # 分析材质
        material, material_conf = self.material_classifier.classify(image, colors, pattern)
        
        # 判断服装类别和款式
        category, style = self._classify_category(garment_type, image, mask)
        
        # 分析其他属性
        sleeve_length = None
        neckline = None
        
        if category in [GarmentCategory.TOP, GarmentCategory.DRESS]:
            sleeve_length = self._detect_sleeve_length(image, mask)
            neckline = self._detect_neckline(image, mask)
        
        # 计算总体置信度
        confidence = (pattern_conf + material_conf) / 2
        
        return GarmentSemantics(
            category=category,
            style=style,
            colors=colors,
            primary_color=primary_color,
            material=material,
            pattern=pattern,
            sleeve_length=sleeve_length,
            neckline=neckline,
            attributes={
                'color_count': str(len(colors)),
                'is_multicolor': str(len([c for c in colors if c.percentage > 0.15]) > 1)
            },
            confidence=confidence
        )
    
    def _classify_category(
        self, 
        type_hint: str, 
        image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[GarmentCategory, str]:
        """分类服装类别"""
        type_hint = type_hint.lower()
        
        if 'upper' in type_hint or 'top' in type_hint:
            category = GarmentCategory.TOP
            style = TopStyle.UNKNOWN.value
        elif 'lower' in type_hint or 'bottom' in type_hint:
            category = GarmentCategory.BOTTOM
            style = BottomStyle.UNKNOWN.value
        elif 'dress' in type_hint:
            category = GarmentCategory.DRESS
            style = "连衣裙"
        elif 'outer' in type_hint:
            category = GarmentCategory.OUTERWEAR
            style = "外套"
        else:
            # 根据图像宽高比判断
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    h = ys.max() - ys.min()
                    w = xs.max() - xs.min()
                    aspect = h / (w + 1e-6)
                    
                    if aspect > 2:
                        category = GarmentCategory.DRESS
                        style = "连衣裙"
                    elif aspect > 1:
                        category = GarmentCategory.TOP
                        style = TopStyle.UNKNOWN.value
                    else:
                        category = GarmentCategory.BOTTOM
                        style = BottomStyle.UNKNOWN.value
                else:
                    category = GarmentCategory.UNKNOWN
                    style = "未知"
            else:
                category = GarmentCategory.UNKNOWN
                style = "未知"
        
        return category, style
    
    def _detect_sleeve_length(self, image: np.ndarray, mask: Optional[np.ndarray]) -> SleeveLength:
        """检测袖长"""
        # 简化实现：基于图像宽度估计
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                w = xs.max() - xs.min()
                h = ys.max() - ys.min()
                ratio = w / (h + 1e-6)
                
                if ratio > 1.5:
                    return SleeveLength.LONG
                elif ratio > 1.0:
                    return SleeveLength.THREE_QUARTER
                elif ratio > 0.6:
                    return SleeveLength.SHORT
                else:
                    return SleeveLength.SLEEVELESS
        
        return SleeveLength.UNKNOWN
    
    def _detect_neckline(self, image: np.ndarray, mask: Optional[np.ndarray]) -> NecklineType:
        """检测领型"""
        # 简化实现：返回未知
        # 完整实现需要深度学习模型
        return NecklineType.UNKNOWN
    
    def to_dict(self, semantics: GarmentSemantics) -> Dict:
        """转换为字典格式"""
        # 创建置信度分数
        confidence_scores = {}
        for color in semantics.colors:
            confidence_scores[color.name] = color.percentage
        
        return {
            'category': semantics.category.value,
            'garment_type': semantics.category.value,
            'style': semantics.style,
            'primary_color': semantics.primary_color,
            'colors': [
                {
                    'name': c.name,
                    'hex': c.hex_code,
                    'rgb': c.rgb,
                    'percentage': round(c.percentage, 3),
                    'is_primary': c.is_primary
                }
                for c in semantics.colors
            ],
            'material': semantics.material.value,
            'pattern': semantics.pattern.value,
            'sleeve_type': semantics.sleeve_length.value if semantics.sleeve_length else '--',
            'sleeve_length': semantics.sleeve_length.value if semantics.sleeve_length else None,
            'neckline': semantics.neckline.value if semantics.neckline else '--',
            'attributes': semantics.attributes,
            'confidence': round(semantics.confidence, 3),
            'confidence_scores': confidence_scores
        }


def analyze_garment(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """
    分析服装语义的便捷函数
    
    Args:
        image: 服装图像
        mask: 服装掩码
        
    Returns:
        语义信息字典
    """
    analyzer = GarmentSemanticAnalyzer()
    semantics = analyzer.analyze(image, mask)
    return analyzer.to_dict(semantics)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        if image is not None:
            result = analyze_garment(image)
            
            print("服装语义分析结果:")
            print(f"  类别: {result['category']}")
            print(f"  款式: {result['style']}")
            print(f"  主色调: {result['primary_color']}")
            print(f"  材质: {result['material']}")
            print(f"  图案: {result['pattern']}")
            print(f"  颜色列表:")
            for c in result['colors']:
                print(f"    - {c['name']} ({c['hex']}): {c['percentage']*100:.1f}%")

