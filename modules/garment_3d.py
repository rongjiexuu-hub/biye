"""
服装3D化模块
将2D服装图像转换为3D模型，并可穿戴到人体模型上
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class UV_Mapping:
    """UV映射信息"""
    uv_coords: np.ndarray       # UV坐标 (N, 2)
    face_indices: np.ndarray    # 对应的面片索引
    texture_image: np.ndarray   # 纹理图像


@dataclass
class Garment3D:
    """3D服装模型"""
    vertices: np.ndarray        # 顶点坐标 (V, 3)
    faces: np.ndarray           # 面片索引 (F, 3)
    normals: np.ndarray         # 法向量 (V, 3)
    uv_coords: np.ndarray       # UV坐标 (V, 2)
    texture: np.ndarray         # 纹理图像
    garment_type: str           # 服装类型
    properties: Dict            # 物理属性


class GarmentWarpingEngine:
    """
    服装纹理变形引擎
    使用 TPS (Thin-Plate Spline) 算法将2D服装图像扭曲以贴合3D模型
    """
    
    def __init__(self):
        pass
        
    def warp_garment_to_mask(
        self,
        garment_image: np.ndarray,
        source_points: np.ndarray,
        target_points: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        使用 TPS 算法进行图像扭曲
        
        Args:
            garment_image: 源服装图像
            source_points: 源控制点 (N, 2)
            target_points: 目标控制点 (N, 2)
            output_size: 输出图像尺寸 (W, H)
            
        Returns:
            扭曲后的图像
        """
        if len(source_points) < 3:
            return cv2.resize(garment_image, output_size)
            
        # OpenCV TPS 变形器
        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # 格式化点为 (1, N, 2)
        source_points = source_points.reshape(1, -1, 2).astype(np.float32)
        target_points = target_points.reshape(1, -1, 2).astype(np.float32)
        
        # 创建匹配对象
        matches = [cv2.DMatch(i, i, 0) for i in range(source_points.shape[1])]
        
        # 估计变形
        tps.estimateTransformation(target_points, source_points, matches)
        
        # 应用变形
        warped = tps.warpImage(garment_image)
        
        return cv2.resize(warped, output_size)

class GarmentUVMapper:
    """
    服装UV映射器
    将2D服装图像映射到3D人体模型的UV空间
    """
    
    # SMPL模型的身体部位UV区域（近似值）
    BODY_UV_REGIONS = {
        'torso_front': {'u': (0.3, 0.7), 'v': (0.2, 0.6)},
        'torso_back': {'u': (0.3, 0.7), 'v': (0.6, 1.0)},
        'left_arm': {'u': (0.0, 0.3), 'v': (0.2, 0.5)},
        'right_arm': {'u': (0.7, 1.0), 'v': (0.2, 0.5)},
        'left_leg': {'u': (0.3, 0.5), 'v': (0.0, 0.2)},
        'right_leg': {'u': (0.5, 0.7), 'v': (0.0, 0.2)},
    }
    
    def __init__(self, texture_size: int = 1024):
        """
        初始化UV映射器
        
        Args:
            texture_size: 纹理图像尺寸
        """
        self.texture_size = texture_size
    
    def create_garment_texture(
        self,
        garment_image: np.ndarray,
        garment_mask: np.ndarray,
        garment_type: str
    ) -> np.ndarray:
        """
        创建服装纹理图 - 直接使用裁剪的服装图像
        
        Args:
            garment_image: 服装图像 (BGR)
            garment_mask: 服装掩码
            garment_type: 服装类型 (upper_body, lower_body, etc.)
            
        Returns:
            UV纹理图像
        """
        # 裁剪服装区域
        ys, xs = np.where(garment_mask > 0)
        if len(ys) == 0:
            # 返回默认纹理
            texture = np.ones((self.texture_size, self.texture_size, 4), dtype=np.uint8) * 200
            texture[:, :, 3] = 255
            return texture
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        # 添加边距
        margin = 10
        y_min = max(0, y_min - margin)
        y_max = min(garment_image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(garment_image.shape[1], x_max + margin)
        
        cropped = garment_image[y_min:y_max, x_min:x_max]
        cropped_mask = garment_mask[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            texture = np.ones((self.texture_size, self.texture_size, 4), dtype=np.uint8) * 200
            texture[:, :, 3] = 255
            return texture
        
        # 保持宽高比缩放到纹理大小
        h, w = cropped.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = self.texture_size
            new_h = int(self.texture_size / aspect)
        else:
            new_h = self.texture_size
            new_w = int(self.texture_size * aspect)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建纹理图（居中放置）
        texture = np.zeros((self.texture_size, self.texture_size, 4), dtype=np.uint8)
        
        # 填充背景
        texture[:, :, :3] = 200  # 浅灰背景
        texture[:, :, 3] = 255
        
        # 计算放置位置
        y_offset = (self.texture_size - new_h) // 2
        x_offset = (self.texture_size - new_w) // 2
        
        # 放置裁剪的服装
        if resized.shape[2] == 3:
            texture[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :3] = resized
            texture[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 3] = resized_mask
        else:
            texture[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return texture
    
    def project_texture_to_mesh(
        self,
        texture: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        uv_coords: np.ndarray
    ) -> np.ndarray:
        """
        将纹理投影到网格上
        
        Args:
            texture: 纹理图像
            vertices: 顶点坐标
            faces: 面片索引
            uv_coords: UV坐标
            
        Returns:
            顶点颜色 (V, 4)
        """
        h, w = texture.shape[:2]
        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        for i, uv in enumerate(uv_coords):
            u, v = uv
            # 限制UV范围
            u = max(0, min(1, u))
            v = max(0, min(1, v))
            
            # 转换到像素坐标
            px = int(u * (w - 1))
            py = int(v * (h - 1))
            
            vertex_colors[i] = texture[py, px]
        
        return vertex_colors


class GarmentMeshGenerator:
    """
    服装网格生成器
    从2D服装图像生成3D网格
    """
    
    def __init__(self):
        """初始化网格生成器"""
        pass
    
    def generate_from_contour(
        self,
        contour: np.ndarray,
        depth_scale: float = 0.15,
        num_layers: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从2D轮廓生成3D网格
        使用改进的深度挤压方法，模拟服装的立体感
        
        Args:
            contour: 2D轮廓点 (N, 2)
            depth_scale: 深度缩放
            num_layers: 深度层数
            
        Returns:
            (vertices, faces)
        """
        # 简化轮廓
        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        points_2d = simplified.reshape(-1, 2).astype(np.float32)
        
        # 归一化坐标到[-1, 1]
        center = points_2d.mean(axis=0)
        points_2d = points_2d - center
        scale = np.max(np.abs(points_2d)) + 1e-8
        points_2d = points_2d / scale
        
        n_points = len(points_2d)
        
        if n_points < 3:
            # 如果点太少，创建一个简单的矩形
            return self.generate_cloth_mesh(1.0, 1.5, 15)
        
        # 计算轮廓的边界框
        min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
        min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
        
        # 生成前表面网格（使用三角剖分）
        front_vertices, front_faces = self._triangulate_contour(points_2d)
        
        # 为前表面添加深度变化（模拟布料的褶皱）
        for i, v in enumerate(front_vertices):
            # 根据位置计算深度（中心凸起，边缘平坦）
            dist_from_center = np.sqrt(v[0]**2 + v[1]**2)
            depth = depth_scale * (1.0 - dist_from_center * 0.5)
            
            # 添加一些褶皱效果
            wrinkle = 0.02 * np.sin(v[1] * 8) * np.cos(v[0] * 6)
            front_vertices[i, 2] = depth + wrinkle
        
        # 生成后表面（镜像）
        back_vertices = front_vertices.copy()
        back_vertices[:, 2] = -back_vertices[:, 2] * 0.3  # 后面更平坦
        
        # 合并顶点
        n_front = len(front_vertices)
        vertices = np.vstack([front_vertices, back_vertices])
        
        # 后表面的面片（反转方向）
        back_faces = front_faces.copy() + n_front
        back_faces = back_faces[:, ::-1]  # 反转顶点顺序
        
        # 合并面片
        faces = np.vstack([front_faces, back_faces])
        
        # 添加侧面连接
        side_faces = self._generate_side_faces(points_2d, n_front)
        if len(side_faces) > 0:
            faces = np.vstack([faces, side_faces])
        
        return vertices.astype(np.float32), faces.astype(np.int32)
    
    def _triangulate_contour(self, points_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对2D轮廓进行三角剖分
        """
        from scipy.spatial import Delaunay
        
        # 创建内部网格点
        min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
        min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
        
        # 生成内部点
        resolution = 15
        x_range = np.linspace(min_x, max_x, resolution)
        y_range = np.linspace(min_y, max_y, resolution)
        
        internal_points = []
        contour_path = cv2.convexHull(points_2d)
        
        for x in x_range:
            for y in y_range:
                # 检查点是否在轮廓内
                if cv2.pointPolygonTest(contour_path, (float(x), float(y)), False) >= 0:
                    internal_points.append([x, y])
        
        if len(internal_points) < 3:
            internal_points = points_2d.tolist()
        
        # 合并轮廓点和内部点
        all_points = np.vstack([points_2d, np.array(internal_points)])
        
        # 去重
        all_points = np.unique(all_points, axis=0)
        
        if len(all_points) < 3:
            # 回退到简单网格
            return self._simple_grid(min_x, max_x, min_y, max_y)
        
        try:
            # Delaunay三角剖分
            tri = Delaunay(all_points)
            faces = tri.simplices
            
            # 过滤掉轮廓外的三角形
            valid_faces = []
            for face in faces:
                centroid = all_points[face].mean(axis=0)
                if cv2.pointPolygonTest(contour_path, (float(centroid[0]), float(centroid[1])), False) >= 0:
                    valid_faces.append(face)
            
            if len(valid_faces) == 0:
                valid_faces = faces.tolist()
            
            # 添加Z坐标
            vertices_3d = np.hstack([all_points, np.zeros((len(all_points), 1))])
            
            return vertices_3d, np.array(valid_faces)
        except:
            return self._simple_grid(min_x, max_x, min_y, max_y)
    
    def _simple_grid(self, min_x, max_x, min_y, max_y, resolution=10):
        """创建简单的网格"""
        vertices = []
        for j in range(resolution + 1):
            for i in range(resolution + 1):
                x = min_x + (max_x - min_x) * i / resolution
                y = min_y + (max_y - min_y) * j / resolution
                vertices.append([x, y, 0])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        faces = []
        for j in range(resolution):
            for i in range(resolution):
                v0 = j * (resolution + 1) + i
                v1 = v0 + 1
                v2 = v0 + (resolution + 1)
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return vertices, np.array(faces)
    
    def _generate_side_faces(self, contour_points: np.ndarray, n_front: int) -> np.ndarray:
        """生成侧面连接面片"""
        n_points = len(contour_points)
        faces = []
        
        # 连接前后表面的边缘
        for i in range(min(n_points, 50)):  # 限制数量避免太多面片
            next_i = (i + 1) % n_points
            
            v0 = i
            v1 = next_i
            v2 = i + n_front
            v3 = next_i + n_front
            
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
        
        return np.array(faces) if faces else np.array([]).reshape(0, 3)
    
    def generate_cloth_mesh(
        self,
        width: float,
        height: float,
        resolution: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成简单的布料网格（矩形）
        
        Args:
            width: 宽度
            height: 高度
            resolution: 分辨率
            
        Returns:
            (vertices, faces)
        """
        vertices = []
        for j in range(resolution + 1):
            for i in range(resolution + 1):
                x = (i / resolution - 0.5) * width
                y = (j / resolution - 0.5) * height
                # 添加轻微的深度变化模拟布料
                z = 0.05 * np.sin(x * 3) * np.cos(y * 3)
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        faces = []
        for j in range(resolution):
            for i in range(resolution):
                v0 = j * (resolution + 1) + i
                v1 = v0 + 1
                v2 = v0 + (resolution + 1)
                v3 = v2 + 1
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        faces = np.array(faces, dtype=np.int32)
        
        return vertices, faces


class Garment3DReconstructor:
    """
    服装3D重建器
    综合处理2D服装到3D模型的转换
    支持使用预定义的服装模板
    """
    
    # 服装物理属性预设
    MATERIAL_PROPERTIES = {
        '棉': {'density': 0.3, 'stiffness': 0.5, 'friction': 0.8},
        '牛仔布': {'density': 0.5, 'stiffness': 0.8, 'friction': 0.9},
        '丝绸': {'density': 0.1, 'stiffness': 0.2, 'friction': 0.3},
        '羊毛': {'density': 0.4, 'stiffness': 0.6, 'friction': 0.7},
        '涤纶': {'density': 0.2, 'stiffness': 0.4, 'friction': 0.5},
        '皮革': {'density': 0.6, 'stiffness': 0.9, 'friction': 0.6},
    }
    
    # 服装类型到模板的映射（优先使用用户提供的专业模板）
    TEMPLATE_MAPPING = {
        'upper_body': 'tshirt_white.obj',  # 优先使用专业模板
        'UPPER_BODY': 'tshirt_white.obj',
        't-shirt': 'tshirt_white.obj',
        'shirt': 'tshirt_white.obj',
        'tshirt': 'tshirt_white.obj',
        'lower_body': 'pants_template.obj',
        'LOWER_BODY': 'pants_template.obj',
        'full_body': 'dress_template.obj',
        'FULL_BODY': 'dress_template.obj',
        'dress': 'dress_template.obj',
    }
    
    # 备用模板（如果专业模板太大或不可用）
    FALLBACK_TEMPLATES = {
        'tshirt_white.obj': 'tshirt_template.obj',
    }
    
    def __init__(self, template_dir: str = 'models/garment_templates'):
        """
        初始化3D重建器
        
        Args:
            template_dir: 服装模板目录
        """
        self.uv_mapper = GarmentUVMapper()
        self.mesh_generator = GarmentMeshGenerator()
        self.warping_engine = GarmentWarpingEngine()
        self.template_dir = template_dir
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """加载所有可用的服装模板"""
        if not os.path.exists(self.template_dir):
            print(f"模板目录不存在: {self.template_dir}")
            return
        
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.obj'):
                filepath = os.path.join(self.template_dir, filename)
                try:
                    vertices, faces, uv_coords = self._load_obj(filepath)
                    self.templates[filename] = {
                        'vertices': vertices,
                        'faces': faces,
                        'uv_coords': uv_coords
                    }
                    print(f"  加载服装模板: {filename} (顶点: {len(vertices)}, 面片: {len(faces)})")
                except Exception as e:
                    print(f"  加载模板失败 {filename}: {e}")
    
    def _load_obj(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载OBJ文件"""
        vertices = []
        faces = []
        uv_coords = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v' and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt' and len(parts) >= 3:
                    uv_coords.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    # 解析面片 (可能是 v, v/vt, v/vt/vn 格式)
                    face_verts = []
                    for p in parts[1:]:
                        idx = p.split('/')[0]
                        face_verts.append(int(idx) - 1)  # OBJ索引从1开始
                    if len(face_verts) >= 3:
                        faces.append(face_verts[:3])
        
        vertices = np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32)
        faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
        
        # 如果没有UV坐标，生成默认的
        if len(uv_coords) == 0:
            uv_coords = self._generate_uv_coords(vertices)
        else:
            uv_coords = np.array(uv_coords, dtype=np.float32)
            # 确保UV坐标数量与顶点数量匹配
            if len(uv_coords) < len(vertices):
                uv_coords = self._generate_uv_coords(vertices)
        
        return vertices, faces, uv_coords
    
    def reconstruct(
        self,
        garment_image: np.ndarray,
        garment_mask: np.ndarray,
        garment_contour: np.ndarray,
        garment_type: str,
        material: str = '棉'
    ) -> Garment3D:
        """
        从2D服装图像重建3D模型
        优先使用模板，如果没有模板则从轮廓生成
        
        Args:
            garment_image: 服装图像
            garment_mask: 服装掩码
            garment_contour: 服装轮廓
            garment_type: 服装类型
            material: 材质类型
            
        Returns:
            Garment3D对象
        """
        # 尝试使用模板
        template_name = self.TEMPLATE_MAPPING.get(garment_type)
        
        if template_name and template_name in self.templates:
            print(f"  使用服装模板: {template_name}")
            template = self.templates[template_name]
            vertices = template['vertices'].copy()
            faces = template['faces'].copy()
            uv_coords = template['uv_coords'].copy()
            
            # 使用 TPS 扭曲图像以贴合模板（可选增强）
            # 这里简单起见，我们先保持原样，后续可以在 drape_on_body 中进一步优化
        else:
            print(f"  没有找到模板 '{garment_type}'，从轮廓生成网格")
            # 回退到从轮廓生成
            vertices, faces = self.mesh_generator.generate_from_contour(
                garment_contour,
                depth_scale=0.15,
                num_layers=8
            )
            uv_coords = self._generate_uv_coords(vertices)
        
        # 使用 TPS 算法对服装图片进行预扭曲，使其更好地贴合 UV
        warped_image = self._warp_garment_image(garment_image, garment_mask, garment_contour)
        
        # 计算法向量
        normals = self._compute_normals(vertices, faces)
        
        # 如果UV坐标长度不匹配顶点，重新生成
        if len(uv_coords) != len(vertices):
            uv_coords = self._generate_uv_coords(vertices)
        
        # 创建纹理
        texture = self.uv_mapper.create_garment_texture(
            warped_image, garment_mask, garment_type
        )
        
        # 获取物理属性
        properties = self.MATERIAL_PROPERTIES.get(material, self.MATERIAL_PROPERTIES['棉'])
        properties['material'] = material
        
        return Garment3D(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            texture=texture,
            garment_type=garment_type,
            properties=properties
        )

    def _warp_garment_image(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        contour: np.ndarray
    ) -> np.ndarray:
        """
        内部方法：使用TPS对服装图像进行初步扭曲
        """
        if contour is None or len(contour) < 10:
            return image
            
        # 1. 提取源控制点（轮廓上的均匀采样点）
        src_points = contour[::len(contour)//10].reshape(-1, 2)
        
        # 2. 提取目标控制点（规则化的矩形或目标形状）
        # 这里简单起见，我们将其映射到一个规则的边界框上
        x, y, w, h = cv2.boundingRect(contour)
        target_points = src_points.copy()
        # 将点推向边界框边缘，模拟展开效果
        center_x, center_y = x + w/2, y + h/2
        for i in range(len(target_points)):
            # 简单的径向拉伸
            dx = target_points[i, 0] - center_x
            dy = target_points[i, 1] - center_y
            target_points[i, 0] = center_x + dx * 1.1
            target_points[i, 1] = center_y + dy * 1.1
            
        # 3. 执行扭曲
        try:
            warped = self.warping_engine.warp_garment_to_mask(
                image, src_points, target_points, (image.shape[1], image.shape[0])
            )
            return warped
        except Exception as e:
            print(f"  TPS扭曲失败: {e}，使用原图")
            return image
    
    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """计算顶点法向量"""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            for idx in face:
                normals[idx] += normal
        
        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)
        
        return normals
    
    def _generate_uv_coords(self, vertices: np.ndarray) -> np.ndarray:
        """生成UV坐标（简单的平面投影）"""
        # 使用XY平面投影
        xy = vertices[:, :2]
        
        # 归一化到[0, 1]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        
        uv = (xy - xy_min) / (xy_max - xy_min + 1e-8)
        
        return uv.astype(np.float32)
    
    def drape_on_body(
        self,
        garment: Garment3D,
        body_vertices: np.ndarray,
        body_faces: np.ndarray,
        body_joints: np.ndarray,
        smpl_params: Optional[Dict] = None
    ) -> Garment3D:
        """
        将服装穿戴到人体模型上
        使用线性混合蒙皮 (LBS) 算法使衣服跟随人体姿态
        
        Args:
            garment: 服装3D模型
            body_vertices: 当前姿态下的人体顶点 (V, 3)
            body_faces: 人体面片
            body_joints: 当前姿态下的关节位置 (24, 3)
            smpl_params: SMPL参数（包含旋转矩阵等，用于LBS）
            
        Returns:
            调整后的服装模型
        """
        # 如果提供了SMPL参数和姿态矩阵，执行LBS变形
        if smpl_params and 'pose_matrices' in smpl_params:
            print("  执行基于LBS的服装变形...")
            return self._apply_lbs_to_garment(garment, smpl_params)
            
        # 否则回退到简单的缩放和平移适配
        print("  执行简单的几何适配...")
        # 根据服装类型确定穿戴区域
        if garment.garment_type == 'upper_body':
            target_region = self._get_upper_body_region(body_vertices, body_joints)
        elif garment.garment_type == 'lower_body':
            target_region = self._get_lower_body_region(body_vertices, body_joints)
        else:
            target_region = body_vertices
        
        scaled_vertices = self._fit_garment_to_body(garment.vertices, target_region)
        offset_vertices = self._offset_from_body(scaled_vertices, body_vertices, offset=0.015)
        
        return Garment3D(
            vertices=offset_vertices,
            faces=garment.faces,
            normals=garment.normals,
            uv_coords=garment.uv_coords,
            texture=garment.texture,
            garment_type=garment.garment_type,
            properties=garment.properties
        )

    def _apply_lbs_to_garment(self, garment: Garment3D, smpl_params: Dict) -> Garment3D:
        """
        对服装应用线性混合蒙皮
        """
        # 获取SMPL数据
        pose_matrices = smpl_params['pose_matrices'] # (24, 4, 4) 蒙皮矩阵
        rest_pose_vertices = smpl_params.get('rest_vertices') # T-Pose下的SMPL顶点
        smpl_weights = smpl_params.get('weights') # SMPL蒙皮权重 (6890, 24)
        
        if rest_pose_vertices is None or smpl_weights is None:
            # 如果没有休息姿态数据，无法准确计算权重
            return garment
            
        # 1. 为服装顶点寻找最近的SMPL顶点并传递权重
        # 这是一个简化版的自动权重绑定
        num_garment_v = len(garment.vertices)
        
        # 适配服装到T-Pose人体（休息姿态）
        target_region = rest_pose_vertices
        if garment.garment_type == 'upper_body':
            # 简化：上衣对应上半身
            mask = rest_pose_vertices[:, 1] > rest_pose_vertices[:, 1].mean()
            target_region = rest_pose_vertices[mask]
            
        # 预先适配到T-Pose
        v_rest = self._fit_garment_to_body(garment.vertices, target_region)
        
        # 寻找最近点权重
        from scipy.spatial import cKDTree
        tree = cKDTree(rest_pose_vertices)
        _, indices = tree.query(v_rest)
        garment_weights = smpl_weights[indices]
        
        # 2. 执行LBS变形
        # v_final = sum(w_i * M_i * v_rest)
        v_final = np.zeros_like(v_rest)
        v_homo = np.hstack([v_rest, np.ones((num_garment_v, 1))])
        
        for i in range(24):
            w = garment_weights[:, i:i+1]
            if np.any(w > 1e-4):
                # 应用该关节的变换矩阵
                transformed = v_homo @ pose_matrices[i].T
                v_final += w * transformed[:, :3]
                
        return Garment3D(
            vertices=v_final,
            faces=garment.faces,
            normals=garment.normals,
            uv_coords=garment.uv_coords,
            texture=garment.texture,
            garment_type=garment.garment_type,
            properties=garment.properties
        )
    
    def _get_upper_body_region(self, body_vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """获取上身区域顶点"""
        # SMPL关节索引
        PELVIS = 0
        NECK = 12
        
        if len(joints) > max(PELVIS, NECK):
            y_min = joints[PELVIS, 1]
            y_max = joints[NECK, 1]
        else:
            y_center = body_vertices[:, 1].mean()
            y_min = y_center
            y_max = body_vertices[:, 1].max()
        
        mask = (body_vertices[:, 1] >= y_min) & (body_vertices[:, 1] <= y_max)
        return body_vertices[mask]
    
    def _get_lower_body_region(self, body_vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """获取下身区域顶点"""
        PELVIS = 0
        LEFT_ANKLE = 7
        
        if len(joints) > max(PELVIS, LEFT_ANKLE):
            y_min = joints[LEFT_ANKLE, 1]
            y_max = joints[PELVIS, 1]
        else:
            y_center = body_vertices[:, 1].mean()
            y_min = body_vertices[:, 1].min()
            y_max = y_center
        
        mask = (body_vertices[:, 1] >= y_min) & (body_vertices[:, 1] <= y_max)
        return body_vertices[mask]
    
    def _fit_garment_to_body(self, garment_vertices: np.ndarray, target_region: np.ndarray) -> np.ndarray:
        """将服装缩放适配到目标区域"""
        # 计算边界
        g_min = garment_vertices.min(axis=0)
        g_max = garment_vertices.max(axis=0)
        g_center = (g_min + g_max) / 2
        g_size = g_max - g_min
        
        t_min = target_region.min(axis=0)
        t_max = target_region.max(axis=0)
        t_center = (t_min + t_max) / 2
        t_size = t_max - t_min
        
        # 计算缩放比例
        scale = t_size / (g_size + 1e-8)
        scale = min(scale) * 0.9  # 稍微小一点以留出空间
        
        # 应用变换
        scaled = (garment_vertices - g_center) * scale + t_center
        
        return scaled
    
    def _offset_from_body(
        self, 
        garment_vertices: np.ndarray, 
        body_vertices: np.ndarray,
        offset: float = 0.02
    ) -> np.ndarray:
        """将服装顶点偏移到人体外部"""
        # 简化实现：沿法向量方向偏移
        # 实际应该计算到人体表面的距离并推出
        
        result = garment_vertices.copy()
        
        # 计算每个服装顶点到最近人体顶点的方向
        for i, gv in enumerate(garment_vertices):
            dists = np.linalg.norm(body_vertices - gv, axis=1)
            nearest_idx = np.argmin(dists)
            
            if dists[nearest_idx] < offset:
                # 太近了，需要推出
                direction = gv - body_vertices[nearest_idx]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                result[i] = body_vertices[nearest_idx] + direction * offset
        
        return result
    
    def export_obj(self, garment: Garment3D, filepath: str):
        """导出为OBJ格式"""
        with open(filepath, 'w') as f:
            f.write(f"# Garment 3D Model: {garment.garment_type}\n")
            f.write(f"# Vertices: {len(garment.vertices)}\n")
            f.write(f"# Faces: {len(garment.faces)}\n\n")
            
            # 写入顶点
            for v in garment.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # 写入法向量
            for n in garment.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            f.write("\n")
            
            # 写入UV坐标
            for uv in garment.uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            
            f.write("\n")
            
            # 写入面片
            for face in garment.faces:
                # OBJ索引从1开始
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                       f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                       f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
        
        # 保存纹理
        if garment.texture is not None:
            texture_path = filepath.replace('.obj', '_texture.png')
            cv2.imwrite(texture_path, garment.texture)
    
    def visualize_projection(self, garment: Garment3D) -> np.ndarray:
        """
        可视化3D投影效果 - 使用透视投影和光照
        支持大型模型（自动采样）
        
        Args:
            garment: 3D服装模型
            
        Returns:
            可视化图像
        """
        img_size = 512
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 25  # 深色背景
        
        # 绘制渐变背景
        for i in range(img_size):
            ratio = i / img_size
            color = int(25 + 15 * ratio)
            img[i, :] = [color, color, int(color * 1.1)]
        
        vertices = garment.vertices.copy()
        faces = garment.faces.copy()
        normals = garment.normals
        
        if len(vertices) == 0 or len(faces) == 0:
            cv2.putText(img, "No mesh data", (img_size//2-80, img_size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return img
        
        # 对大型模型进行采样
        MAX_FACES = 10000
        if len(faces) > MAX_FACES:
            # 随机采样面片
            np.random.seed(42)  # 固定随机种子保证一致性
            sample_idx = np.random.choice(len(faces), MAX_FACES, replace=False)
            faces = faces[sample_idx]
            print(f"  大型模型采样: {len(garment.faces)} -> {len(faces)} 面片")
        
        # 应用旋转（展示3D效果）
        angle_x = np.radians(-20)  # X轴旋转
        angle_y = np.radians(30)   # Y轴旋转
        
        # 旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        
        R = Ry @ Rx
        vertices = vertices @ R.T
        if len(normals) == len(garment.vertices):
            normals = normals @ R.T
        
        # 归一化到图像空间
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_center = (v_min + v_max) / 2
        v_scale = max(v_max - v_min) + 1e-8
        
        vertices = (vertices - v_center) / v_scale
        
        # 透视投影参数
        fov = 1.2
        margin = 60
        scale = (img_size - 2 * margin) / 2
        
        # 投影到2D
        z_offset = 2.0
        proj_x = (vertices[:, 0] * fov / (vertices[:, 2] + z_offset) * scale + img_size / 2).astype(int)
        proj_y = (img_size / 2 - vertices[:, 1] * fov / (vertices[:, 2] + z_offset) * scale).astype(int)
        
        # 计算面片深度并排序（画家算法）
        face_depths = []
        valid_faces = []
        for i, face in enumerate(faces):
            # 检查索引有效性
            if np.any(face >= len(vertices)) or np.any(face < 0):
                continue
            avg_z = vertices[face, 2].mean()
            face_depths.append((len(valid_faces), avg_z))
            valid_faces.append(face)
        
        if len(valid_faces) == 0:
            cv2.putText(img, "Invalid mesh data", (img_size//2-100, img_size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            return img
        
        face_depths.sort(key=lambda x: x[1])  # 从远到近排序
        
        # 光照方向
        light_dir = np.array([0.3, 0.6, 0.8])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # 基础颜色（从纹理采样或使用默认色）
        base_color = np.array([230, 230, 235])  # 白色/浅灰
        
        # 如果有纹理，提取主色调
        if garment.texture is not None and garment.texture.size > 0:
            tex = garment.texture
            if len(tex.shape) >= 3 and tex.shape[2] >= 3:
                h, w = tex.shape[:2]
                center = tex[h//4:3*h//4, w//4:3*w//4, :3]
                if center.size > 0:
                    # 过滤掉接近背景色的像素
                    mask = np.all(center > 30, axis=2) & np.all(center < 250, axis=2)
                    if np.any(mask):
                        base_color = center[mask].mean(axis=0)[::-1]
        
        # 绘制面片
        for face_idx, _ in face_depths:
            face = valid_faces[face_idx]
            
            pts = np.array([
                [proj_x[face[0]], proj_y[face[0]]],
                [proj_x[face[1]], proj_y[face[1]]],
                [proj_x[face[2]], proj_y[face[2]]]
            ], dtype=np.int32)
            
            # 检查点是否在图像范围内
            if np.any(pts < -50) or np.any(pts >= img_size + 50):
                continue
            
            # 限制到图像范围
            pts = np.clip(pts, 0, img_size - 1)
            
            # 计算面片法向量
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(face_normal)
            if norm_len < 1e-8:
                continue
            face_normal = face_normal / norm_len
            
            # 背面剔除（允许一些边缘面片）
            if face_normal[2] < -0.3:
                continue
            
            # 计算光照强度
            intensity = max(0.25, min(1.0, np.dot(face_normal, light_dir) * 0.6 + 0.4))
            
            # 计算最终颜色
            color = (base_color * intensity).astype(int)
            color = tuple(int(c) for c in np.clip(color, 0, 255))
            
            cv2.fillPoly(img, [pts], color)
        
        # 添加信息
        cv2.putText(img, "3D Garment Model", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Type: {garment.garment_type}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(img, f"Vertices: {len(garment.vertices):,}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(img, f"Faces: {len(garment.faces):,}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return img
    
    def visualize_uv_map(self, garment: Garment3D) -> np.ndarray:
        """
        可视化UV贴图 - 显示展开的服装纹理
        
        Args:
            garment: 3D服装模型
            
        Returns:
            UV贴图可视化
        """
        img_size = 512
        
        # 如果有纹理，使用纹理进行展示
        if garment.texture is not None and garment.texture.size > 0:
            texture = garment.texture
            
            # 调整大小
            if texture.shape[0] != img_size or texture.shape[1] != img_size:
                texture = cv2.resize(texture, (img_size, img_size))
            
            # 处理RGBA
            if texture.shape[2] == 4:
                rgb = texture[:, :, :3]
                alpha = texture[:, :, 3:4] / 255.0
                # 创建棋盘格背景
                bg = self._create_checker_background(img_size)
                result = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            else:
                result = texture.copy()
            
            # 添加UV网格线
            uv_coords = garment.uv_coords
            if len(uv_coords) > 0:
                for face in garment.faces[::5]:  # 每5个面画一个，避免太密集
                    pts = np.array([
                        [int(uv_coords[face[0], 0] * (img_size - 1)), 
                         int((1 - uv_coords[face[0], 1]) * (img_size - 1))],
                        [int(uv_coords[face[1], 0] * (img_size - 1)), 
                         int((1 - uv_coords[face[1], 1]) * (img_size - 1))],
                        [int(uv_coords[face[2], 0] * (img_size - 1)), 
                         int((1 - uv_coords[face[2], 1]) * (img_size - 1))]
                    ], dtype=np.int32)
                    cv2.polylines(result, [pts], True, (100, 100, 150), 1, cv2.LINE_AA)
            
            # 添加标题
            overlay = result.copy()
            cv2.rectangle(overlay, (0, 0), (200, 50), (30, 30, 30), -1)
            result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
            cv2.putText(result, "UV Texture Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return result
        
        # 如果没有纹理，创建UV布局可视化
        img = self._create_checker_background(img_size)
        
        # 绘制UV三角形轮廓
        uv_coords = garment.uv_coords
        for i, face in enumerate(garment.faces):
            pts = np.array([
                [int(uv_coords[face[0], 0] * (img_size - 1)), 
                 int((1 - uv_coords[face[0], 1]) * (img_size - 1))],
                [int(uv_coords[face[1], 0] * (img_size - 1)), 
                 int((1 - uv_coords[face[1], 1]) * (img_size - 1))],
                [int(uv_coords[face[2], 0] * (img_size - 1)), 
                 int((1 - uv_coords[face[2], 1]) * (img_size - 1))]
            ], dtype=np.int32)
            
            # 根据面片索引变化颜色
            hue = int((i / len(garment.faces)) * 180)
            color = cv2.cvtColor(np.uint8([[[hue, 180, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
            color = tuple(int(c) for c in color)
            
            cv2.fillPoly(img, [pts], color)
            cv2.polylines(img, [pts], True, (50, 50, 50), 1)
        
        # 添加标题
        cv2.putText(img, "UV Layout", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        
        return img
    
    def _create_checker_background(self, size: int, cell_size: int = 32) -> np.ndarray:
        """创建棋盘格背景"""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(0, size, cell_size):
            for j in range(0, size, cell_size):
                if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                    img[i:i+cell_size, j:j+cell_size] = [200, 200, 200]
                else:
                    img[i:i+cell_size, j:j+cell_size] = [170, 170, 170]
        return img
    
    def export_texture(self, garment: Garment3D, filepath: str):
        """
        导出纹理图像
        
        Args:
            garment: 3D服装模型
            filepath: 输出路径
        """
        if garment.texture is not None:
            cv2.imwrite(filepath, garment.texture)
        else:
            # 创建一个简单的默认纹理
            default_texture = np.ones((512, 512, 3), dtype=np.uint8) * 180
            cv2.putText(default_texture, "No Texture", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            cv2.imwrite(filepath, default_texture)
    
    def to_dict(self, garment: Garment3D) -> Dict:
        """转换为字典格式"""
        return {
            'garment_type': garment.garment_type,
            'num_vertices': len(garment.vertices),
            'num_faces': len(garment.faces),
            'properties': garment.properties,
            'has_texture': garment.texture is not None,
            'bounds': {
                'min': garment.vertices.min(axis=0).tolist(),
                'max': garment.vertices.max(axis=0).tolist()
            }
        }


def reconstruct_garment_3d(
    image: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    garment_type: str = 'upper_body',
    material: str = '棉'
) -> Garment3D:
    """
    服装3D重建便捷函数
    
    Args:
        image: 服装图像
        mask: 服装掩码
        contour: 服装轮廓
        garment_type: 服装类型
        material: 材质
        
    Returns:
        Garment3D对象
    """
    reconstructor = Garment3DReconstructor()
    return reconstructor.reconstruct(image, mask, contour, garment_type, material)


if __name__ == '__main__':
    # 测试代码
    print("服装3D化模块加载成功")
    
    # 创建测试服装
    mesh_gen = GarmentMeshGenerator()
    vertices, faces = mesh_gen.generate_cloth_mesh(1.0, 1.5, resolution=10)
    
    print(f"生成布料网格:")
    print(f"  顶点数: {len(vertices)}")
    print(f"  面片数: {len(faces)}")

