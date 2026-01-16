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
    
    # 服装类型到模板的映射
    TEMPLATE_MAPPING = {
        'upper_body': 'tshirt_template.obj',
        'UPPER_BODY': 'tshirt_template.obj',
        't-shirt': 'tshirt_template.obj',
        'shirt': 'tshirt_template.obj',
        'tshirt': 'tshirt_template.obj',
        'lower_body': 'pants_template.obj',
        'LOWER_BODY': 'pants_template.obj',
        'full_body': 'dress_template.obj',
        'FULL_BODY': 'dress_template.obj',
        'dress': 'dress_template.obj',
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
                except Exception as e:
                    print(f"加载模板失败 {filename}: {e}")

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
        template_name = self.TEMPLATE_MAPPING.get(garment_type)

        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            vertices = template['vertices'].copy()
            faces = template['faces'].copy()
            uv_coords = template['uv_coords'].copy()
        else:
            vertices, faces = self.mesh_generator.generate_from_contour(garment_contour)
            uv_coords = self._generate_uv_coords(vertices)

        warped_image = self._warp_garment_image(garment_image, garment_mask, garment_contour)

        normals = self._compute_normals(vertices, faces)

        if len(uv_coords) != len(vertices):
            uv_coords = self._generate_uv_coords(vertices)

        texture = self.uv_mapper.create_garment_texture(warped_image, garment_mask, garment_type)

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

        src_points = contour[::len(contour)//10].reshape(-1, 2)

        x, y, w, h = cv2.boundingRect(contour)
        target_points = src_points.copy()
        center_x, center_y = x + w/2, y + h/2
        for i in range(len(target_points)):
            dx = target_points[i, 0] - center_x
            dy = target_points[i, 1] - center_y
            target_points[i, 0] = center_x + dx * 1.1
            target_points[i, 1] = center_y + dy * 1.1

        try:
            warped = self.warping_engine.warp_garment_to_mask(
                image, src_points, target_points, (image.shape[1], image.shape[0])
            )
            return warped
        except Exception as e:
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

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)

        return normals

    def _generate_uv_coords(self, vertices: np.ndarray) -> np.ndarray:
        """生成UV坐标（简单的平面投影）"""
        xy = vertices[:, :2]

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
        """
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

    def _get_upper_body_region(self, body_vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """获取上身区域顶点"""
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
        g_min = garment_vertices.min(axis=0)
        g_max = garment_vertices.max(axis=0)
        g_center = (g_min + g_max) / 2
        g_size = g_max - g_min

        t_min = target_region.min(axis=0)
        t_max = target_region.max(axis=0)
        t_center = (t_min + t_max) / 2
        t_size = t_max - t_min

        scale = t_size / (g_size + 1e-8)
        scale = min(scale) * 0.9

        scaled = (garment_vertices - g_center) * scale + t_center

        return scaled

    def _offset_from_body(
        self,
        garment_vertices: np.ndarray,
        body_vertices: np.ndarray,
        offset: float = 0.02
    ) -> np.ndarray:
        """将服装顶点偏移到人体外部"""
        result = garment_vertices.copy()

        for i, gv in enumerate(garment_vertices):
            dists = np.linalg.norm(body_vertices - gv, axis=1)
            nearest_idx = np.argmin(dists)

            if dists[nearest_idx] < offset:
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

            for v in garment.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            for n in garment.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write("\n")

            for uv in garment.uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            f.write("\n")

            for face in garment.faces:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                       f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                       f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")

        if garment.texture is not None:
            texture_path = filepath.replace('.obj', '_texture.png')
            cv2.imwrite(texture_path, garment.texture)

    def visualize_projection(self, garment: Garment3D) -> np.ndarray:
        """
        可视化3D投影效果 - 使用透视投影和光照
        """
        img_size = 512
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 25

        vertices = garment.vertices.copy()
        faces = garment.faces.copy()
        normals = garment.normals

        angle_x = np.radians(-20)
        angle_y = np.radians(30)

        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])

        R = Ry @ Rx
        vertices = vertices @ R.T
        if len(normals) == len(garment.vertices):
            normals = normals @ R.T

        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_center = (v_min + v_max) / 2
        v_scale = max(v_max - v_min) + 1e-8

        vertices = (vertices - v_center) / v_scale

        fov = 1.2
        margin = 60
        scale = (img_size - 2 * margin) / 2

        z_offset = 2.0
        proj_x = (vertices[:, 0] * fov / (vertices[:, 2] + z_offset) * scale + img_size / 2).astype(int)
        proj_y = (img_size / 2 - vertices[:, 1] * fov / (vertices[:, 2] + z_offset) * scale).astype(int)

        face_depths = []
        for i, face in enumerate(faces):
            avg_z = vertices[face, 2].mean()
            face_depths.append((i, avg_z))

        face_depths.sort(key=lambda x: x[1])

        light_dir = np.array([0.3, 0.6, 0.8])
        light_dir = light_dir / np.linalg.norm(light_dir)

        base_color = np.array([230, 230, 235])

        for face_idx, _ in face_depths:
            face = faces[face_idx]
            pts = np.array([[proj_x[face[0]], proj_y[face[0]]], [proj_x[face[1]], proj_y[face[1]]], [proj_x[face[2]], proj_y[face[2]]]], dtype=np.int32)

            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(face_normal)
            if norm_len < 1e-8:
                continue
            face_normal = face_normal / norm_len

            if face_normal[2] < -0.3:
                continue

            intensity = max(0.25, min(1.0, np.dot(face_normal, light_dir) * 0.6 + 0.4))

            color = (base_color * intensity).astype(int)
            color = tuple(int(c) for c in np.clip(color, 0, 255))

            cv2.fillPoly(img, [pts], color)

        cv2.putText(img, "3D Garment Model", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

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
    """
    reconstructor = Garment3DReconstructor()
    return reconstructor.reconstruct(image, mask, contour, garment_type, material)


if __name__ == '__main__':
    print("服装3D化模块加载成功")

    mesh_gen = GarmentMeshGenerator()
    vertices, faces = mesh_gen.generate_cloth_mesh(1.0, 1.5, resolution=10)

    print(f"生成布料网格:")
    print(f"  顶点数: {len(vertices)}")
    print(f"  面片数: {len(faces)}")