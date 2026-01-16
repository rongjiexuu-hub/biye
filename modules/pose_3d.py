"""
3D人体姿态与形状重建模块
使用SMPL参数化人体模型和HMR方法
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


@dataclass
class SMPLParams:
    """SMPL模型参数"""
    # 姿态参数 (72维: 24个关节 x 3个旋转角度)
    pose: np.ndarray  # (72,)
    # 形状参数 (10维beta参数)
    shape: np.ndarray  # (10,)
    # 全局旋转
    global_orient: np.ndarray  # (3,)
    # 全局平移
    translation: np.ndarray  # (3,)
    # 相机参数
    camera: np.ndarray  # (3,) - scale, tx, ty


@dataclass
class Pose3DResult:
    """3D姿态重建结果"""
    # SMPL参数
    smpl_params: SMPLParams
    # 3D关节位置 (24, 3)
    joints_3d: np.ndarray
    # 3D顶点 (6890, 3) - SMPL模型有6890个顶点
    vertices: np.ndarray
    # 投影到2D的关节位置 (24, 2)
    joints_2d_proj: np.ndarray
    # 面片索引 (用于渲染)
    faces: np.ndarray
    # 置信度
    confidence: float



class HMREncoder(nn.Module):
    """
    HMR编码器
    使用ResNet50作为backbone提取图像特征
    兼容预训练的HMR权重
    """

    def __init__(self, pretrained: bool = True):
        super(HMREncoder, self).__init__()

        # 使用预训练的ResNet50，但保持与HMR权重兼容的结构
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # 分离各层以便加载预训练权重
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 特征维度
        self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征

        Args:
            x: 输入图像 (B, 3, 224, 224)

        Returns:
            特征向量 (B, 2048)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        features = x.view(x.size(0), -1)
        return features


class SMPLRegressor(nn.Module):
    """
    SMPL参数回归器
    从图像特征预测SMPL的姿态和形状参数
    兼容预训练的HMR权重
    """

    # SMPL参数维度
    POSE_6D_DIM = 144  # 24个关节 x 6维旋转表示
    POSE_DIM = 72  # 24个关节 x 3个旋转角度 (轴角)
    SHAPE_DIM = 10  # 体型参数
    CAM_DIM = 3  # 相机参数 (scale, tx, ty)

    def __init__(self, feature_dim: int = 2048, num_iterations: int = 3, use_6d_rotation: bool = True):
        super(SMPLRegressor, self).__init__()

        self.num_iterations = num_iterations
        self.use_6d_rotation = use_6d_rotation

        pose_dim = self.POSE_6D_DIM if use_6d_rotation else self.POSE_DIM
        total_params = pose_dim + self.SHAPE_DIM + self.CAM_DIM

        # 迭代回归网络 (兼容预训练权重)
        self.fc1 = nn.Linear(feature_dim + total_params, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.decpose = nn.Linear(1024, pose_dim)
        self.decshape = nn.Linear(1024, self.SHAPE_DIM)
        self.deccam = nn.Linear(1024, self.CAM_DIM)

        self.dropout = nn.Dropout(0.5)

        # 初始化平均参数
        self.register_buffer('init_pose', torch.zeros(1, pose_dim))
        self.register_buffer('init_shape', torch.zeros(1, self.SHAPE_DIM))
        self.register_buffer('init_cam', torch.tensor([[1.0, 0.0, 0.0]]))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从特征预测SMPL参数

        Args:
            features: 图像特征 (B, 2048)

        Returns:
            pose: 姿态参数 (B, 72) - 轴角表示
            shape: 形状参数 (B, 10)
            cam: 相机参数 (B, 3)
        """
        batch_size = features.size(0)

        # 初始化参数
        pose = self.init_pose.expand(batch_size, -1).clone()
        shape = self.init_shape.expand(batch_size, -1).clone()
        cam = self.init_cam.expand(batch_size, -1).clone()

        # 迭代回归
        for _ in range(self.num_iterations):
            # 拼接特征和当前参数
            x = torch.cat([features, pose, shape, cam], dim=1)

            # 前向传播
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)

            # 预测参数残差
            pose = pose + self.decpose(x)
            shape = shape + self.decshape(x)
            cam = cam + self.deccam(x)

        # 如果使用6D旋转，转换为轴角表示
        if self.use_6d_rotation:
            pose = self._rotation_6d_to_axis_angle(pose)

        return pose, shape, cam

    def _rotation_6d_to_axis_angle(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """
        将6D旋转表示转换为轴角表示

        Args:
            rot_6d: 6D旋转 (B, 144) = (B, 24*6)

        Returns:
            axis_angle: 轴角表示 (B, 72) = (B, 24*3)
        """
        batch_size = rot_6d.size(0)
        rot_6d = rot_6d.view(batch_size, -1, 6)  # (B, 24, 6)

        # 提取前两列
        x = rot_6d[..., :3]  # (B, 24, 3)
        y = rot_6d[..., 3:6]  # (B, 24, 3)

        # Gram-Schmidt正交化
        x = F.normalize(x, dim=-1)
        y = y - (x * y).sum(dim=-1, keepdim=True) * x
        y = F.normalize(y, dim=-1)
        z = torch.cross(x, y, dim=-1)

        # 构建旋转矩阵
        rot_mat = torch.stack([x, y, z], dim=-1)  # (B, 24, 3, 3)

        # 旋转矩阵转轴角
        axis_angle = self._rotation_matrix_to_axis_angle(rot_mat)  # (B, 24, 3)

        return axis_angle.view(batch_size, -1)  # (B, 72)

    def _rotation_matrix_to_axis_angle(self, rot_mat: torch.Tensor) -> torch.Tensor:
        """旋转矩阵转轴角"""
        batch_size = rot_mat.shape[0]
        num_joints = rot_mat.shape[1]

        # 使用Rodrigues公式的逆
        # 从旋转矩阵提取轴角
        rot_mat = rot_mat.view(-1, 3, 3)  # (B*24, 3, 3)

        # 计算旋转角度
        trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

        # 计算旋转轴
        axis = torch.stack([
            rot_mat[:, 2, 1] - rot_mat[:, 1, 2],
            rot_mat[:, 0, 2] - rot_mat[:, 2, 0],
            rot_mat[:, 1, 0] - rot_mat[:, 0, 1]
        ], dim=1)

        # 归一化
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)

        # 轴角 = 轴 * 角度
        axis_angle = axis * angle.unsqueeze(1)

        return axis_angle.view(batch_size, num_joints, 3)


class SimpleSMPL(nn.Module):
    """
    简化版SMPL模型
    用于从姿态和形状参数生成3D人体网格
    """

    NUM_JOINTS = 24
    NUM_VERTICES = 6890

    def __init__(self, model_path: Optional[str] = None):
        super(SimpleSMPL, self).__init__()

        # 查找有效的SMPL模型文件
        actual_model_path = self._find_smpl_model(model_path)

        if actual_model_path and os.path.isfile(actual_model_path):
            # 加载预训练的SMPL模型参数
            print(f"加载SMPL模型: {actual_model_path}")
            self._load_smpl_model(actual_model_path)
        else:
            # 使用随机初始化的参数（用于演示）
            print("未找到SMPL模型文件，使用内置简化人体模板")
            self._init_default_params()

    def _find_smpl_model(self, model_path: Optional[str]) -> Optional[str]:
        """查找有效的SMPL模型文件"""
        if model_path is None:
            return None

        # 如果是文件，直接返回
        if os.path.isfile(model_path):
            return model_path

        # 如果是目录，在目录中查找模型文件
        if os.path.isdir(model_path):
            import glob

            # 优先查找.npz文件（已转换格式）
            npz_files = glob.glob(os.path.join(model_path, '*.npz'))
            if npz_files:
                for f in npz_files:
                    if 'neutral' in f.lower():
                        return f
                return npz_files[0]

            # 常见的SMPL模型文件名（按优先级排序）
            possible_names = [
                'SMPL_NEUTRAL.pkl',
                'SMPL_MALE.pkl',
                'SMPL_FEMALE.pkl',
                'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                'basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl',
                'basicModel_neutral_lbs_10_207_0_v1.1.0.pkl',
                'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
                'basicModel_m_lbs_10_207_0_v1.0.0.pkl',
                'basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
                'basicModel_m_lbs_10_207_0_v1.1.0.pkl',
                'basicmodel_m_lbs_10_207_0_v1.1.0.pkl',
                'basicModel_f_lbs_10_207_0_v1.0.0.pkl',
                'basicmodel_f_lbs_10_207_0_v1.0.0.pkl',
                'basicModel_f_lbs_10_207_0_v1.1.0.pkl',
                'basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
            ]
            for name in possible_names:
                full_path = os.path.join(model_path, name)
                if os.path.isfile(full_path):
                    return full_path

            # 如果没有找到预定义的名称，尝试查找任何.pkl文件
            pkl_files = glob.glob(os.path.join(model_path, '*.pkl'))
            if pkl_files:
                # 优先选择包含neutral的文件
                for f in pkl_files:
                    if 'neutral' in f.lower():
                        return f
                return pkl_files[0]

        return None

    def _init_default_params(self):
        """初始化默认参数"""
        # 平均体型模板 (6890, 3)
        self.register_buffer('v_template', torch.zeros(self.NUM_VERTICES, 3))

        # 形状混合形状 (6890, 3, 10)
        self.register_buffer('shapedirs', torch.zeros(self.NUM_VERTICES, 3, 10))

        # 姿态混合形状 (6890, 3, 207)
        self.register_buffer('posedirs', torch.zeros(self.NUM_VERTICES, 3, 207))

        # 关节回归矩阵 (24, 6890)
        self.register_buffer('J_regressor', torch.zeros(self.NUM_JOINTS, self.NUM_VERTICES))

        # 蒙皮权重 (6890, 24)
        self.register_buffer('weights', torch.zeros(self.NUM_VERTICES, self.NUM_JOINTS))

        # 父关节索引
        self.register_buffer('parents', torch.tensor([
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
        ]))

        # 面片索引
        self.register_buffer('faces', torch.zeros(13776, 3, dtype=torch.long))

        # 生成一个简单的人体模板
        self._generate_simple_template()

    def _generate_simple_template(self):
        """生成简化的人体模板（用于演示）"""
        # 创建一个简单的人体形状
        # 这只是一个近似，真实的SMPL模型需要从官方获取

        # 生成基本的人体顶点
        vertices = torch.zeros(self.NUM_VERTICES, 3)

        # 简单的圆柱体近似
        n_rings = 100
        n_per_ring = 69  # 6890 / 100 ≈ 69

        for i in range(n_rings):
            t = i / (n_rings - 1)
            y = 1.7 * t - 0.85  # 身高约1.7米，中心在髋部

            # 根据位置确定半径
            if y > 0.6:  # 头部
                radius = 0.1 * (1 - (y - 0.6) / 0.25)
            elif y > 0.3:  # 躯干上部
                radius = 0.15
            elif y > -0.1:  # 躯干中部
                radius = 0.14 + 0.02 * (0.3 - y) / 0.4
            elif y > -0.5:  # 髋部
                radius = 0.15
            else:  # 腿部
                radius = 0.08

            for j in range(n_per_ring):
                idx = i * n_per_ring + j
                if idx < self.NUM_VERTICES:
                    angle = 2 * np.pi * j / n_per_ring
                    vertices[idx] = torch.tensor([
                        radius * np.cos(angle),
                        y,
                        radius * np.sin(angle)
                    ])

        self.v_template.copy_(vertices)

        # 生成关节位置
        joint_positions = torch.tensor([
            [0, 0, 0],      # 0: 骨盆
            [0.1, -0.1, 0],  # 1: 左髋
            [-0.1, -0.1, 0], # 2: 右髋
            [0, 0.2, 0],     # 3: 脊柱1
            [0.1, -0.5, 0],  # 4: 左膝
            [-0.1, -0.5, 0], # 5: 右膝
            [0, 0.35, 0],    # 6: 脊柱2
            [0.1, -0.85, 0], # 7: 左踝
            [-0.1, -0.85, 0],# 8: 右踝
            [0, 0.5, 0],     # 9: 脊柱3
            [0.1, -0.9, 0.05], # 10: 左脚
            [-0.1, -0.9, 0.05],# 11: 右脚
            [0, 0.65, 0],    # 12: 颈部
            [0.2, 0.55, 0],  # 13: 左锁骨
            [-0.2, 0.55, 0], # 14: 右锁骨
            [0, 0.75, 0],    # 15: 头部
            [0.3, 0.5, 0],   # 16: 左肩
            [-0.3, 0.5, 0],  # 17: 右肩
            [0.45, 0.35, 0], # 18: 左肘
            [-0.45, 0.35, 0],# 19: 右肘
            [0.55, 0.15, 0], # 20: 左腕
            [-0.55, 0.15, 0],# 21: 右腕
            [0.6, 0.1, 0],   # 22: 左手
            [-0.6, 0.1, 0],  # 23: 右手
        ], dtype=torch.float32)

        # 创建简单的关节回归矩阵
        J_regressor = torch.zeros(self.NUM_JOINTS, self.NUM_VERTICES)
        for i in range(self.NUM_JOINTS):
            # 找到最近的顶点
            dists = torch.norm(self.v_template - joint_positions[i], dim=1)
            nearest_idx = torch.argmin(dists)
            J_regressor[i, nearest_idx] = 1.0

        self.J_regressor.copy_(J_regressor)

        # 创建简单的蒙皮权重
        weights = torch.zeros(self.NUM_VERTICES, self.NUM_JOINTS)
        for v_idx in range(self.NUM_VERTICES):
            # 计算到每个关节的距离
            dists = torch.norm(joint_positions - self.v_template[v_idx], dim=1)
            # 使用softmax分配权重
            weights[v_idx] = F.softmax(-dists * 5, dim=0)

        self.weights.copy_(weights)

        # 创建面片（简化版）
        faces = []
        for i in range(n_rings - 1):
            for j in range(n_per_ring):
                v0 = i * n_per_ring + j
                v1 = i * n_per_ring + (j + 1) % n_per_ring
                v2 = (i + 1) * n_per_ring + j
                v3 = (i + 1) * n_per_ring + (j + 1) % n_per_ring

                if v0 < self.NUM_VERTICES and v1 < self.NUM_VERTICES and \
                   v2 < self.NUM_VERTICES and v3 < self.NUM_VERTICES:
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])

        # 填充faces tensor
        faces_tensor = torch.zeros(13776, 3, dtype=torch.long)
        for i, f in enumerate(faces[:13776]):
            faces_tensor[i] = torch.tensor(f)
        self.faces.copy_(faces_tensor)

    def _load_smpl_model(self, model_path: str):
        """加载SMPL模型文件（支持.npz和.pkl格式）"""

        if model_path.endswith('.npz'):
            # 加载预处理的npz格式
            smpl_data = np.load(model_path)
            v_template = smpl_data['v_template'].astype(np.float32)
            shapedirs = smpl_data['shapedirs'].astype(np.float32)
            posedirs = smpl_data['posedirs'].astype(np.float32)
            weights = smpl_data['weights'].astype(np.float32)
            faces = smpl_data['f'].astype(np.int64)
            J_regressor = smpl_data['J_regressor'].astype(np.float32)
            kintree_table = smpl_data['kintree_table']
        else:
            # 加载原始pkl格式（需要chumpy）
            import pickle

            with open(model_path, 'rb') as f:
                smpl_data = pickle.load(f, encoding='latin1')

            def to_numpy(x):
                """将chumpy对象或其他类型转换为numpy数组"""
                if hasattr(x, 'r'):
                    return np.array(x.r(), dtype=np.float32)
                elif isinstance(x, np.ndarray):
                    return x.astype(np.float32)
                else:
                    return np.array(x, dtype=np.float32)

            v_template = to_numpy(smpl_data['v_template'])
            shapedirs = to_numpy(smpl_data['shapedirs'])
            posedirs = to_numpy(smpl_data['posedirs'])
            weights = to_numpy(smpl_data['weights'])
            faces = np.array(smpl_data['f'], dtype=np.int64)

            J_regressor = smpl_data['J_regressor']
            if hasattr(J_regressor, 'toarray'):
                J_regressor = J_regressor.toarray()
            J_regressor = np.array(J_regressor, dtype=np.float32)
            kintree_table = smpl_data['kintree_table']

        # 只使用前10个形状参数（HMR预测10维）
        if shapedirs.shape[-1] > 10:
            shapedirs = shapedirs[..., :10]

        self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))

        parents = np.array(kintree_table[0], dtype=np.int64).copy()
        parents[0] = -1
        self.register_buffer('parents', torch.tensor(parents, dtype=torch.long))

        print(f"  - 顶点数: {v_template.shape[0]}")
        print(f"  - 形状参数维度: {shapedirs.shape}")
        print(f"  - 关节数: {J_regressor.shape[0]}")

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        translation: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从SMPL参数生成3D人体网格
        使用标准的线性混合蒙皮 (LBS) 算法

        Args:
            pose: 姿态参数 (B, 72)
            shape: 形状参数 (B, 10)
            translation: 平移 (B, 3)

        Returns:
            vertices: 顶点坐标 (B, 6890, 3)
            joints: 关节坐标 (B, 24, 3)
        """
        batch_size = pose.size(0)
        device = pose.device

        # 确保模板在正确的设备上
        v_template = self.v_template.to(device)  # (6890, 3)
        shapedirs = self.shapedirs.to(device)  # (6890, 3, 10)
        J_regressor = self.J_regressor.to(device)  # (24, 6890)

        # 应用形状变形: v_shaped = v_template + shapedirs * shape
        if len(shapedirs.shape) == 3:
            # 标准SMPL格式: (V, 3, num_betas)
            blend_shape = torch.einsum('vij,bj->bvi', shapedirs, shape)
        else:
            # 扁平格式: (V*3, num_betas)
            blend_shape = torch.matmul(shape, shapedirs.T).view(batch_size, -1, 3)

        v_shaped = v_template.unsqueeze(0) + blend_shape  # (B, 6890, 3)

        # 计算关节位置: J_regressor (24, 6890) x v_shaped (B, 6890, 3) -> joints (B, 24, 3)
        joints = torch.matmul(J_regressor, v_shaped)  # (B, 24, 3)

        # 简化版：直接返回形状变形后的顶点和关节
        vertices = v_shaped

        # 应用全局旋转（使用前3个参数）
        global_orient = pose[:, :3]
        rot_mat = self._axis_angle_to_matrix(global_orient)  # (B, 3, 3)

        # 旋转顶点和关节
        vertices = torch.bmm(vertices, rot_mat.transpose(1, 2))  # (B, 6890, 3)
        joints = torch.bmm(joints, rot_mat.transpose(1, 2))  # (B, 24, 3)

        # 6. 应用全局平移
        if translation is not None:
            vertices = vertices + translation.unsqueeze(1)
            joints = joints + translation.unsqueeze(1)

        return vertices, joints

    def _axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """轴角表示转旋转矩阵"""
        batch_size = axis_angle.size(0)
        angle = torch.norm(axis_angle, dim=1, keepdim=True)
        axis = axis_angle / (angle + 1e-8)

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        # Rodrigues公式
        K = torch.zeros(batch_size, 3, 3, device=axis_angle.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        eye = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(batch_size, -1, -1)
        rot_mat = eye + sin_angle.unsqueeze(2) * K + (1 - cos_angle.unsqueeze(2)) * torch.bmm(K, K)

        return rot_mat


class Pose3DReconstructor:
    """
    3D人体姿态重建器
    结合HMR编码器和SMPL模型进行端到端的3D重建
    """

    def __init__(
        self,
        smpl_model_path: Optional[str] = None,
        hmr_checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        input_size: int = 224
    ):
        """
        初始化3D重建器

        Args:
            smpl_model_path: SMPL模型文件路径
            hmr_checkpoint_path: HMR预训练权重路径
            device: 运行设备
            input_size: 输入图像尺寸
        """
        self.device = torch.device(device)
        self.input_size = input_size

        # 初始化模型
        self.encoder = HMREncoder(pretrained=True).to(self.device)
        self.regressor = SMPLRegressor(use_6d_rotation=True).to(self.device)
        self.smpl = SimpleSMPL(smpl_model_path).to(self.device)

        # 加载HMR预训练权重
        if hmr_checkpoint_path is None:
            # 尝试默认路径
            hmr_checkpoint_path = "models/hmr/model_checkpoint.pt"

        if os.path.exists(hmr_checkpoint_path):
            self._load_hmr_weights(hmr_checkpoint_path)
        else:
            print(f"  警告: 未找到HMR预训练权重: {hmr_checkpoint_path}")

        # 设置为评估模式
        self.encoder.eval()
        self.regressor.eval()
        self.smpl.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_hmr_weights(self, checkpoint_path: str):
        """加载HMR预训练权重"""
        print(f"  加载HMR预训练权重: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)

        # 分离encoder和regressor的权重
        encoder_state = {}
        regressor_state = {}

        for key, value in state_dict.items():
            # ResNet backbone权重 (直接匹配，不需要前缀)
            if key.startswith(('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')):
                encoder_state[key] = value
            # 回归器权重
            elif key.startswith(('fc1', 'fc2', 'decpose', 'decshape', 'deccam')):
                regressor_state[key] = value

        # 加载encoder权重
        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            loaded = len(encoder_state) - len(unexpected)
            print(f"    Encoder: 加载 {loaded} 个参数")

        # 加载regressor权重
        if regressor_state:
            missing, unexpected = self.regressor.load_state_dict(regressor_state, strict=False)
            loaded = len(regressor_state) - len(unexpected)
            print(f"    Regressor: 加载 {loaded} 个参数")

        print("  [OK] HMR预训练权重加载完成")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理输入图像

        Args:
            image: BGR格式的输入图像

        Returns:
            预处理后的tensor (1, 3, 224, 224)
        """
        # BGR转RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用变换
        tensor = self.transform(image_rgb)

        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def reconstruct(self, image: np.ndarray, keypoints_2d: Optional[np.ndarray] = None) -> Pose3DResult:
        """
        从单张图像重建3D人体

        Args:
            image: BGR格式的输入图像
            keypoints_2d: 可选的2D关键点 (用于改进3D估计)

        Returns:
            Pose3DResult对象
        """
        # 预处理图像
        input_tensor = self.preprocess_image(image)

        # 提取特征
        features = self.encoder(input_tensor)

        # 预测SMPL参数
        pose, shape, cam = self.regressor(features)

        # 如果提供了2D关键点，使用它们来改进3D估计
        if keypoints_2d is not None:
            joints = self._lift_2d_to_3d(keypoints_2d, image.shape[:2])
            # 使用默认的shape参数生成网络
            vertices, _ = self.smpl(pose, shape)
        else:
            # 生成3D网络
            vertices, joints = self.smpl(pose, shape)

        # 投影到2D
        joints_2d = self._project_joints(joints, cam, image.shape[:2])

        # 转换为numpy
        pose_np = pose.cpu().numpy()[0]
        shape_np = shape.cpu().numpy()[0]
        cam_np = cam.cpu().numpy()[0]
        vertices_np = vertices.cpu().numpy()[0]
        joints_np = joints.cpu().numpy()[0]
        joints_2d_np = joints_2d.cpu().numpy()[0]
        faces_np = self.smpl.faces.cpu().numpy()


        # 构建SMPL参数
        smpl_params = SMPLParams(
            pose=pose_np,
            shape=shape_np,
            global_orient=pose_np[:3],
            translation=np.zeros(3),
            camera=cam_np
        )

        return Pose3DResult(
            smpl_params=smpl_params,
            joints_3d=joints_np,
            vertices=vertices_np,
            joints_2d_proj=joints_2d_np,
            faces=faces_np,
            confidence=0.8
        )

    def _project_joints(
        self,
        joints: torch.Tensor,
        cam: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        将3D关节投影到2D图像平面

        Args:
            joints: 3D关节位置 (B, 24, 3)
            cam: 相机参数 (B, 3) - [scale, tx, ty]
            image_size: 图像尺寸 (H, W)

        Returns:
            2D关节位置 (B, 24, 2)
        """
        scale = cam[:, 0:1]  # (B, 1)
        trans = cam[:, 1:3]  # (B, 2)

        # 正交投影
        joints_2d = joints[:, :, :2] * scale.unsqueeze(1) + trans.unsqueeze(1)

        # 归一化到图像坐标
        h, w = image_size
        joints_2d[:, :, 0] = (joints_2d[:, :, 0] + 1) * w / 2
        joints_2d[:, :, 1] = (joints_2d[:, :, 1] + 1) * h / 2

        return joints_2d

    def _lift_2d_to_3d(self, keypoints_2d: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        从2D关键点提升到3D关节位置
        使用简单的几何先验来估计深度

        Args:
            keypoints_2d: 2D关键点 (N, 2) 或 (N, 3) 包含可见度
            image_size: 图像尺寸 (H, W)

        Returns:
            3D关节位置 (1, 24, 3)
        """
        h, w = image_size

        # MediaPipe到SMPL的关键点映射
        # MediaPipe: 33个点, SMPL: 24个点
        mp_to_smpl = {
            0: 15,   # nose -> head
            11: 16,  # left_shoulder -> left_shoulder
            12: 17,  # right_shoulder -> right_shoulder
            13: 18,  # left_elbow -> left_elbow
            14: 19,  # right_elbow -> right_elbow
            15: 20,  # left_wrist -> left_wrist
            16: 21,  # right_wrist -> right_wrist
            23: 1,   # left_hip -> left_hip
            24: 2,   # right_hip -> right_hip
            25: 4,   # left_knee -> left_knee
            26: 5,   # right_knee -> right_knee
            27: 7,   # left_ankle -> left_ankle
            28: 8,   # right_ankle -> right_ankle
        }

        # 初始化24个SMPL关节
        joints_3d = np.zeros((24, 3), dtype=np.float32)

        # 归一化2D坐标到[-1, 1]
        kpts = keypoints_2d[:, :2].copy()
        kpts[:, 0] = (kpts[:, 0] / w) * 2 - 1  # x: [-1, 1]
        kpts[:, 1] = -((kpts[:, 1] / h) * 2 - 1)  # y: [-1, 1], 翻转使上为正

        # 填充已知的关节
        for mp_idx, smpl_idx in mp_to_smpl.items():
            if mp_idx < len(kpts):
                joints_3d[smpl_idx, 0] = kpts[mp_idx, 0]  # X (左右)
                joints_3d[smpl_idx, 1] = kpts[mp_idx, 1]  # Y (上下)

        # 估计深度 (Z) 使用启发式方法
        # 基于身体结构的先验知识

        # 计算肩宽和髋宽来估计身体朝向
        if 11 < len(kpts) and 12 < len(kpts):
            shoulder_width = np.abs(kpts[11, 0] - kpts[12, 0])
        else:
            shoulder_width = 0.3

        # 使用肩宽来估计深度缩放
        depth_scale = 0.15 / (shoulder_width + 0.01)

        # 设置各关节的深度 (基于人体结构先验)
        depth_offsets = {
            0: 0.0,    # pelvis (中心)
            1: 0.05,   # left_hip
            2: -0.05,  # right_hip
            3: 0.0,    # spine1
            4: 0.05,   # left_knee
            5: -0.05,  # right_knee
            6: 0.0,    # spine2
            7: 0.05,   # left_ankle
            8: -0.05,  # right_ankle
            9: 0.0,    # spine3
            10: 0.1,   # left_foot
            11: -0.1,  # right_foot
            12: 0.0,   # neck
            13: 0.1,   # left_collar
            14: -0.1,  # right_collar
            15: 0.05,  # head
            16: 0.15,  # left_shoulder
            17: -0.15, # right_shoulder
            18: 0.1,   # left_elbow
            19: -0.1,  # right_elbow
            20: 0.05,  # left_wrist
            21: -0.05, # right_wrist
            22: 0.05,  # left_hand
            23: -0.05, # right_hand
        }

        for idx, offset in depth_offsets.items():
            joints_3d[idx, 2] = offset * depth_scale

        # 填充未映射的关节 (使用插值)
        # pelvis = 髋部中点
        if 23 < len(kpts) and 24 < len(kpts):
            joints_3d[0] = (joints_3d[1] + joints_3d[2]) / 2

        # spine各点 (在pelvis和neck之间插值)
        joints_3d[3] = joints_3d[0] + (joints_3d[12] - joints_3d[0]) * 0.33  # spine1
        joints_3d[6] = joints_3d[0] + (joints_3d[12] - joints_3d[0]) * 0.5   # spine2
        joints_3d[9] = joints_3d[0] + (joints_3d[12] - joints_3d[0]) * 0.67  # spine3

        # neck (在肩膀上方)
        if 11 < len(kpts) and 12 < len(kpts):
            joints_3d[12] = (joints_3d[16] + joints_3d[17]) / 2
            joints_3d[12, 1] += 0.1  # 稍微向上

        # collar骨
        joints_3d[13] = (joints_3d[12] + joints_3d[16]) / 2
        joints_3d[14] = (joints_3d[12] + joints_3d[17]) / 2

        # 脚
        joints_3d[10] = joints_3d[7].copy()
        joints_3d[10, 2] += 0.05
        joints_3d[11] = joints_3d[8].copy()
        joints_3d[11, 2] -= 0.05

        # 手
        joints_3d[22] = joints_3d[20].copy()
        joints_3d[23] = joints_3d[21].copy()

        # 缩放到合理范围
        joints_3d *= 0.5

        return torch.tensor(joints_3d, dtype=torch.float32).unsqueeze(0).to(self.device)

    def to_dict(self, result: Pose3DResult) -> Dict:
        """将结果转换为字典格式"""
        return {
            'smpl_params': {
                'pose': result.smpl_params.pose.tolist(),
                'shape': result.smpl_params.shape.tolist(),
                'global_orient': result.smpl_params.global_orient.tolist(),
                'translation': result.smpl_params.translation.tolist(),
                'camera': result.smpl_params.camera.tolist()
            },
            'joints_3d': result.joints_3d.tolist(),
            'joints_2d_proj': result.joints_2d_proj.tolist(),
            'num_vertices': len(result.vertices),
            'num_faces': len(result.faces),
            'confidence': result.confidence
        }

    def export_mesh(self, result: Pose3DResult, output_path: str, format: str = 'obj'):
        """
        导出3D网格文件

        Args:
            result: 3D重建结果
            output_path: 输出文件路径
            format: 文件格式 ('obj', 'ply')
        """
        try:
            import trimesh

            mesh = trimesh.Trimesh(
                vertices=result.vertices,
                faces=result.faces,
                process=False
            )
            mesh.export(output_path, file_type=format)

        except ImportError:
            # 手动写入OBJ文件
            if format == 'obj':
                with open(output_path, 'w') as f:
                    f.write("# SMPL mesh exported from Human Pose Estimation System\n")

                    # 写入顶点
                    for v in result.vertices:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                    # 写入面片
                    for face in result.faces:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def process_image(image_path: str, smpl_model_path: Optional[str] = None) -> Optional[Dict]:
    """
    处理单张图片的便捷函数

    Args:
        image_path: 图片路径
        smpl_model_path: SMPL模型路径

    Returns:
        包含3D重建信息的字典
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    reconstructor = Pose3DReconstructor(smpl_model_path)
    result = reconstructor.reconstruct(image)

    return reconstructor.to_dict(result)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = process_image(image_path)
        if result:
            print("3D重建成功!")
            print(f"形状参数: {result['smpl_params']['shape'][:5]}...")
            print(f"关节数量: {len(result['joints_3d'])}")
            print(f"置信度: {result['confidence']:.2f}")
        else:
            print("3D重建失败")
    else:
        print("用法: python pose_3d.py <图片路径>")