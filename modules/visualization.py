"""
可视化模块
用于渲染2D/3D姿态估计结果
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import io
import base64
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """可视化配置"""
    render_width: int = 800
    render_height: int = 600
    keypoint_color: Tuple[int, int, int] = (0, 255, 0)
    skeleton_color: Tuple[int, int, int] = (255, 0, 0)
    mesh_color: str = 'lightblue'
    mesh_alpha: float = 0.6
    keypoint_radius: int = 5
    skeleton_thickness: int = 2
    show_axes: bool = True
    background_color: str = '#1a1a2e'


class Visualizer:
    """
    姿态可视化器
    支持2D关键点、3D骨架和3D网格的可视化
    """
    
    # SMPL关节名称
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]
    
    # 骨架连接 (SMPL 24关节)
    SKELETON_CONNECTIONS = [
        # 躯干
        (0, 1), (0, 2), (0, 3),  # 骨盆到髋部和脊柱
        (3, 6), (6, 9), (9, 12),  # 脊柱
        (12, 15),  # 颈部到头部
        # 左腿
        (1, 4), (4, 7), (7, 10),
        # 右腿
        (2, 5), (5, 8), (8, 11),
        # 肩膀
        (9, 13), (9, 14),
        (13, 16), (14, 17),
        # 左臂
        (16, 18), (18, 20), (20, 22),
        # 右臂
        (17, 19), (19, 21), (21, 23),
    ]
    
    # 身体部位颜色映射
    BODY_PART_COLORS = {
        'torso': '#FF6B6B',
        'left_arm': '#4ECDC4',
        'right_arm': '#45B7D1',
        'left_leg': '#96CEB4',
        'right_leg': '#FFEAA7',
        'head': '#DDA0DD'
    }
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config or VisualizationConfig()
    
    def draw_keypoints_2d(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        draw_indices: bool = False
    ) -> np.ndarray:
        """
        在图像上绘制2D关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标 (N, 2)
            confidences: 置信度 (N,)
            draw_indices: 是否绘制关键点索引
            
        Returns:
            绘制后的图像
        """
        output = image.copy()
        
        if confidences is None:
            confidences = np.ones(len(keypoints))
        
        for i, (point, conf) in enumerate(zip(keypoints, confidences)):
            if conf > 0.3:
                x, y = int(point[0]), int(point[1])
                
                # 根据置信度调整颜色
                alpha = min(1.0, conf)
                color = tuple(int(c * alpha) for c in self.config.keypoint_color)
                
                cv2.circle(output, (x, y), self.config.keypoint_radius, color, -1)
                cv2.circle(output, (x, y), self.config.keypoint_radius, (255, 255, 255), 1)
                
                if draw_indices:
                    cv2.putText(output, str(i), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return output
    
    def draw_skeleton_2d(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        connections: List[Tuple[int, int]],
        confidences: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        在图像上绘制2D骨架
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标 (N, 2)
            connections: 骨架连接
            confidences: 置信度 (N,)
            
        Returns:
            绘制后的图像
        """
        output = image.copy()
        
        if confidences is None:
            confidences = np.ones(len(keypoints))
        
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                conf = min(confidences[start_idx], confidences[end_idx])
                
                if conf > 0.3:
                    start_point = tuple(keypoints[start_idx].astype(int))
                    end_point = tuple(keypoints[end_idx].astype(int))
                    
                    cv2.line(output, start_point, end_point,
                            self.config.skeleton_color, self.config.skeleton_thickness)
        
        return output
    
    def visualize_pose_2d(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        connections: List[Tuple[int, int]],
        confidences: Optional[np.ndarray] = None,
        title: str = "2D Pose Estimation"
    ) -> str:
        """
        生成2D姿态可视化图像（返回base64编码）
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标
            connections: 骨架连接
            confidences: 置信度
            title: 图像标题
            
        Returns:
            Base64编码的图像
        """
        # 绘制骨架和关键点
        vis_image = self.draw_skeleton_2d(image, keypoints, connections, confidences)
        vis_image = self.draw_keypoints_2d(vis_image, keypoints, confidences)
        
        # 转换为matplotlib图像
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14, color='white')
        ax.axis('off')
        fig.patch.set_facecolor(self.config.background_color)
        
        # 保存为base64
        return self._fig_to_base64(fig)
    
    def visualize_skeleton_3d(
        self,
        joints_3d: np.ndarray,
        connections: Optional[List[Tuple[int, int]]] = None,
        title: str = "3D Skeleton",
        elev: float = 10,
        azim: float = -90
    ) -> str:
        """
        生成3D骨架可视化
        
        Args:
            joints_3d: 3D关节坐标 (N, 3) - SMPL格式: X右, Y上, Z前
            connections: 骨架连接
            title: 图像标题
            elev: 仰角
            azim: 方位角
            
        Returns:
            Base64编码的图像
        """
        if connections is None:
            connections = self.SKELETON_CONNECTIONS
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置背景颜色
        ax.set_facecolor(self.config.background_color)
        fig.patch.set_facecolor(self.config.background_color)
        
        # 坐标转换: SMPL(X右,Y上,Z前) -> matplotlib(X右,Y前,Z上)
        # 交换Y和Z，让人站立显示
        plot_x = joints_3d[:, 0]  # X -> X (左右)
        plot_y = joints_3d[:, 2]  # Z -> Y (前后)  
        plot_z = joints_3d[:, 1]  # Y -> Z (上下，身高)
        
        # 绘制关节点
        ax.scatter(
            plot_x, plot_y, plot_z,
            c='#00ff88',
            s=80,
            alpha=0.9,
            edgecolors='white',
            linewidths=1.5
        )
        
        # 绘制骨架连接
        for start_idx, end_idx in connections:
            if start_idx < len(joints_3d) and end_idx < len(joints_3d):
                xs = [plot_x[start_idx], plot_x[end_idx]]
                ys = [plot_y[start_idx], plot_y[end_idx]]
                zs = [plot_z[start_idx], plot_z[end_idx]]
                ax.plot(xs, ys, zs, c='#ff6b6b', linewidth=3, alpha=0.9)
        
        # 设置视角 - 从正面看
        ax.view_init(elev=elev, azim=azim)
        
        # 设置轴标签
        ax.set_xlabel('X (Left-Right)', color='white', fontsize=10)
        ax.set_ylabel('Z (Front-Back)', color='white', fontsize=10)
        ax.set_zlabel('Y (Height)', color='white', fontsize=10)
        ax.set_title(title, fontsize=14, color='white', pad=10)
        
        # 设置轴颜色
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        
        # 设置相等的轴比例
        all_coords = np.column_stack([plot_x, plot_y, plot_z])
        max_range = np.max(np.abs(all_coords)) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def visualize_mesh_3d(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        title: str = "3D Body Mesh",
        elev: float = 10,
        azim: float = -90
    ) -> str:
        """
        生成3D网格可视化
        
        Args:
            vertices: 顶点坐标 (V, 3) - SMPL格式: X右, Y上, Z前
            faces: 面片索引 (F, 3)
            title: 图像标题
            elev: 仰角
            azim: 方位角
            
        Returns:
            Base64编码的图像
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置背景颜色
        ax.set_facecolor(self.config.background_color)
        fig.patch.set_facecolor(self.config.background_color)
        
        # 坐标转换: SMPL(X右,Y上,Z前) -> matplotlib(X右,Y前,Z上)
        # 交换Y和Z，让人站立显示
        transformed_vertices = np.column_stack([
            vertices[:, 0],  # X -> X
            vertices[:, 2],  # Z -> Y
            vertices[:, 1]   # Y -> Z (身高方向)
        ])
        
        # 创建面片集合
        mesh_faces = []
        for face in faces[:5000]:  # 限制面片数量以提高渲染速度
            triangle = transformed_vertices[face]
            mesh_faces.append(triangle)
        
        # 创建3D多边形集合
        mesh = Poly3DCollection(
            mesh_faces,
            alpha=self.config.mesh_alpha,
            facecolor=self.config.mesh_color,
            edgecolor='#404060',
            linewidth=0.1
        )
        ax.add_collection3d(mesh)
        
        # 设置视角 - 从正面看
        ax.view_init(elev=elev, azim=azim)
        
        # 设置轴范围
        max_range = np.max(np.abs(transformed_vertices)) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # 设置标签
        ax.set_xlabel('X', color='white', fontsize=10)
        ax.set_ylabel('Z', color='white', fontsize=10)
        ax.set_zlabel('Y (Height)', color='white', fontsize=10)
        ax.set_title(title, fontsize=14, color='white', pad=10)
        
        ax.tick_params(colors='white')
        
        return self._fig_to_base64(fig)
    
    def create_comparison_view(
        self,
        original_image: np.ndarray,
        pose_2d_image: str,
        skeleton_3d_image: str,
        mesh_3d_image: str
    ) -> str:
        """
        创建对比视图
        
        Args:
            original_image: 原始图像
            pose_2d_image: 2D姿态图像（base64）
            skeleton_3d_image: 3D骨架图像（base64）
            mesh_3d_image: 3D网格图像（base64）
            
        Returns:
            Base64编码的合成图像
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor(self.config.background_color)
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', color='white', fontsize=12)
        axes[0, 0].axis('off')
        
        # 2D姿态
        pose_2d_arr = self._base64_to_array(pose_2d_image)
        axes[0, 1].imshow(pose_2d_arr)
        axes[0, 1].set_title('2D Pose Estimation', color='white', fontsize=12)
        axes[0, 1].axis('off')
        
        # 3D骨架
        skeleton_3d_arr = self._base64_to_array(skeleton_3d_image)
        axes[1, 0].imshow(skeleton_3d_arr)
        axes[1, 0].set_title('3D Skeleton', color='white', fontsize=12)
        axes[1, 0].axis('off')
        
        # 3D网格
        mesh_3d_arr = self._base64_to_array(mesh_3d_image)
        axes[1, 1].imshow(mesh_3d_arr)
        axes[1, 1].set_title('3D Body Mesh', color='white', fontsize=12)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def visualize_shape_parameters(
        self,
        shape_params: np.ndarray,
        param_names: Optional[List[str]] = None
    ) -> str:
        """
        可视化SMPL形状参数
        
        Args:
            shape_params: 形状参数 (10,)
            param_names: 参数名称
            
        Returns:
            Base64编码的图像
        """
        if param_names is None:
            param_names = [f'β{i}' for i in range(len(shape_params))]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.config.background_color)
        ax.set_facecolor(self.config.background_color)
        
        colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in shape_params]
        bars = ax.bar(param_names, shape_params, color=colors, edgecolor='white', linewidth=1)
        
        ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Shape Parameters', color='white', fontsize=12)
        ax.set_ylabel('Value', color='white', fontsize=12)
        ax.set_title('SMPL Shape Parameters (β)', color='white', fontsize=14)
        ax.tick_params(colors='white')
        
        # 添加数值标签
        for bar, val in zip(bars, shape_params):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -10),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       color='white', fontsize=9)
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_rotating_mesh_frames(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_frames: int = 36
    ) -> List[str]:
        """
        创建旋转网格动画帧
        
        Args:
            vertices: 顶点坐标
            faces: 面片索引
            n_frames: 帧数
            
        Returns:
            Base64编码的图像列表
        """
        frames = []
        for i in range(n_frames):
            azim = i * (360 / n_frames)
            frame = self.visualize_mesh_3d(vertices, faces, elev=10, azim=azim)
            frames.append(frame)
        return frames

    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图像转换为base64编码"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _base64_to_array(self, base64_str: str) -> np.ndarray:
        """将base64编码转换为numpy数组"""
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR to RGB
    
    def save_image(self, base64_str: str, output_path: str):
        """保存base64图像到文件"""
        img_bytes = base64.b64decode(base64_str)
        with open(output_path, 'wb') as f:
            f.write(img_bytes)


def create_pose_overlay(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    joints_3d: np.ndarray,
    connections: List[Tuple[int, int]]
) -> Dict[str, str]:
    """
    创建姿态叠加视图
    
    Args:
        image: 原始图像
        keypoints_2d: 2D关键点
        joints_3d: 3D关节
        connections: 骨架连接
        
    Returns:
        包含各种可视化结果的字典
    """
    vis = Visualizer()
    
    results = {
        'pose_2d': vis.visualize_pose_2d(image, keypoints_2d, connections),
        'skeleton_3d': vis.visualize_skeleton_3d(joints_3d, connections),
    }
    
    return results


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    # 创建测试数据
    test_joints = np.random.randn(24, 3) * 0.5
    test_vertices = np.random.randn(100, 3) * 0.5
    test_faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    
    vis = Visualizer()
    
    # 测试3D骨架可视化
    skeleton_img = vis.visualize_skeleton_3d(test_joints, title="Test Skeleton")
    print(f"3D骨架图像生成成功, base64长度: {len(skeleton_img)}")
    
    # 测试形状参数可视化
    shape_params = np.random.randn(10) * 2
    shape_img = vis.visualize_shape_parameters(shape_params)
    print(f"形状参数图像生成成功, base64长度: {len(shape_img)}")

