# 人体姿态与形状估计系统

基于深度学习的单图像3D人体重建系统，支持2D关键点检测、3D姿态估计和SMPL模型生成。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)

## 📋 功能特性

- **2D姿态估计**: 使用MediaPipe检测33个人体关键点
- **3D人体重建**: 基于HMR方法预测SMPL模型参数
- **SMPL模型**: 参数化人体模型，支持姿态和形状控制
- **可视化渲染**: 多视角3D骨架和网格可视化
- **Web界面**: 现代化的拖拽上传界面
- **模型导出**: 支持导出OBJ格式3D模型文件

## 🏗️ 系统架构

```
├── app.py                  # Flask Web应用
├── main.py                 # 主程序入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── download_models.py      # 模型下载脚本
│
├── modules/                # 核心模块
│   ├── __init__.py
│   ├── pose_2d.py         # 2D姿态估计
│   ├── pose_3d.py         # 3D姿态重建
│   └── visualization.py   # 可视化渲染
│
├── models/                 # 模型文件
│   ├── smpl/              # SMPL模型
│   └── hmr/               # HMR权重
│
├── templates/              # HTML模板
│   └── index.html
│
├── uploads/                # 上传文件
├── results/                # 结果文件
└── samples/                # 示例图片
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd biye

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备模型文件

```bash
python download_models.py
```

按照提示下载SMPL模型文件（需要从官方网站注册下载）。

### 4. 启动服务

```bash
# 启动Web服务
python main.py --web

# 或直接运行
python app.py
```

访问 http://localhost:5000 使用Web界面。

## 💻 使用方式

### Web界面

1. 打开浏览器访问 http://localhost:5000
2. 拖拽或点击上传人像照片
3. 点击"开始分析"
4. 查看2D关键点、3D骨架、3D网格等结果
5. 下载OBJ模型或JSON数据

### 命令行

```bash
# 处理单张图片
python main.py --image path/to/image.jpg --output results/

# 批量处理
python main.py --batch path/to/images/ --output results/
```

### Python API

```python
from modules.pose_2d import Pose2DEstimator
from modules.pose_3d import Pose3DReconstructor
import cv2

# 读取图像
image = cv2.imread("path/to/image.jpg")

# 2D姿态估计
estimator_2d = Pose2DEstimator()
result_2d = estimator_2d.estimate(image)
print(f"检测到 {len(result_2d.keypoints)} 个关键点")

# 3D重建
reconstructor = Pose3DReconstructor()
result_3d = reconstructor.reconstruct(image)
print(f"形状参数: {result_3d.smpl_params.shape}")

# 导出3D模型
reconstructor.export_mesh(result_3d, "output.obj")
```

## 📊 技术原理

### 2D姿态估计

使用MediaPipe Pose进行人体关键点检测：
- 33个关键点覆盖全身
- 支持单人检测
- 实时性能

### 3D人体重建

采用HMR (Human Mesh Recovery) 方法：
1. **特征提取**: ResNet50 backbone提取图像特征
2. **参数回归**: 迭代回归预测SMPL参数
3. **网格生成**: SMPL模型生成3D人体网格

### SMPL模型

参数化人体模型：
- **姿态参数** (72维): 24个关节的轴角表示
- **形状参数** (10维): 控制身高、胖瘦等体型特征
- **输出**: 6890个顶点的3D网格

## ⚙️ 配置说明

编辑 `config.yaml` 自定义配置：

```yaml
# 服务器配置
server:
  host: "0.0.0.0"
  port: 5000

# 模型配置
models:
  smpl_model_path: "models/smpl"
  device: "cuda"  # 或 "cpu"

# 2D姿态估计
pose_2d:
  model_complexity: 2  # 0, 1, 2
  min_detection_confidence: 0.5
```

## 📝 API接口

### POST /api/upload
上传图片文件

### POST /api/estimate
执行姿态估计
```json
{
  "filename": "uploaded_file.jpg"
}
```

### GET /api/download/:filename
下载结果文件

### GET /api/health
健康检查

## 🔧 常见问题

### CUDA内存不足
将 `config.yaml` 中的 `device` 改为 `cpu`

### 未检测到人体
- 确保图片中人体完整可见
- 尝试使用更高分辨率的图片
- 避免严重遮挡

### 模型加载失败
- 检查模型文件是否完整
- 确认文件路径正确

## 📚 参考文献

- [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/)
- [End-to-end Recovery of Human Shape and Pose](https://github.com/akanazawa/hmr)
- [Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop](https://github.com/nkolot/SPIN)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)

## 📄 许可证

本项目仅供学术研究使用。SMPL模型受其官方许可协议约束。

