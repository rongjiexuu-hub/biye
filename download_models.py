"""
模型下载脚本
下载预训练的SMPL模型和HMR权重
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import hashlib


# 模型目录
MODELS_DIR = Path("models")
SMPL_DIR = MODELS_DIR / "smpl"
HMR_DIR = MODELS_DIR / "hmr"

# 模型信息
MODEL_INFO = {
    "smpl_neutral": {
        "description": "SMPL中性模型 (来自官方SMPL网站)",
        "note": "需要从 https://smpl.is.tue.mpg.de 注册下载",
        "local_path": SMPL_DIR / "SMPL_NEUTRAL.pkl"
    },
    "smpl_male": {
        "description": "SMPL男性模型",
        "note": "需要从官方网站下载",
        "local_path": SMPL_DIR / "SMPL_MALE.pkl"
    },
    "smpl_female": {
        "description": "SMPL女性模型", 
        "note": "需要从官方网站下载",
        "local_path": SMPL_DIR / "SMPL_FEMALE.pkl"
    }
}


def create_directories():
    """创建必要的目录"""
    MODELS_DIR.mkdir(exist_ok=True)
    SMPL_DIR.mkdir(exist_ok=True)
    HMR_DIR.mkdir(exist_ok=True)
    print(f"[OK] 创建目录: {MODELS_DIR}")


def download_file(url: str, save_path: Path, desc: str = None):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def check_models():
    """检查模型文件是否存在"""
    print("\n检查模型文件状态:")
    print("-" * 50)
    
    all_exist = True
    for name, info in MODEL_INFO.items():
        path = info["local_path"]
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  [OK] {name}: 已存在 ({size:.1f} MB)")
        else:
            print(f"  [X] {name}: 未找到")
            print(f"    说明: {info['description']}")
            print(f"    注意: {info['note']}")
            all_exist = False
    
    print("-" * 50)
    return all_exist


def create_smpl_placeholder():
    """创建SMPL模型占位说明文件"""
    readme_path = SMPL_DIR / "README.md"
    
    content = """# SMPL模型文件

本目录用于存放SMPL人体模型文件。

## 获取模型

SMPL模型需要从官方网站注册下载：

1. 访问 https://smpl.is.tue.mpg.de/
2. 注册账号并同意许可协议
3. 下载 SMPL for Python
4. 解压后将以下文件复制到本目录：
   - `SMPL_NEUTRAL.pkl` (或 `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`)
   - `SMPL_MALE.pkl` (可选)
   - `SMPL_FEMALE.pkl` (可选)

## 文件重命名

如果下载的文件名不同，请按以下方式重命名：
- `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` → `SMPL_NEUTRAL.pkl`
- `basicModel_m_lbs_10_207_0_v1.0.0.pkl` → `SMPL_MALE.pkl`
- `basicModel_f_lbs_10_207_0_v1.0.0.pkl` → `SMPL_FEMALE.pkl`

## 注意事项

- SMPL模型受许可协议保护，仅限研究用途
- 请勿将模型文件上传到公开仓库
- 本项目在没有SMPL模型文件时会使用简化的人体模板

## 替代方案

如果无法获取官方SMPL模型，系统会使用内置的简化人体模板进行演示。
虽然效果不如完整SMPL模型，但可以展示基本的3D重建流程。
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] 创建说明文件: {readme_path}")


def create_hmr_placeholder():
    """创建HMR模型占位说明文件"""
    readme_path = HMR_DIR / "README.md"
    
    content = """# HMR预训练权重

本目录用于存放HMR (Human Mesh Recovery) 预训练模型权重。

## 模型说明

本项目使用基于ResNet50的HMR架构，可以使用以下预训练权重：

1. **ImageNet预训练** (默认)
   - 系统默认使用PyTorch提供的ImageNet预训练ResNet50
   - 无需额外下载

2. **HMR官方权重** (可选)
   - 从 https://github.com/akanazawa/hmr 获取
   - 提供更好的人体姿态估计效果

3. **SPIN权重** (可选)
   - 从 https://github.com/nkolot/SPIN 获取
   - 使用迭代回归方法，效果更好

## 使用说明

将下载的权重文件放置在本目录，并在 `config.yaml` 中配置路径：

```yaml
models:
  hmr_model_path: "models/hmr/hmr_weights.pth"
```

## 注意事项

- 预训练权重文件较大 (约 200-500 MB)
- 确保权重文件与模型架构匹配
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] 创建说明文件: {readme_path}")


def download_sample_image():
    """下载示例图片用于测试"""
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    # 创建示例图片说明
    readme_path = samples_dir / "README.md"
    content = """# 示例图片

将人像照片放入此目录进行测试。

## 要求

- 图片格式: JPG, PNG, WebP
- 最大尺寸: 10MB
- 人体应该完整可见
- 背景尽量简单

## 示例

你可以使用任何包含完整人体的照片进行测试。
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] 创建示例目录: {samples_dir}")


def main():
    """主函数"""
    print("=" * 50)
    print("人体姿态估计系统 - 模型下载工具")
    print("=" * 50)
    
    # 创建目录
    create_directories()
    
    # 创建说明文件
    create_smpl_placeholder()
    create_hmr_placeholder()
    
    # 下载示例图片目录
    download_sample_image()
    
    # 检查模型状态
    all_exist = check_models()
    
    if not all_exist:
        print("\n⚠ 部分模型文件缺失")
        print("请按照上述说明手动下载模型文件")
        print("\n提示: 即使没有SMPL模型，系统也可以使用简化模板运行")
    else:
        print("\n[OK] 所有模型文件已就绪！")
    
    print("\n下一步:")
    print("  1. 安装依赖: pip install -r requirements.txt")
    print("  2. 启动服务: python app.py")
    print("  3. 访问: http://localhost:5000")


if __name__ == "__main__":
    main()

