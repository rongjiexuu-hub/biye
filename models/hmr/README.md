# HMR预训练权重

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
