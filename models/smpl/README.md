# SMPL模型文件

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
