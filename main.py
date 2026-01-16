"""
人体姿态与形状估计系统 - 主程序入口
支持命令行和Web服务两种运行模式
"""

import os
import sys
import argparse
import cv2
import json
from pathlib import Path
from datetime import datetime


def run_web_server():
    """启动Web服务器"""
    from app import app, init_models, load_config
    
    config = load_config()
    
    print("正在初始化模型...")
    init_models()
    
    host = config.get('server', {}).get('host', '0.0.0.0')
    port = config.get('server', {}).get('port', 5000)
    debug = config.get('server', {}).get('debug', True)
    
    print(f"\n服务器启动地址: http://localhost:{port}")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


def run_cli(image_path: str, output_dir: str = "output"):
    """命令行模式处理单张图片"""
    from modules.pose_2d import Pose2DEstimator
    from modules.pose_3d import Pose3DReconstructor
    from modules.visualization import Visualizer
    
    print("=" * 50)
    print("人体姿态与形状估计系统 - 命令行模式")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 读取图像
    print(f"\n读取图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("错误: 无法读取图像文件")
        sys.exit(1)
    
    print(f"图像尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 初始化模型
    print("\n初始化模型...")
    pose_2d_estimator = Pose2DEstimator()
    pose_3d_reconstructor = Pose3DReconstructor()
    visualizer = Visualizer()
    
    # 2D姿态估计
    print("\n执行2D姿态估计...")
    pose_2d_result = pose_2d_estimator.estimate(image)
    
    if pose_2d_result is None:
        print("错误: 未检测到人体")
        sys.exit(1)
    
    print(f"  检测到 {len(pose_2d_result.keypoints)} 个关键点")
    print(f"  置信度: {pose_2d_result.confidence:.2%}")
    
    # 3D姿态重建 - 使用2D关键点改进
    print("\n执行3D姿态重建...")
    pose_3d_result = pose_3d_reconstructor.reconstruct(
        image,
        keypoints_2d=pose_2d_result.landmarks_pixel
    )
    
    print(f"  SMPL形状参数: {pose_3d_result.smpl_params.shape[:5]}...")
    print(f"  3D关节数量: {len(pose_3d_result.joints_3d)}")
    print(f"  网格顶点数: {len(pose_3d_result.vertices)}")
    
    # 生成可视化
    print("\n生成可视化结果...")
    
    # 2D姿态图
    pose_2d_image = visualizer.draw_skeleton_2d(
        image, 
        pose_2d_result.landmarks_pixel,
        pose_2d_estimator.SKELETON_CONNECTIONS,
        [kp.visibility for kp in pose_2d_result.keypoints]
    )
    pose_2d_image = visualizer.draw_keypoints_2d(
        pose_2d_image,
        pose_2d_result.landmarks_pixel,
        [kp.visibility for kp in pose_2d_result.keypoints]
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(image_path).stem
    
    # 保存2D姿态图
    pose_2d_path = output_path / f"{base_name}_{timestamp}_pose2d.jpg"
    cv2.imwrite(str(pose_2d_path), pose_2d_image)
    print(f"  保存2D姿态图: {pose_2d_path}")
    
    # 保存3D骨架图
    skeleton_3d_base64 = visualizer.visualize_skeleton_3d(
        pose_3d_result.joints_3d,
        title="3D Skeleton"
    )
    skeleton_3d_path = output_path / f"{base_name}_{timestamp}_skeleton3d.png"
    visualizer.save_image(skeleton_3d_base64, str(skeleton_3d_path))
    print(f"  保存3D骨架图: {skeleton_3d_path}")
    
    # 保存3D网格图
    mesh_3d_base64 = visualizer.visualize_mesh_3d(
        pose_3d_result.vertices,
        pose_3d_result.faces,
        title="3D Body Mesh"
    )
    mesh_3d_path = output_path / f"{base_name}_{timestamp}_mesh3d.png"
    visualizer.save_image(mesh_3d_base64, str(mesh_3d_path))
    print(f"  保存3D网格图: {mesh_3d_path}")
    
    # 保存形状参数图
    shape_base64 = visualizer.visualize_shape_parameters(
        pose_3d_result.smpl_params.shape
    )
    shape_path = output_path / f"{base_name}_{timestamp}_shape.png"
    visualizer.save_image(shape_base64, str(shape_path))
    print(f"  保存形状参数图: {shape_path}")
    
    # 导出3D模型
    mesh_path = output_path / f"{base_name}_{timestamp}_mesh.obj"
    pose_3d_reconstructor.export_mesh(pose_3d_result, str(mesh_path))
    print(f"  导出3D模型: {mesh_path}")
    
    # 保存JSON数据
    json_data = {
        "image_path": str(image_path),
        "timestamp": timestamp,
        "pose_2d": pose_2d_estimator.to_dict(pose_2d_result),
        "pose_3d": pose_3d_reconstructor.to_dict(pose_3d_result)
    }
    json_path = output_path / f"{base_name}_{timestamp}_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  保存JSON数据: {json_path}")
    
    print("\n[OK] 处理完成！")
    print(f"  输出目录: {output_path.absolute()}")
    
    # 清理资源
    pose_2d_estimator.close()


def run_batch(input_dir: str, output_dir: str = "output"):
    """批量处理模式"""
    from modules.pose_2d import Pose2DEstimator
    from modules.pose_3d import Pose3DReconstructor
    from modules.visualization import Visualizer
    
    print("=" * 50)
    print("人体姿态与形状估计系统 - 批量处理模式")
    print("=" * 50)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"错误: 目录中没有找到图像文件 - {input_dir}")
        sys.exit(1)
    
    print(f"\n找到 {len(image_files)} 个图像文件")
    
    # 初始化模型
    print("\n初始化模型...")
    pose_2d_estimator = Pose2DEstimator()
    pose_3d_reconstructor = Pose3DReconstructor()
    visualizer = Visualizer()
    
    # 处理结果统计
    success_count = 0
    fail_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_file.name}")
        
        try:
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"  [X] 无法读取图像")
                fail_count += 1
                continue
            
            # 2D姿态估计
            pose_2d_result = pose_2d_estimator.estimate(image)
            if pose_2d_result is None:
                print(f"  [X] 未检测到人体")
                fail_count += 1
                continue
            
            # 3D姿态重建
            pose_3d_result = pose_3d_reconstructor.reconstruct(image)
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = image_file.stem
            
            # 保存3D模型
            mesh_path = output_path / f"{base_name}_mesh.obj"
            pose_3d_reconstructor.export_mesh(pose_3d_result, str(mesh_path))
            
            # 保存JSON数据
            json_data = {
                "image_path": str(image_file),
                "pose_2d": pose_2d_estimator.to_dict(pose_2d_result),
                "pose_3d": pose_3d_reconstructor.to_dict(pose_3d_result)
            }
            json_path = output_path / f"{base_name}_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"  [OK] 完成 (置信度: {pose_2d_result.confidence:.2%})")
            success_count += 1
            
        except Exception as e:
            print(f"  [X] 处理失败: {e}")
            fail_count += 1
    
    print("\n" + "=" * 50)
    print(f"处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print(f"输出目录: {output_path.absolute()}")
    
    # 清理资源
    pose_2d_estimator.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="人体姿态与形状估计系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  启动Web服务:
    python main.py --web
    
  处理单张图片:
    python main.py --image path/to/image.jpg
    
  批量处理:
    python main.py --batch path/to/images/ --output results/
"""
    )
    
    parser.add_argument('--web', action='store_true',
                       help='启动Web服务器')
    parser.add_argument('--image', type=str,
                       help='处理单张图片')
    parser.add_argument('--batch', type=str,
                       help='批量处理目录中的图片')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录 (默认: output)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web服务器端口 (默认: 5000)')
    
    args = parser.parse_args()
    
    if args.web:
        run_web_server()
    elif args.image:
        run_cli(args.image, args.output)
    elif args.batch:
        run_batch(args.batch, args.output)
    else:
        # 默认启动Web服务
        print("提示: 使用 --help 查看所有选项")
        print("默认启动Web服务...\n")
        run_web_server()


if __name__ == "__main__":
    main()

