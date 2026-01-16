"""
人体姿态与形状估计系统 - Web应用
Flask后端服务
"""

import os
import uuid
import yaml
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import traceback
import json

# 导入自定义模块
from modules.pose_2d import Pose2DEstimator, PoseResult2D
from modules.pose_3d import Pose3DReconstructor, Pose3DResult
from modules.visualization import Visualizer, VisualizationConfig
from modules.garment_segmentation import GarmentSegmenter, SegmentationResult
from modules.garment_semantic import GarmentSemanticAnalyzer
from modules.garment_3d import Garment3DReconstructor


# 加载配置
def load_config(config_path: str = 'config.yaml') -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


# 初始化Flask应用
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 加载配置
config = load_config()

# 配置上传文件夹
UPLOAD_FOLDER = config.get('upload', {}).get('folder', 'uploads')
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = set(config.get('upload', {}).get('allowed_extensions', ['jpg', 'jpeg', 'png', 'webp']))
MAX_CONTENT_LENGTH = config.get('upload', {}).get('max_size_mb', 10) * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# 全局模型实例
pose_2d_estimator = None
pose_3d_reconstructor = None
visualizer = None
garment_segmenter = None
garment_semantic_analyzer = None
garment_3d_reconstructor = None


def init_models():
    """初始化模型"""
    global pose_2d_estimator, pose_3d_reconstructor, visualizer
    global garment_segmenter, garment_semantic_analyzer, garment_3d_reconstructor
    
    pose_2d_config = config.get('pose_2d', {})
    pose_3d_config = config.get('pose_3d', {})
    vis_config = config.get('visualization', {})
    
    # 初始化2D姿态估计器
    pose_2d_estimator = Pose2DEstimator(
        min_detection_confidence=pose_2d_config.get('min_detection_confidence', 0.5),
        min_tracking_confidence=pose_2d_config.get('min_tracking_confidence', 0.5),
        model_complexity=pose_2d_config.get('model_complexity', 2)
    )
    
    # 初始化3D重建器
    smpl_model_path = config.get('models', {}).get('smpl_model_path')
    device = config.get('models', {}).get('device', 'cpu')
    
    # 检查是否有CUDA
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用，使用CPU模式")
    
    pose_3d_reconstructor = Pose3DReconstructor(
        smpl_model_path=smpl_model_path,
        device=device,
        input_size=pose_3d_config.get('input_size', 224)
    )
    
    # 初始化可视化器
    visualizer = Visualizer(VisualizationConfig(
        render_width=vis_config.get('render_width', 800),
        render_height=vis_config.get('render_height', 600)
    ))
    
    # 初始化服装分析模块
    garment_segmenter = GarmentSegmenter()
    garment_semantic_analyzer = GarmentSemanticAnalyzer()
    garment_3d_reconstructor = Garment3DReconstructor()
    
    print("[OK] 所有模型初始化完成")


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_result_id() -> str:
    """生成唯一的结果ID"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


# ==================== 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': pose_2d_estimator is not None and pose_3d_reconstructor is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传图片文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件格式，仅支持: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # 保存文件
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'filepath': filepath,
            'message': '文件上传成功'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/estimate', methods=['POST'])
def estimate_pose():
    """执行姿态估计"""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': '缺少文件名参数'}), 400
    
    filename = data['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 404
    
    try:
        # 读取图像
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '无法读取图像文件'}), 400
        
        result_id = generate_result_id()
        result = {'result_id': result_id}
        
        # 2D姿态估计
        print("执行2D姿态估计...")
        pose_2d_result = pose_2d_estimator.estimate(image)
        
        if pose_2d_result is None:
            return jsonify({'error': '未检测到人体，请上传包含完整人体的图片'}), 400
        
        result['pose_2d'] = pose_2d_estimator.to_dict(pose_2d_result)
        
        # 3D姿态重建 - 使用2D关键点来改进3D估计
        print("执行3D姿态重建...")
        pose_3d_result = pose_3d_reconstructor.reconstruct(
            image, 
            keypoints_2d=pose_2d_result.landmarks_pixel  # 传入2D关键点
        )
        result['pose_3d'] = pose_3d_reconstructor.to_dict(pose_3d_result)
        
        # 生成可视化
        print("生成可视化结果...")
        visualizations = {}
        
        # 2D姿态可视化
        pose_2d_vis = visualizer.visualize_pose_2d(
            image,
            pose_2d_result.landmarks_pixel,
            pose_2d_estimator.SKELETON_CONNECTIONS,
            [kp.visibility for kp in pose_2d_result.keypoints],
            title="2D Pose Estimation"
        )
        visualizations['pose_2d'] = pose_2d_vis
        
        # 3D骨架可视化 (多角度)
        skeleton_3d_front = visualizer.visualize_skeleton_3d(
            pose_3d_result.joints_3d,
            title="3D Skeleton - Front View",
            elev=10, azim=-90  # 正面视图
        )
        skeleton_3d_side = visualizer.visualize_skeleton_3d(
            pose_3d_result.joints_3d,
            title="3D Skeleton - Side View",
            elev=10, azim=0  # 侧面视图
        )
        visualizations['skeleton_3d_front'] = skeleton_3d_front
        visualizations['skeleton_3d_side'] = skeleton_3d_side
        
        # 3D网格可视化
        mesh_3d = visualizer.visualize_mesh_3d(
            pose_3d_result.vertices,
            pose_3d_result.faces,
            title="3D Body Mesh"
        )
        visualizations['mesh_3d'] = mesh_3d
        
        # 形状参数可视化
        shape_vis = visualizer.visualize_shape_parameters(
            pose_3d_result.smpl_params.shape
        )
        visualizations['shape_params'] = shape_vis
        
        result['visualizations'] = visualizations
        
        # 保存结果
        result_path = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            # 保存时不包含base64图像数据（太大）
            save_result = {k: v for k, v in result.items() if k != 'visualizations'}
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        # 导出3D模型
        mesh_path = os.path.join(RESULTS_FOLDER, f"{result_id}_mesh.obj")
        pose_3d_reconstructor.export_mesh(pose_3d_result, mesh_path)
        result['mesh_file'] = f"{result_id}_mesh.obj"
        
        print(f"[OK] 处理完成: {result_id}")
        
        return jsonify({
            'success': True,
            'result': result,
            'message': '姿态估计完成'
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/api/analyze_garment', methods=['POST'])
def analyze_garment():
    """分析服装（从人像或单独服装图片）"""
    import base64
    
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': '缺少文件名参数'}), 400
    
    filename = data['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    mode = data.get('mode', 'garment')  # 'person' 或 'garment'
    
    if not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 404
    
    try:
        # 读取图像
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '无法读取图像文件'}), 400
        
        result_id = generate_result_id()
        
        print("分析服装图片...")
        
        # 分割服装
        garment_region = garment_segmenter.segment_standalone_garment(image)
        
        if garment_region is None:
            return jsonify({'error': '未能检测到服装'}), 400
        
        # 语义分析
        semantics = garment_semantic_analyzer.analyze(
            image,
            garment_region.mask,
            garment_region.garment_type.value
        )
        
        # 3D重建
        garment_3d = garment_3d_reconstructor.reconstruct(
            image,
            garment_region.mask,
            garment_region.contour,
            garment_region.garment_type.value,
            semantics.material.value
        )
        
        # 生成可视化图像
        visualizations = {}
        
        # 1. 分割可视化
        seg_vis = garment_segmenter.visualize_segmentation(image, garment_region)
        _, seg_buffer = cv2.imencode('.png', seg_vis)
        visualizations['segmentation'] = base64.b64encode(seg_buffer).decode('utf-8')
        
        # 2. 3D投影可视化
        proj_vis = garment_3d_reconstructor.visualize_projection(garment_3d)
        _, proj_buffer = cv2.imencode('.png', proj_vis)
        visualizations['projection_3d'] = base64.b64encode(proj_buffer).decode('utf-8')
        
        # 3. UV贴图可视化
        uv_vis = garment_3d_reconstructor.visualize_uv_map(garment_3d)
        _, uv_buffer = cv2.imencode('.png', uv_vis)
        visualizations['uv_map'] = base64.b64encode(uv_buffer).decode('utf-8')
        
        # 语义信息转换
        semantic_info = garment_semantic_analyzer.to_dict(semantics)
        semantic_result = {
            'garment_type': semantic_info.get('garment_type', '--'),
            'primary_color': semantic_info.get('primary_color', '--'),
            'material': semantic_info.get('material', '--'),
            'style': semantic_info.get('style', '--'),
            'sleeve_type': semantic_info.get('sleeve_type', '--'),
            'neckline': semantic_info.get('neckline', '--'),
            'confidence_scores': semantic_info.get('confidence_scores', {})
        }
        
        # 导出3D模型
        garment_mesh_path = os.path.join(RESULTS_FOLDER, f"{result_id}_garment_mesh.obj")
        garment_3d_reconstructor.export_obj(garment_3d, garment_mesh_path)
        
        # 保存纹理贴图
        texture_path = os.path.join(RESULTS_FOLDER, f"{result_id}_texture.png")
        garment_3d_reconstructor.export_texture(garment_3d, texture_path)
        
        result = {
            'result_id': result_id,
            'visualizations': visualizations,
            'semantic': semantic_result,
            'mesh_file': f"{result_id}_garment_mesh.obj",
            'texture_file': f"{result_id}_texture.png",
            'garment_info': {
                'type': garment_region.garment_type.value,
                'bbox': garment_region.bbox,
                'area': garment_region.area
            }
        }
        
        print(f"[OK] 服装分析完成: {result_id}")
        
        return jsonify({
            'success': True,
            'result': result,
            'message': '服装分析完成'
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """下载结果文件"""
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """提供上传的文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/results/<result_id>')
def get_result(result_id):
    """获取历史结果"""
    result_path = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    
    if not os.path.exists(result_path):
        return jsonify({'error': '结果不存在'}), 404
    
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    return jsonify({'success': True, 'result': result})


@app.route('/api/results')
def list_results():
    """列出所有结果"""
    results = []
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.endswith('.json'):
            result_id = filename[:-5]
            results.append({
                'result_id': result_id,
                'created_at': os.path.getctime(os.path.join(RESULTS_FOLDER, filename))
            })
    
    # 按时间排序
    results.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({'success': True, 'results': results})


# ==================== 启动 ====================

if __name__ == '__main__':
    print("=" * 50)
    print("人体姿态与形状估计系统")
    print("=" * 50)
    
    # 初始化模型
    print("\n正在初始化模型...")
    init_models()
    
    # 启动服务器
    host = config.get('server', {}).get('host', '0.0.0.0')
    port = config.get('server', {}).get('port', 5000)
    debug = config.get('server', {}).get('debug', True)
    
    print(f"\n服务器启动地址: http://{host}:{port}")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)

