"""
SMPL模型转换脚本
将原始SMPL pkl文件转换为纯numpy格式，绕过chumpy依赖
"""

import pickle
import numpy as np
import sys
from pathlib import Path


def convert_smpl_model(input_path: str, output_path: str = None):
    """
    转换SMPL模型为纯numpy格式
    
    Args:
        input_path: 原始SMPL .pkl 文件路径
        output_path: 输出的 .npz 文件路径（默认与输入同名）
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix('.npz')
    else:
        output_path = Path(output_path)
    
    print(f"正在转换: {input_path}")
    print(f"输出到: {output_path}")
    
    # 自定义Unpickler来处理chumpy对象
    class SMPLUnpickler(pickle.Unpickler):
        """处理包含chumpy对象的pickle文件"""
        
        def find_class(self, module, name):
            # 拦截chumpy相关的类
            if 'chumpy' in module:
                return self._make_chumpy_replacement(name)
            return super().find_class(module, name)
        
        def _make_chumpy_replacement(self, name):
            """创建chumpy对象的替代类"""
            class ChumpyReplacement:
                def __init__(self, *args, **kwargs):
                    self._value = None
                    if args:
                        self._value = args[0]
                    
                def __reduce__(self):
                    return (ChumpyReplacement, (self._value,))
                
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self._value = state.get('x', state.get('a', None))
                        if self._value is None:
                            for v in state.values():
                                if isinstance(v, np.ndarray):
                                    self._value = v
                                    break
                    elif isinstance(state, np.ndarray):
                        self._value = state
                
                def r(self):
                    return self._value
                
                def __array__(self):
                    return np.array(self._value)
            
            return ChumpyReplacement
    
    # 加载原始模型
    with open(input_path, 'rb') as f:
        smpl_data = SMPLUnpickler(f, encoding='latin1').load()
    
    def extract_value(x):
        """从各种类型中提取numpy数组"""
        if x is None:
            return None
        if hasattr(x, 'r') and callable(x.r):
            val = x.r()
            if val is not None:
                return np.array(val, dtype=np.float32)
        if hasattr(x, '_value') and x._value is not None:
            return np.array(x._value, dtype=np.float32)
        if hasattr(x, '__array__'):
            return np.array(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        return np.array(x, dtype=np.float32)
    
    # 提取所有参数
    result = {}
    
    # 基本参数
    result['v_template'] = extract_value(smpl_data['v_template'])
    result['shapedirs'] = extract_value(smpl_data['shapedirs'])
    result['posedirs'] = extract_value(smpl_data['posedirs'])
    result['weights'] = extract_value(smpl_data['weights'])
    result['f'] = np.array(smpl_data['f'], dtype=np.int32)
    
    # J_regressor是稀疏矩阵
    J_regressor = smpl_data['J_regressor']
    if hasattr(J_regressor, 'toarray'):
        result['J_regressor'] = J_regressor.toarray().astype(np.float32)
    else:
        result['J_regressor'] = np.array(J_regressor, dtype=np.float32)
    
    # kinematic tree
    result['kintree_table'] = np.array(smpl_data['kintree_table'], dtype=np.int32)
    
    # 保存为npz
    np.savez(output_path, **result)
    
    print(f"\n转换完成!")
    print(f"  - v_template: {result['v_template'].shape}")
    print(f"  - shapedirs: {result['shapedirs'].shape}")
    print(f"  - posedirs: {result['posedirs'].shape}")
    print(f"  - weights: {result['weights'].shape}")
    print(f"  - J_regressor: {result['J_regressor'].shape}")
    print(f"  - faces: {result['f'].shape}")
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_smpl.py <smpl_model.pkl> [output.npz]")
        print("\n示例:")
        print("  python convert_smpl.py models/smpl/basicModel_neutral_lbs_10_207_0_v1.1.0.pkl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_smpl_model(input_path, output_path)


if __name__ == "__main__":
    main()

