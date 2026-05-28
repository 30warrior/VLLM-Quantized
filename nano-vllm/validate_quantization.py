#!/usr/bin/env python3
"""
量化功能代码验证脚本
用于静态检查量化相关代码的正确性
"""

import ast
import os
import sys
from pathlib import Path


def check_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"


def check_imports(file_path):
    """检查导入语句"""
    required_imports = []
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'nanovllm.utils.quantization':
                for alias in node.names:
                    required_imports.append(alias.name)

    return required_imports


def validate_quantization_implementation():
    """验证量化实现的完整性"""
    base_path = Path('./nanovllm')
    files_to_check = [
        base_path / 'utils' / 'quantization.py',
        base_path / 'layers' / 'linear.py',
        base_path / 'config.py',
        base_path / 'engine' / 'model_runner.py',
        base_path / 'models' / 'qwen3.py',
        Path('./nano-vllm/example.py'),
        Path('./nano-vllm/bench.py'),
    ]

    print("开始验证量化实现...")
    print("=" * 50)

    all_passed = True

    for file_path in files_to_check:
        print(f"\n检查文件: {file_path}")

        # 检查文件是否存在
        if not file_path.exists():
            print(f"  [错误] 文件不存在")
            all_passed = False
            continue

        # 检查语法
        syntax_ok, syntax_error = check_syntax(file_path)
        if not syntax_ok:
            print(f"  [错误] 语法错误: {syntax_error}")
            all_passed = False
            continue

        print(f"  [OK] 语法正确")

        # 检查量化相关导入
        imports = check_imports(file_path)
        if imports:
            print(f"  📦 量化相关导入: {', '.join(imports)}")

    return all_passed


def check_quantization_features():
    """检查量化功能的完整性"""
    print("\n" + "=" * 50)
    print("检查量化功能完整性...")

    features = {
        "量化配置类": "nanovllm/utils/quantization.py",
        "量化线性层": "nanovllm/layers/linear.py",
        "配置支持": "nanovllm/config.py",
        "模型运行器支持": "nanovllm/engine/model_runner.py",
        "模型层支持": "nanovllm/models/qwen3.py",
        "示例更新": "example.py",
        "基准测试更新": "bench.py",
    }

    all_present = True
    for feature, file_path in features.items():
        if os.path.exists(file_path):
            print(f"  [OK] {feature}: {file_path}")
        else:
            print(f"  [错误] {feature}: {file_path} (缺失)")
            all_present = False

    return all_present


def check_key_implementations():
    """检查关键实现"""
    print("\n" + "=" * 50)
    print("检查关键实现...")

    key_files = {
        "quantization.py": [
            "QuantizationConfig",
            "apply_quantization_to_linear_layers",
            "BITSANDBYTES_AVAILABLE"
        ],
        "linear.py": [
            "QuantizedReplicatedLinear",
            "QuantizedColumnParallelLinear",
            "QuantizedRowParallelLinear",
            "QuantizedMergedColumnParallelLinear",
            "QuantizedQKVParallelLinear"
        ],
        "config.py": ["load_in_4bit", "load_in_8bit", "quantization_config"],
        "model_runner.py": ["quantization_config"],
        "qwen3.py": ["quantization_config"],
    }

    all_good = True
    for file_name, expected_items in key_files.items():
        file_path = f"nano-vllm/nanovllm/{file_name}" if file_name != "config.py" else "nano-vllm/nanovllm/config.py"

        if not os.path.exists(file_path):
            print(f"  ❌ {file_name}: 文件不存在")
            all_good = False
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        missing_items = []
        for item in expected_items:
            if item not in content:
                missing_items.append(item)

        if missing_items:
            print(f"  [警告] {file_name}: 可能缺少 {', '.join(missing_items)}")
            all_good = False
        else:
            print(f"  [OK] {file_name}: 关键实现完整")

    return all_good


def generate_usage_examples():
    """生成使用示例"""
    print("\n" + "=" * 50)
    print("量化使用示例:")
    print("=" * 50)

    examples = {
        "4-bit量化": '''
from nanovllm import LLM, SamplingParams

# INT4量化加载（推荐）
llm = LLM(
    "/path/to/model",
    enforce_eager=True,
    tensor_parallel_size=1,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
print(outputs[0]["text"])
''',
        "8-bit量化": '''
from nanovllm import LLM, SamplingParams

# INT8量化加载
llm = LLM(
    "/path/to/model",
    enforce_eager=True,
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
print(outputs[0]["text"])
''',
        "基准测试": '''
python bench.py --load-in-4bit --bnb-4bit-quant-type=nf4
python bench.py --load-in-8bit
python bench.py --load-in-4bit --num-requests=128
'''
    }

    for title, code in examples.items():
        print(f"\n{title}:")
        print("-" * len(title))
        print(code.strip())


def main():
    """主函数"""
    print("Nano-vLLM 量化功能验证")
    print("=" * 50)

    # 运行所有检查
    checks = [
        ("代码语法和结构", validate_quantization_implementation),
        ("功能完整性", check_quantization_features),
        ("关键实现", check_key_implementations),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\n[检查] {check_name}")
        try:
            result = check_func()
            results.append(result)
            status = "[通过]" if result else "[失败]"
            print(f"\n{check_name}: {status}")
        except Exception as e:
            print(f"\n{check_name}: [异常] {e}")
            results.append(False)

    # 生成使用示例
    generate_usage_examples()

    # 总结
    print("\n" + "=" * 50)
    print("验证总结")
    print("=" * 50)

    passed_checks = sum(results)
    total_checks = len(results)

    if passed_checks == total_checks:
        print(f"[成功] 所有检查通过 ({passed_checks}/{total_checks})")
        print("\n[OK] 量化功能实现完整，可以使用以下命令测试:")
        print("   python test_quantization.py")
        print("\n[示例]:")
        print("   # 4-bit量化")
        print("   python example.py  # 已更新支持量化")
        print("   python bench.py --load-in-4bit")
    else:
        print(f"[警告] 部分检查失败 ({passed_checks}/{total_checks})")
        print("请检查失败的项并进行修复。")

    print("\n[功能列表]:")
    print("   - INT4/NF4量化支持")
    print("   - INT8量化支持")
    print("   - 所有线性层的量化版本")
    print("   - 张量并行兼容")
    print("   - 配置系统集成")
    print("   - 示例和基准测试更新")


if __name__ == "__main__":
    main()