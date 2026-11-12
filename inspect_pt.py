"""
inspect_pt.py — 检查 .pt 文件的结构与元信息

用法:
    python inspect_pt.py path/to/model.pt
    python inspect_pt.py best.pt                  # 根目录下的模型
    python inspect_pt.py *.pt                     # 批量检查
"""

import sys
import os
import torch


def inspect_pt(file_path: str) -> None:
    """加载并打印单个 .pt 文件的结构信息。"""
    if not os.path.isfile(file_path):
        print(f"[ERROR] 文件不存在: {file_path}")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print("=" * 60)
    print(f"文件: {file_path}")
    print(f"大小: {file_size_mb:.2f} MB")
    print("=" * 60)

    try:
        weights = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        return

    print(f"顶层类型: {type(weights).__name__}")

    # ---------- dict 类型 ----------
    if isinstance(weights, dict):
        print(f"字典键 ({len(weights)}): {list(weights.keys())}")
        print()

        # YOLO 系列常见键
        if "model" in weights:
            model_obj = weights["model"]
            print(f"  [model] 类型: {type(model_obj).__name__}")
            if hasattr(model_obj, "names"):
                print(f"  [model] 类别数: {len(model_obj.names)}, 类别: {model_obj.names}")
            if hasattr(model_obj, "yaml"):
                print(f"  [model] YAML 配置: {model_obj.yaml}")

        for key in ("state_dict", "model_state_dict"):
            if key in weights and isinstance(weights[key], dict):
                sd = weights[key]
                print(f"  [{key}] 参数数量: {len(sd)}")
                for name, param in list(sd.items())[:5]:
                    shape = param.shape if hasattr(param, "shape") else "?"
                    print(f"    {name}: {shape}")
                if len(sd) > 5:
                    print(f"    ... 还有 {len(sd) - 5} 个参数")
                print()

        # epoch / optimizer 等训练相关键
        for key in ("epoch", "best_fitness", "optimizer", "ema"):
            if key in weights:
                val = weights[key]
                if not isinstance(val, (dict, torch.nn.Module)):
                    print(f"  [{key}]: {val}")
                else:
                    print(f"  [{key}]: {type(val).__name__}")

    # ---------- 模型对象 ----------
    elif hasattr(weights, "state_dict"):
        print(f"✅ 整个模型对象 (type={type(weights).__name__})")
        sd = weights.state_dict()
        print(f"参数数量: {len(sd)}")
        for name, param in list(sd.items())[:10]:
            shape = param.shape if hasattr(param, "shape") else "?"
            print(f"  {name}: {shape}")
        if len(sd) > 10:
            print(f"  ... 还有 {len(sd) - 10} 个参数")

        # YOLO 特有属性
        if hasattr(weights, "names"):
            print(f"\n类别数: {len(weights.names)}, 类别: {weights.names}")
        if hasattr(weights, "yaml"):
            print(f"YAML 配置: {weights.yaml}")

    else:
        print(f"⚠️  未识别的顶层类型: {type(weights).__name__}")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    for path in sys.argv[1:]:
        inspect_pt(path)


if __name__ == "__main__":
    main()
