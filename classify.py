from pathlib import Path
import shutil
from collections import Counter
from typing import Dict, Union

from ultralytics import YOLO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def _iter_images_one_level(source_dir: Path):
    """遍历源目录顶层图片，以及最多一层子文件夹内的图片。"""
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    yield child


def _unique_dest_path(dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 1
    while True:
        candidate = dest_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _predict_image(model: YOLO, img_path: Path, conf: float):
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=0.45,
        save=False,
        verbose=False,
    )
    return results[0]


def _get_qualifying_class_names(result) -> set:
    """返回置信度达标检测框对应的类别名（去重，每类最多一个）。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return set()

    names = result.names or {}
    class_names = set()
    for cls_id in boxes.cls.tolist():
        class_name = names.get(int(cls_id), str(int(cls_id)))
        class_names.add(class_name)
    return class_names


def classify_folder(
    model_path: Union[str, Path],
    source_dir: Union[str, Path],
    conf: float = 0.25,
) -> Dict[str, int]:
    """
    遍历文件夹顶层及下一层子文件夹中的图片，将原图按检测类别复制到对应子目录（不移动原图）。

    规则:
        - 读取 source_dir 下图片，以及 source_dir/*/ 下一层文件夹内的图片（不再往下递归）
        - 检测框置信度 >= conf 时，在该类别子目录下创建原图副本
        - 同一张图片在同一类别下最多创建 1 个副本（多个同类框也只复制一次）
        - 一张图片可同时复制到多个不同类别目录
        - 无任何达标检测框时，跳过该图片，不做任何处理

    参数:
        model_path: 训练好的模型路径 (.pt 文件)
        source_dir: 待分类图片文件夹
        conf: 置信度阈值

    返回:
        统计信息，含 total 及各类别复制数量
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir 不是有效文件夹: {source_dir}")

    model = YOLO(str(model_path))
    stats: Counter = Counter()
    # 先收集路径，避免同一次运行写入的类别目录被再次扫描
    images = list(_iter_images_one_level(source_dir))

    for img_path in images:
        result = _predict_image(model, img_path, conf)
        class_names = _get_qualifying_class_names(result)
        if not class_names:
            continue

        for class_name in sorted(class_names):
            dest_dir = source_dir / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = _unique_dest_path(dest_dir, img_path.name)
            shutil.copy2(str(img_path), str(dest_path))
            print(f"{img_path} -> {class_name} -> {dest_path}")

            stats[class_name] += 1

    result = dict(stats)
    result['total'] = sum(stats.values())
    print(f"分类完成，共复制 {result['total']} 份")
    return result


if __name__ == '__main__':
    """粗分类图片到各个类别 
    'unknown': 973, 
    'biaoju': 1463, 
    'reshuiqi': 400, 
    'biaoshi': 938, 
    'jianceyi': 3310, 
    'zaoju': 840, 
    'yongqihuanjing': 299, 
    'liguan': 467, 
    'yandao': 284, 
    'zaoyan': 732, 
    'total': 9706
    """
    model_path = '/Users/tjw/Desktop/work/workplace/视觉/禁用贴/task_result_20260813_153844/weights/best.pt'
    source_dir = '/Users/tjw/Desktop/work/workplace/图片/07200006'  # 替换为待分类图片文件夹

    stats = classify_folder(model_path, source_dir, conf=0.89)
    print("统计:", stats)
