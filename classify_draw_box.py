import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
DETECTED_DIR = 'detected'
UNKNOWN_DIR = 'unknow'
CLS_OUTPUT_DIR = 'classify_result'
SKIP_DIRS = {DETECTED_DIR, UNKNOWN_DIR, CLS_OUTPUT_DIR}


# ===================== 通用 =====================

def _iter_images_one_level(source_dir: Path) -> List[Path]:
    """收集源目录顶层图片，以及下一层子文件夹内的图片（跳过输出目录）。"""
    images: List[Path] = []
    folder_counts: Counter = Counter()

    for path in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
            folder_counts['.'] += 1
        elif path.is_dir() and path.name not in SKIP_DIRS:
            n = 0
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    images.append(child)
                    n += 1
            folder_counts[path.name] = n

    print(f'扫描到 {len(folder_counts)} 个目录，共 {len(images)} 张图片:')
    for name, n in folder_counts.items():
        print(f'  {name}: {n}')
    return images


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


def _judge_model_kind(model_path: Union[str, Path]) -> str:
    """判断模型类型: 'detect' 目标检测 / 'classify' 分类。
    .pt 用 ultralytics 元数据；.onnx 无元数据，按输出维度判断（分类 [1,C]，检测 [1,N,M]）。
    """
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    if suffix == '.pt':
        task = getattr(YOLO(str(model_path)), 'task', None)
        return 'classify' if task == 'classify' else 'detect'
    if suffix == '.onnx':
        out_shape = ort.InferenceSession(str(model_path)).get_outputs()[0].shape
        return 'classify' if len(out_shape) == 2 else 'detect'
    raise ValueError(f"不支持的模型格式: {suffix}，仅支持 .pt / .onnx")


# ===================== 目标检测（旧逻辑，不动） =====================

def _predict_image(model: YOLO, img_path: Path, conf: float):
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=0.45,
        save=False,
        verbose=False,
    )
    return results[0]


def _has_any_box(result) -> bool:
    """有任意达标框即可。"""
    boxes = result.boxes
    return boxes is not None and len(boxes) > 0


def classify_folder_with_boxes(
    model_path: Union[str, Path],
    source_dir: Union[str, Path],
    conf: float = 0.25,
    output_dir: Union[str, Path, None] = None,
) -> Dict[str, int]:
    """
    遍历文件夹顶层及下一层子文件夹中的图片，推理后绘制检测框。
    原图保留，不删除、不移动。

    规则:
        - 检出任意达标框：绘制全部达标框，保存到 detected
        - 未检测到：原图复制到 unknow
        - 坏图/读失败：跳过，不中断整批
        - 跳过 detected / unknow 目录，避免重复处理输出结果
        - output_dir 指定时，detected/unknow 写到该目录；否则写到 source_dir
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir 不是有效文件夹: {source_dir}")
    out_root = Path(output_dir) if output_dir else source_dir
    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    stats: Counter = Counter()
    images = _iter_images_one_level(source_dir)

    for i, img_path in enumerate(images, 1):
        try:
            result = _predict_image(model, img_path, conf)
        except Exception as e:
            print(f"[{i}/{len(images)}] {img_path} -> error: {e}")
            stats['error'] += 1
            continue

        if not _has_any_box(result):
            dest_dir = out_root / UNKNOWN_DIR
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = _unique_dest_path(dest_dir, img_path.name)
            shutil.copy2(str(img_path), str(dest_path))
            print(f"[{i}/{len(images)}] {img_path} -> {UNKNOWN_DIR} -> {dest_path}")
            stats[UNKNOWN_DIR] += 1
            continue

        annotated = result.plot()
        dest_dir = out_root / DETECTED_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = _unique_dest_path(dest_dir, img_path.name)
        cv2.imwrite(str(dest_path), annotated)
        print(f"[{i}/{len(images)}] {img_path} -> {DETECTED_DIR} -> {dest_path}")
        stats[DETECTED_DIR] += 1

    result = dict(stats)
    result['total'] = sum(v for k, v in stats.items() if k != 'error')
    print(
        f"分类完成，检出 {stats[DETECTED_DIR]} 份带框图，"
        f"未检测 {stats[UNKNOWN_DIR]} 份归入 {UNKNOWN_DIR}，错误 {stats['error']}"
    )
    return result


# ===================== 分类（onnxruntime） =====================

class OnnxClassifier:
    """onnx cls 推理（逐张，兼容固定 batch=1）。

    自训模型图内已烘焙 /255 + ImageNet 归一化，直接喂 0-255 RGB 像素，
    再做归一化会把信号压缩 255 倍导致输出趋近常数。
    输入尺寸从模型读取；普通导出无类别名，类别以索引表示。
    """

    def __init__(self, model_path: Union[str, Path]):
        self.sess = ort.InferenceSession(str(model_path))
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        h = inp.shape[-1]
        self.size = int(h) if isinstance(h, int) else 256
        self.num_classes = int(self.sess.get_outputs()[0].shape[-1])

    def predict(self, img_path: Path) -> Tuple[int, float]:
        """返回 (top1 类别索引, top1 概率)。坏图抛异常由调用方统计。"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError('cv2.imread 读取失败（坏图）')
        img = cv2.resize(img[:, :, ::-1], (self.size, self.size)).astype(np.float32)
        out = self.sess.run(None, {self.input_name: img.transpose(2, 0, 1)[None]})[0][0].astype(np.float64)
        if not (out.min() >= 0 and np.isclose(out.sum(), 1.0, atol=1e-3)):  # 图内无 softmax 则补
            e = np.exp(out - out.max())
            out = e / e.sum()
        top1 = int(out.argmax())
        return top1, float(out[top1])


def classify_folder_by_cls(
    model_path: Union[str, Path],
    source_dir: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
) -> Dict[str, int]:
    """cls 分类：按 top1 类别索引把原图副本放入 classify_result/<类别索引>/（原图不动）。"""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir 不是有效文件夹: {source_dir}")

    model = OnnxClassifier(model_path)
    print(f'分类模型: 输入 {model.size}x{model.size}, {model.num_classes} 类')

    base = Path(output_dir) if output_dir else source_dir
    base.mkdir(parents=True, exist_ok=True)
    output_root = base / CLS_OUTPUT_DIR
    stats: Counter = Counter()
    images = _iter_images_one_level(source_dir)

    for i, img_path in enumerate(images, 1):
        try:
            top1, score = model.predict(img_path)
        except Exception as e:
            print(f"[{i}/{len(images)}] {img_path} -> error: {e}")
            stats['error'] += 1
            continue

        dest_dir = output_root / str(top1)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = _unique_dest_path(dest_dir, img_path.name)
        shutil.copy2(str(img_path), str(dest_path))
        print(f"[{i}/{len(images)}] {img_path} -> {top1} ({score:.4f}) -> {dest_path}")
        stats[str(top1)] += 1

    result = dict(stats)
    result['total'] = sum(v for k, v in stats.items() if k != 'error')
    print(f"分类完成: {result}, 输出目录 {output_root}")
    return result


# ===================== 统一入口 =====================

def classify_folder(
    model_path: Union[str, Path],
    source_dir: Union[str, Path],
    conf: float = 0.8,
    output_dir: Union[str, Path, None] = None,
) -> Dict[str, int]:
    """先判断模型类型再走对应流程：检测 -> 旧画框流程；分类 -> 按类别索引归档。"""
    kind = _judge_model_kind(model_path)
    print(f'模型类型: {Path(model_path).name} -> {kind}')
    if kind == 'classify':
        return classify_folder_by_cls(model_path, source_dir, output_dir=output_dir)
    return classify_folder_with_boxes(model_path, source_dir, conf=conf, output_dir=output_dir)


if __name__ == '__main__':
    MODEL_PATH = '/Users/tjw/PycharmProjects/pic_model_deploy/唐俊伟/runzhiwb636/valve_detect/model_file/best.pt'
    SOURCE_DIR = '/Users/tjw/Desktop/work/workplace/图片/J030402-4w_approve_split/safety_device_check/不同意'
    OUTPUT_DIR = '/Users/tjw/Desktop/work/workplace/图片/J030402-4w_approve_split/safety_device_check/result'
    CONF = 0.2

    print(f'模型: {MODEL_PATH}\n源图: {SOURCE_DIR}\n输出: {OUTPUT_DIR}')
    stats = classify_folder(MODEL_PATH, SOURCE_DIR, conf=CONF, output_dir=OUTPUT_DIR)
    print('统计:', stats)
