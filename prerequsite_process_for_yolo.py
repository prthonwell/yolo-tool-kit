"""LabelMe 标注（文件夹或 zip）→ YOLO 训练数据（images/labels + data.yaml + 模型 yaml）。

负样本规则：
- JSON 存在但 shapes 为空 → 生成空 txt
- 图片没有对应 JSON → 同样生成空 txt
"""
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import yaml

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
MAX_SIZE = 640
JPEG_QUALITY = 90
VALID_SHAPES = {'rectangle', 'polygon', 'circle'}
# 输出文件名 → ultralytics 包内源配置（scale 由文件名 n/s/m/l/x 决定）
YOLO_MODELS = {
    'yolo11n': '11/yolo11.yaml',
    'yolo11s': '11/yolo11.yaml',
    'yolo11m': '11/yolo11.yaml',
    'yolo11l': '11/yolo11.yaml',
    'yolo11x': '11/yolo11.yaml',
    'yolo11n-seg': '11/yolo11-seg.yaml',
    'yolo11s-seg': '11/yolo11-seg.yaml',
    'yolo11m-seg': '11/yolo11-seg.yaml',
    'yolo11l-seg': '11/yolo11-seg.yaml',
    'yolo11x-seg': '11/yolo11-seg.yaml',
}


def _shape_bbox(shape: dict):
    shape_type = shape.get('shape_type')
    points = shape.get('points') or []

    if shape_type == 'rectangle' and len(points) >= 2:
        xs = [p[0] for p in points[:2]]
        ys = [p[1] for p in points[:2]]
    elif shape_type == 'polygon' and len(points) >= 3:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
    elif shape_type == 'circle' and len(points) >= 2:
        cx, cy = points[0]
        r = ((points[1][0] - cx) ** 2 + (points[1][1] - cy) ** 2) ** 0.5
        xs = [cx - r, cx + r]
        ys = [cy - r, cy + r]
    else:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def json_to_detect_lines(data: dict, class_list, scale: float, new_w: int, new_h: int):
    lines = []
    for shape in data.get('shapes', []) or []:
        if shape.get('shape_type') not in VALID_SHAPES:
            continue
        label = shape.get('label', '')
        if label not in class_list:
            continue
        bbox = _shape_bbox(shape)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = x_min * scale, x_max * scale
        y_min, y_max = y_min * scale, y_max * scale
        x_center = ((x_min + x_max) / 2) / new_w
        y_center = ((y_min + y_max) / 2) / new_h
        bw = (x_max - x_min) / new_w
        bh = (y_max - y_min) / new_h
        if bw <= 0 or bh <= 0:
            continue
        lines.append(
            f'{class_list.index(label)} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}'
        )
    return lines


def json_to_segment_lines(data: dict, class_list):
    w = data.get('imageWidth')
    h = data.get('imageHeight')
    if not w or not h:
        return []
    lines = []
    for shape in data.get('shapes', []) or []:
        label = shape.get('label', '')
        if label not in class_list:
            continue
        points = shape.get('points') or []
        if len(points) < 3:
            continue
        coords = []
        for x, y in points:
            coords.append(f'{x / w:.6f}')
            coords.append(f'{y / h:.6f}')
        lines.append(f'{class_list.index(label)} {" ".join(coords)}')
    return lines


def _strip_image_exts(name: str) -> str:
    """去掉重复图片后缀，兼容 foo.jpg.jpg → foo。"""
    stem = name
    while True:
        low = stem.lower()
        stripped = False
        for ext in IMAGE_EXTS:
            if low.endswith(ext):
                stem = stem[: -len(ext)]
                stripped = True
                break
        if not stripped:
            break
    return stem


def _sample_id(path: Path) -> str:
    """样本 ID：优先按去后缀后的文件名，兼容 img.jpg + img.jpg.json。"""
    if path.suffix.lower() == '.json':
        return _strip_image_exts(path.stem)
    return _strip_image_exts(path.name)


def _find_json_for_image(json_dir: Path, img_path: Path):
    """按图片匹配 JSON：优先 stem.json，再试 去后缀.json / stem.jpg.json 等。"""
    candidates = [
        json_dir / f'{img_path.stem}.json',
        json_dir / f'{_sample_id(img_path)}.json',
        json_dir / f'{img_path.name}.json',
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _find_image_for_stem(img_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = img_dir / f'{stem}{ext}'
        if p.is_file():
            return p
        p2 = img_dir / f'{stem}{ext.upper()}'
        if p2.is_file():
            return p2
    # 兼容 stem 已含后缀 / 实际文件为 xxx.jpg.jpg
    target = _strip_image_exts(stem)
    for path in _iter_images(img_dir):
        if _sample_id(path) == target or path.stem == stem:
            return path
    return None


def _iter_images(img_dir: Path):
    for path in sorted(img_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def _resolve_img_json_dirs(input_dir: Path):
    """支持同目录同名，或 images/ + annotations/。"""
    input_dir = Path(input_dir)
    images_dir = input_dir / 'images'
    ann_dir = input_dir / 'annotations'
    if images_dir.is_dir() and ann_dir.is_dir():
        return images_dir, ann_dir
    return input_dir, input_dir


def collect_image_json_pairs(input_dir: Path):
    """以图片为主收集样本：(img_path, json_path|None)。

    无对应 JSON 的图片也会收录，json_path 为 None，后续写空 txt 负样本。
    """
    img_dir, json_dir = _resolve_img_json_dirs(input_dir)
    pairs = []
    for img_path in _iter_images(img_dir):
        pairs.append((img_path, _find_json_for_image(json_dir, img_path)))

    matched_json = {jp.resolve() for _, jp in pairs if jp is not None}
    orphan_json = 0
    for json_path in sorted(json_dir.glob('*.json')):
        if json_path.resolve() not in matched_json:
            orphan_json += 1
    return pairs, orphan_json


def _load_labelme(json_path):
    """读取 LabelMe JSON；缺失或不合法时返回空 shapes。"""
    if json_path is None:
        return {'shapes': []}
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'shapes': []}
        data.setdefault('shapes', [])
        return data
    except (OSError, json.JSONDecodeError):
        return {'shapes': []}


def collect_class_names(input_dir: Path):
    names = set()
    pairs, _ = collect_image_json_pairs(input_dir)
    if not pairs:
        for child in sorted(p for p in Path(input_dir).iterdir() if p.is_dir()):
            if child.name in {'__MACOSX', '.DS_Store'}:
                continue
            pairs, _ = collect_image_json_pairs(child)
            if pairs:
                break
    for _, json_path in pairs:
        if json_path is None:
            continue
        data = _load_labelme(json_path)
        for shape in data.get('shapes', []) or []:
            label = (shape.get('label') or '').strip()
            if label:
                names.add(label)
    return sorted(names)


def _make_splits(pairs, train_ratio: float, val_ratio: float, test_ratio: float):
    n = len(pairs)
    if n == 0:
        return {'train': [], 'val': [], 'test': []}
    if n == 1:
        return {'train': pairs[:], 'val': pairs[:], 'test': []}
    if n == 2:
        return {'train': pairs[:1], 'val': pairs[1:], 'test': []}

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_val <= 0:
        n_val = 1
        if n_train > 1:
            n_train -= 1
        elif n_test > 0:
            n_test -= 1
    if n_train <= 0:
        n_train = 1
        if n_test > 0:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
    n_test = max(0, n - n_train - n_val)
    return {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:n_train + n_val + n_test],
    }


def _write_data_yaml(output_dir: Path, class_list) -> Path:
    content = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(class_list)},
    }
    dest = output_dir / 'data.yaml'
    with open(dest, 'w', encoding='utf-8') as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
    return dest


def _resolve_model_name(model: str, mode: str) -> str:
    model = (model or 'yolo11n').strip()
    if mode in {'segment', 'segment_auto'} and not model.endswith('-seg'):
        model = f'{model}-seg'
    return model


def copy_model_yaml(model: str, output_dir: Path) -> Path:
    """从 ultralytics 包复制模型结构 yaml 到输出目录，命名为 {model}.yaml。"""
    if model not in YOLO_MODELS:
        raise ValueError(f'不支持的模型: {model}，可选: {", ".join(YOLO_MODELS)}')

    import ultralytics

    src = Path(ultralytics.__file__).parent / 'cfg' / 'models' / YOLO_MODELS[model]
    if not src.is_file():
        raise FileNotFoundError(f'ultralytics 中未找到模型配置: {src}')

    dest = Path(output_dir) / f'{model}.yaml'
    shutil.copy2(src, dest)
    return dest


def _unzip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    children = [p for p in dest.iterdir() if p.name not in {'__MACOSX', '.DS_Store'}]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest


def _find_dataset_root(input_dir: Path) -> Path:
    """若当前目录无图，尝试进入唯一/首个含图片的子目录。"""
    pairs, _ = collect_image_json_pairs(input_dir)
    if pairs:
        return input_dir
    for child in sorted(p for p in Path(input_dir).iterdir() if p.is_dir()):
        if child.name in {'__MACOSX', '.DS_Store'}:
            continue
        pairs, _ = collect_image_json_pairs(child)
        if pairs:
            return child
    return input_dir


def process_labelme_dir(
    input_dir,
    output_dir,
    mode: str = 'detect',
    model: str = 'yolo11n',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    处理 LabelMe 文件夹，自动扫类别，划分 train/val/test，输出 YOLO 数据集。

    - 无 JSON 的图片、或 shapes 为空的 JSON，都会在 train/val 写出空 txt（负样本）
    - mode: detect | segment | segment_auto
    - model: yolo11n / yolo11s / ...（segment 模式自动加 -seg）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f'输入目录不存在: {input_dir}')
    if mode not in {'detect', 'segment', 'segment_auto'}:
        raise ValueError('mode 只能是 detect / segment / segment_auto')
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio 必须等于 1')

    model = _resolve_model_name(model, mode)
    if model not in YOLO_MODELS:
        raise ValueError(f'不支持的模型: {model}，可选: {", ".join(YOLO_MODELS)}')

    output_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = copy_model_yaml(model, output_dir)
    print(f'已复制模型配置: {model_yaml.name}')

    input_dir = _find_dataset_root(input_dir)
    print(f'数据目录: {input_dir}')

    pairs, orphan_json = collect_image_json_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(
            f'未找到图片（期望 images/+annotations/ 或同目录图片）: {input_dir}'
        )

    no_json = sum(1 for _, jp in pairs if jp is None)
    print(f'样本: 图片 {len(pairs)}，无 JSON 负样本 {no_json}，孤立 JSON {orphan_json}')

    class_list = collect_class_names(input_dir)
    if not class_list:
        raise ValueError('未在标注 JSON 中扫描到任何类别（shapes[].label）')
    print(f'自动识别类别 ({len(class_list)}): {", ".join(class_list)}')

    images_root = output_dir / 'images'
    labels_root = output_dir / 'labels'
    for split in ('train', 'val', 'test'):
        (images_root / split).mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    random.shuffle(pairs)
    splits = _make_splits(pairs, train_ratio, val_ratio, test_ratio)
    print(
        f"划分: train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}（共 {len(pairs)} 张）"
    )

    ok = skip = empty_txt = 0
    for split_name, items in splits.items():
        for img_path, json_path in items:
            stem = _sample_id(img_path)
            data = _load_labelme(json_path)

            img = cv2.imread(str(img_path))
            if img is None:
                skip += 1
                continue

            h, w = img.shape[:2]
            scale = min(MAX_SIZE / w, MAX_SIZE / h, 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if scale < 1.0:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(
                str(images_root / split_name / f'{stem}.jpg'),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if split_name in ('train', 'val'):
                if mode == 'detect':
                    lines = json_to_detect_lines(data, class_list, scale, new_w, new_h)
                else:
                    lines = json_to_segment_lines(data, class_list)
                # shapes 为空或无 JSON：写出空 txt 作为负样本
                (labels_root / split_name / f'{stem}.txt').write_text(
                    '\n'.join(lines) + ('\n' if lines else ''),
                    encoding='utf-8',
                )
                if not lines:
                    empty_txt += 1
            ok += 1

    if mode == 'segment_auto':
        (output_dir / 'classes.txt').write_text(
            '\n'.join(class_list) + '\n', encoding='utf-8'
        )

    data_yaml = _write_data_yaml(output_dir, class_list)
    result = {
        'ok': ok,
        'skip': skip,
        'empty_txt': empty_txt,
        'no_json': no_json,
        'train': len(splits['train']),
        'val': len(splits['val']),
        'test': len(splits['test']),
        'classes': class_list,
        'mode': mode,
        'model': model,
        'path': str(output_dir.resolve()),
        'data_yaml': str(data_yaml.resolve()),
        'model_yaml': str(model_yaml.resolve()),
    }
    print(
        f"完成: 处理 {ok} 张，跳过 {skip} 张，空 txt（负样本）{empty_txt}；"
        f"train={result['train']}, val={result['val']}, test={result['test']}"
    )
    print(f'输出: {result["path"]}')
    print(f'配置: {data_yaml.name}, {model_yaml.name}')
    return result


def process_labelme_zip(
    zip_path,
    output_dir,
    mode: str = 'detect',
    model: str = 'yolo11n',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """解压 LabelMe zip 后走 process_labelme_dir。"""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f'压缩包不存在: {zip_path}')

    tmp = Path(tempfile.mkdtemp(prefix='labelme_'))
    try:
        input_dir = _unzip(zip_path, tmp / 'extract')
        print(f'解压完成: {input_dir}')
        return process_labelme_dir(
            input_dir=input_dir,
            output_dir=output_dir,
            mode=mode,
            model=model,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def process_labelme(
    source,
    output_dir,
    mode: str = 'detect',
    model: str = 'yolo11n',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """自动识别 source 是文件夹还是 zip。"""
    source = Path(source)
    if source.is_dir():
        return process_labelme_dir(
            source, output_dir, mode, model, train_ratio, val_ratio, test_ratio, seed
        )
    if source.is_file() and source.suffix.lower() == '.zip':
        return process_labelme_zip(
            source, output_dir, mode, model, train_ratio, val_ratio, test_ratio, seed
        )
    raise FileNotFoundError(f'请传入 LabelMe 文件夹或 .zip: {source}')


if __name__ == '__main__':
    SOURCE = '/Users/tjw/Desktop/work/workplace/图片/zbf&bjq'  # 文件夹或 zip
    OUTPUT_DIR = '/Users/tjw/Desktop/work/workplace/视觉/zbf_bjq'
    MODE = 'detect'  # detect | segment | segment_auto
    MODEL = 'yolo11n'

    process_labelme(
        source=SOURCE,
        output_dir=OUTPUT_DIR,
        mode=MODE,
        model=MODEL,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
