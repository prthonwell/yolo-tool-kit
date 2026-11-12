# yolo-tool-kit

YOLO 常用小工具箱（自 toolScrip 迁入）。

## 工具一览

| 脚本 | 说明 |
|------|------|
| `classify.py` | 普通分类：按预测类别整理图片 |
| `classify_draw_box.py` | 分类画框：检测/分类并画框输出 |
| `prerequsite_process_for_yolo.py` | LabelMe JSON → YOLO txt（含 data.yaml） |
| `inspect_pt.py` | 检查 `.pt` 结构与元信息 |
| `rtsp_frame_capture.py` | RTSP 取帧落盘 |

## 环境

```bash
pip install -r requirements.txt
```

## 注意

- `.gitignore` 已排除 `*.pt` / `*.zip` / `*.rar`，权重与压缩包不要入库。
