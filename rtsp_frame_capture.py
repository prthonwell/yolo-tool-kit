"""RTSP 视频流取帧脚本。

启动后连接配置的视频链接，持续读流，每隔固定时间（默认 5 秒）
把最新一帧存为 jpg 到输出目录，断流自动重连，Ctrl+C 停止。

注意：FFMPEG 对 RTSP 有内部缓冲，如果隔 5 秒才去 read 一次，
拿到的往往是几秒前的旧帧。所以这里按流的帧率持续读、只按间隔
落盘最新帧，保证画面是实时的。
"""
from __future__ import annotations

import os

# RTSP 强制走 TCP，避免 UDP 丢包花屏；必须在 cv2 初始化前设置
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2

RTSP_URL = 'rtsp://jmhk.video.crcgas.com:554/openUrl/vsig3Dw51rW95455ec50f5f46cfb7fc2'
OUTPUT_DIR = 'rtsp_frames'
INTERVAL_SECONDS = 5
RECONNECT_DELAY = 5      # 断流/连接失败后的重连等待秒数
MAX_READ_FAILS = 10      # 连续读帧失败多少次判定为断流
JPEG_QUALITY = 95


def run(
    rtsp_url: str = RTSP_URL,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    interval: float = INTERVAL_SECONDS,
) -> None:
    """连接 RTSP 流，每隔 interval 秒保存一帧到 output_dir。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap: Optional[cv2.VideoCapture] = None
    saved = 0
    last_save = 0.0  # 0 表示第一帧立即落盘
    read_fails = 0

    print(f'输出目录: {output_dir.resolve()}')
    try:
        while True:
            if cap is None:
                print(f'连接视频流: {rtsp_url}')
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print(f'连接失败，{RECONNECT_DELAY}s 后重试...')
                    cap.release()
                    cap = None
                    time.sleep(RECONNECT_DELAY)
                    continue
                print('连接成功，开始取帧')
                read_fails = 0

            ok, frame = cap.read()
            if not ok:
                read_fails += 1
                if read_fails >= MAX_READ_FAILS:
                    print(f'连续 {read_fails} 次读帧失败，判定断流，准备重连')
                    cap.release()
                    cap = None
                    read_fails = 0
                    time.sleep(RECONNECT_DELAY)
                continue
            read_fails = 0

            now = time.monotonic()
            if now - last_save >= interval:
                name = f"frame_{datetime.now():%Y%m%d_%H%M%S}_{saved + 1:04d}.jpg"
                path = output_dir / name
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                saved += 1
                last_save = now
                print(f'[{saved}] {path.name}')
    except KeyboardInterrupt:
        print('\n手动停止')
    finally:
        if cap is not None:
            cap.release()
        print(f'共保存 {saved} 帧')


if __name__ == '__main__':
    # RTSP_URL = 'rtsp://jmhk.video.crcgas.com:554/openUrl/vsig3Dw51rW95455ec50f5f46cfb7fc2'
    # OUTPUT_DIR = '/Users/tjw/Desktop/rtsp_frames'
    # INTERVAL_SECONDS = 5
    run()
