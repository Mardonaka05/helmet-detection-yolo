#!/usr/bin/env python
"""Run the hard-hat detector on an image, a video, a folder or a live RTSP stream.

    python predict.py --weights weights/best.pt --source sample.jpg
    python predict.py --weights weights/best.pt --source clip.mp4 --save
    python predict.py --weights weights/best.pt --source 0 --show
    python predict.py --weights weights/best.pt \
                      --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101" --show

Prints a per-class detection tally at the end. In sergak-ai a `no_helmet`
detection is what raises an alert, so that count is reported separately.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

VIOLATION_CLASS = "no_helmet"


def parse_args():
    p = argparse.ArgumentParser(description="Hard-hat detection inference.")
    p.add_argument("--weights", default="weights/best.pt", help="path to the .pt checkpoint")
    p.add_argument("--source", required=True,
                   help="image, video, folder, webcam index (e.g. 0) or RTSP/HTTP URL")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.45, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", default="0", help="'0', '0,1' or 'cpu'")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="restrict to class ids, e.g. --classes 1 for no_helmet only")
    p.add_argument("--save", action="store_true", help="write annotated output to runs/predict/")
    p.add_argument("--show", action="store_true", help="open a live preview window")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = unlimited; useful for RTSP smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"weights not found: {args.weights}\n"
            "Model weights are distributed through the Releases page, not committed to git."
        )

    model = YOLO(args.weights)
    counts = Counter()
    frames = 0

    stream = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=args.classes,
        save=args.save,
        show=args.show,
        stream=True,
        verbose=False,
    )

    try:
        for result in stream:
            frames += 1
            if result.boxes is not None:
                for class_id in result.boxes.cls.tolist():
                    counts[model.names[int(class_id)]] += 1
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    print()
    print(f"frames processed : {frames}")
    if counts:
        for name, n in counts.most_common():
            print(f"  {name:<14} {n}")
    else:
        print("  no detections")

    violations = counts.get(VIOLATION_CLASS, 0)
    if violations:
        print(f"\n{violations} {VIOLATION_CLASS} detection(s) — this is the event that "
              f"triggers a Telegram alert in sergak-ai.")

    if args.save:
        print("annotated output written to runs/predict/")


if __name__ == "__main__":
    main()
