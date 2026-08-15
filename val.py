#!/usr/bin/env python
"""Evaluate a trained hard-hat detector on a dataset split.

Reproduces the numbers reported in the README:

    python val.py --weights weights/best.pt --data data/helmet.yaml --split test

Ultralytics writes the plots (PR curve, confusion matrix, F1 curve) into
runs/val/<name>/ so they can be compared against the ones committed in reports/.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a YOLO hard-hat detector.")
    p.add_argument("--weights", default="weights/best.pt", help="path to the .pt checkpoint")
    p.add_argument("--data", default="data/helmet.yaml", help="dataset YAML")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.001,
                   help="confidence floor for mAP (keep low; this is not the deployment threshold)")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--device", default="0", help="'0', '0,1' or 'cpu'")
    p.add_argument("--project", default="runs/val")
    p.add_argument("--name", default="exp")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"weights not found: {args.weights}\n"
            "Model weights are distributed through the Releases page, not committed to git."
        )
    if not Path(args.data).exists():
        raise SystemExit(f"dataset YAML not found: {args.data}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
    )

    box = metrics.box
    line = "-" * 46

    print()
    print(line)
    print(f"{'split':<16}{args.split}")
    print(f"{'images':<16}{getattr(metrics, 'seen', 'n/a')}")
    print(f"{'precision':<16}{box.mp:.4f}")
    print(f"{'recall':<16}{box.mr:.4f}")
    print(f"{'mAP@50':<16}{box.map50:.4f}")
    print(f"{'mAP@50-95':<16}{box.map:.4f}")
    print(line)

    for i, class_id in enumerate(box.ap_class_index):
        name = model.names[int(class_id)]
        print(f"  {name:<14} mAP@50={box.ap50[i]:.4f}   mAP@50-95={box.ap[i]:.4f}")
    print(line)
    print(f"plots written to {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
