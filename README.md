# Hard-Hat Detection with YOLOv8

Custom-trained PPE (hard hat) detection for industrial safety monitoring. **97.5% mAP@50** on a held-out set built from real factory CCTV footage.

<p>
  <img src="https://img.shields.io/badge/mAP%4050-97.5%25-success?style=flat-square">
  <img src="https://img.shields.io/badge/mAP%4050--95-82.2%25-success?style=flat-square">
  <img src="https://img.shields.io/badge/model-YOLOv8n-111F68?style=flat-square">
  <img src="https://img.shields.io/badge/Ultralytics-black?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
</p>

This is the detection model behind [**sergak-ai**](https://github.com/Mardonaka05/sergak-ai), a real-time workplace safety monitoring system.

---

## Why this exists

Off-the-shelf hard-hat models are trained on clean, well-lit stock photography and fall apart on real CCTV: low resolution, motion blur, backlit doorways, workers at 20+ metres, hats in a dozen colours under sodium lighting.

This repository documents the dataset work and training setup that produced a model that actually holds up on footage from Uzbek industrial sites.

---

## Results

Model: `yolov8n` · 640px · batch 16 · 2 classes (`helmet`, `no_helmet`)

| Metric | Value |
|---|---|
| **mAP@50** | **0.975** |
| **mAP@50-95** | **0.822** |
| Precision | 0.956 |
| Recall | 0.928 |

### Run comparison

| Run | Epochs | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| `helmet_exp_gpu` | 10 | 0.956 | 0.928 | **0.975** | **0.822** |
| `helmet_exp55` | 5 | 0.919 | 0.898 | 0.957 | 0.794 |
| `train10` | 5 (early stop) | 0.828 | 0.854 | 0.906 | 0.692 |

The jump from run to run came almost entirely from dataset work, not hyperparameters.

Nothing in these tables is typed by hand. Every figure comes from the Ultralytics output committed in [`reports/`](reports/) — the per-epoch history is in [`reports/results.csv`](reports/results.csv) and the exact hyperparameters in [`reports/args.yaml`](reports/args.yaml).

![Training curves](reports/results.png)

![Confusion matrix](reports/confusion_matrix.png)

![Precision–recall curve](reports/BoxPR_curve.png)

![Sample predictions](reports/val_batch0_pred.jpg)

---

## Dataset

| | |
|---|---|
| Classes | `helmet`, `no_helmet` |
| Sources | Public hard-hat datasets, merged and re-labelled, plus frames pulled from the deployment cameras |
| Labelling | Self-hosted **CVAT** (Docker), exported in YOLO format |
| Merge pipeline | `scripts/` — download, deduplicate, remap class ids, split |
| Augmentation | Mosaic, HSV jitter, random scale, horizontal flip |

**What actually moved the number:** adding frames sampled from the cameras the model would run on. Domain match beat both dataset size and every hyperparameter sweep. A model at 90% on stock photos dropped to the low 70s on site footage until real frames went into training.

Dataset config: [`data/helmet.yaml`](data/helmet.yaml). The images themselves are not committed — point `path` at your own copy.

### Pipeline scripts

| Script | Does |
|---|---|
| `scripts/download_roboflow_datasets.py`, `download_kaggle.py`, `download_huggingface.py`, `download_github.py` | pull source datasets |
| `scripts/convert_voc_to_yolo.py` | VOC XML annotations → YOLO txt |
| `scripts/prepare_sh17.py`, `prepare_new_datasets.py`, `prepare_more_datasets.py` | normalise each source to the two-class scheme |
| `scripts/merge_datasets.py`, `final_merge.py` | deduplicate and merge into one train/val/test split |
| `scripts/validate_dataset.py`, `check_datasets.py`, `stats.py` | label sanity checks and class balance |
| `scripts/visualize.py` | draw labels over images for spot-checking |
| `scripts/test_inference.py`, `live_test.py`, `test_gui.py` | single-image, RTSP and GUI testing |
| `scripts/export_production.py` | export the chosen run's `best.pt` for deployment |

---

## Reproduce

```bash
pip install -r requirements.txt

# train
python train.py --data data/helmet.yaml --model yolov8n.pt \
                --epochs 10 --imgsz 640 --batch 16

# evaluate
python val.py --weights weights/best.pt --data data/helmet.yaml --split test

# run on an image, a video, or a live stream
python predict.py --weights weights/best.pt --source path/to/image.jpg
python predict.py --weights weights/best.pt \
                  --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
```

### Training configuration

These are the actual arguments of the run that produced the numbers above, copied from [`reports/args.yaml`](reports/args.yaml):

```yaml
model:      yolov8n.pt
epochs:     10
imgsz:      640
batch:      16
optimizer:  auto
lr0:        0.01
patience:   100
device:     cpu
seed:       0
```

Two things worth flagging, because the file is in the repository and you will see them:

- The run is named `helmet_exp_gpu` but `device: cpu` — the name is left over from an earlier attempt and the winning run happened to finish on CPU. Ten epochs of `yolov8n` at 640px is cheap enough that it did not matter.
- `train.py` ships with heavier defaults (`yolov8l`, 150 epochs, AdamW, batch 6) aimed at an 8 GB GPU. The shipped model is the lighter `yolov8n` run above, chosen for edge inference speed rather than peak accuracy. Pass the arguments shown here to reproduce it.

---

## Repository layout

```
data/helmet.yaml     dataset config (2 classes, split paths)
scripts/             dataset download, convert, merge, validate, export
train.py             training entrypoint
val.py               evaluation — prints precision, recall, mAP per class
predict.py           inference on image / video / folder / RTSP
reports/             metrics, curves, confusion matrix, sample predictions
requirements.txt     pinned dependencies
weights/             released model weights (see Releases)
```

Weights are distributed through [Releases](../../releases) rather than committed, to keep the repository small.

---

## Limitations

- Trained on 2 classes only — it does not distinguish hat colour or detect other PPE (vests, goggles, gloves)
- Performance degrades below roughly 100px of person height in frame
- Night / IR footage was under-represented in training; expect lower recall there
- `yolov8n` was chosen for edge deployment speed; a larger backbone would likely gain a few points of mAP@50-95
- The reported split is the one shipped with the merged dataset; it is not a public benchmark, so these numbers are not directly comparable to published hard-hat results

---

## License

MIT — see [LICENSE](LICENSE). Weights are released for research and evaluation use.

## Author

**Mardonbek Sulaymonqulov** — AI / Computer Vision Engineer
[GitHub](https://github.com/Mardonaka05) · mardonbeksulaymonqulov156@gmail.com
