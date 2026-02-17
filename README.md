# YOLOX with Distance Estimation Head

This repository extends [YOLOX](https://arxiv.org/abs/2107.08430) (an anchor-free object detector) with a **distance regression head** for estimating the real-world distance to detected objects. The primary use case demonstrated here is stop sign distance estimation for autonomous driving applications.

The distance head is a lightweight addition to the standard YOLOX architecture — it shares the backbone and FPN features and adds a single regression branch that predicts distance in meters for each detected object.

---

## Installation

**Prerequisites:** Python ≥ 3.7, PyTorch ≥ 1.7

**Step 1.** Clone the repository.

```bash
git clone <your-repo-url>
```

**Step 2.** Install YOLOX and its dependencies.

```bash
pip install -v -e .
```

This installs the package in development mode so that local changes take effect immediately. Core dependencies (listed in `requirements.txt`) include numpy, torch, torchvision, opencv-python, loguru, pycocotools, and onnx.

**Step 3.** Download pretrained YOLOX weights.

The distance training script initializes from a standard YOLOX checkpoint. Download YOLOX-s weights (or another variant) from the table below:

| Model | Params | mAP (val) | Weights |
|-------|--------|-----------|---------|
| YOLOX-s | 9.0M | 40.5 | [download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| YOLOX-m | 25.3M | 46.9 | [download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
| YOLOX-l | 54.2M | 49.7 | [download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| YOLOX-x | 99.1M | 51.1 | [download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |

```bash
mkdir -p weights
wget -O weights/yolox_s.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

---

## Custom Dataset Format

The distance training pipeline uses a **CSV annotation file** paired with an **image directory**. No COCO-format JSON is required.

### CSV Structure

The CSV must contain the following columns (with a header row):

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Relative path from `--img-dir` to the image file |
| `x_min` | float | Left edge of the bounding box (pixels, original image coordinates) |
| `y_min` | float | Top edge of the bounding box (pixels) |
| `x_max` | float | Right edge of the bounding box (pixels) |
| `y_max` | float | Bottom edge of the bounding box (pixels) |
| `class` | string | Class label (e.g., `stop_sign`) — currently all boxes are mapped to class 0 internally |
| `distance_m` | float | Ground-truth distance to the object in meters |

### Example CSV

```csv
image_path,x_min,y_min,x_max,y_max,class,distance_m
images/img001.jpg,69,74,1200,1205,stop_sign,7.1
images/img002.jpg,236,186,1168,1244,stop_sign,7.4
images/img003.jpg,224,38,836,711,stop_sign,9.7
images/img004.jpg,594,95,714,230,stop_sign,35.0
images/img005.jpg,189,20,457,289,stop_sign,19.2
```

### Directory Layout

```
data/
└── stopsigns/
    ├── annotations.csv
    └── images/
        ├── img001.jpg
        ├── img002.jpg
        └── ...
```

The `image_path` column values are joined with the `--img-dir` argument to locate each file. For example, if `--img-dir data/stopsigns/` and `image_path` is `images/img001.jpg`, the loader reads `data/stopsigns/images/img001.jpg`.

### Multiple Annotations Per Image

A single image can have multiple rows in the CSV (one per bounding box). The dataset loader groups annotations by `image_path` automatically.

### An included example dataset

A small example annotation file is provided at `yolox/data/stopsigns/annotations.csv` with 10 stop sign images for reference.

---

## Training the Distance Head

The training script (`tools/train_distance.py`) **only trains the distance regression branch**. The backbone (PAFPN) and all detection head branches (classification, regression, objectness) are permanently frozen throughout the entire training run. This guarantees that pretrained detection accuracy is fully preserved — only the distance prediction is learned.
Frozen modules are also set to `eval()` mode so that BatchNorm running statistics and dropout behavior remain unchanged.

### Basic Training Command

```bash
python tools/train_distance.py \
    --ckpt weights/yolox_s.pth \
    --csv data/stopsigns/annotations.csv \
    --img-dir data/stopsigns/ \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-3 \
    --output-dir runs/distance_training
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | *required* | Path to pretrained YOLOX checkpoint |
| `--csv` | *required* | Path to CSV annotation file |
| `--img-dir` | *required* | Root directory for images |
| `--epochs` | 100 | Total training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--img-size` | 640 | Input image size (images are letterboxed to this square) |
| `--output-dir` | `runs/distance_training` | Directory for checkpoints |
| `--unfreeze-epoch` | 50 | Epoch at which to unfreeze backbone for fine-tuning |
| `--device` | auto | `cuda` or `cpu` (auto-detects GPU) |

### Outputs

Training produces two checkpoint files in `--output-dir`:

- `best_ckpt.pth` — checkpoint with the lowest training loss
- `latest_ckpt.pth` — checkpoint from the most recent epoch

---

## Inference with Distance Estimation

Use `tools/infer_distance.py` to run detection + distance estimation on images.

### Basic Inference Command

```bash
python tools/infer_distance.py \
    --ckpt runs/distance_training/best_ckpt.pth \
    --img test_images/img001.jpg \
    --output-dir runs/results
```

To run on an entire directory of images:

```bash
python tools/infer_distance.py \
    --ckpt runs/distance_training/best_ckpt.pth \
    --img test_images/ \
    --output-dir runs/results
```

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | *required* | Path to trained checkpoint |
| `--img` | *required* | Image file or directory of images |
| `--output-dir` | `runs/results` | Directory for annotated output images |
| `--conf` | 0.3 | Confidence threshold |
| `--nms` | 0.45 | NMS IoU threshold |
| `--img-size` | 640 | Input image size |
| `--device` | auto | `cuda` or `cpu` |
| `--stop-sign-only` | off | Only display stop sign detections |

### Output

Each input image is saved to `--output-dir` with bounding boxes drawn. Stop sign detections include the estimated distance in the label (e.g., `stop_sign 0.95 | 12.3m`). Other COCO classes are detected with standard bounding boxes but without distance labels.

---

## Standard YOLOX Usage

The original YOLOX detection pipeline is fully preserved. You can run standard detection, training, evaluation, and export as described below.

### Demo (Detection Only)

```bash
# Image
python tools/demo.py image -n yolox-s -c weights/yolox_s.pth \
    --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 \
    --save_result --device cpu

# Video
python tools/demo.py video -n yolox-s -c weights/yolox_s.pth \
    --path /path/to/video --conf 0.25 --nms 0.45 --tsize 640 \
    --save_result --device gpu
```

### Training on COCO

```bash
# Link your COCO dataset
ln -s /path/to/COCO ./datasets/COCO

# Train YOLOX-s on 8 GPUs
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o
```

### Evaluation

```bash
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 --fp16 --fuse
```

### Export to ONNX

```bash
python tools/export_onnx.py -n yolox-s -c weights/yolox_s.pth --output-name yolox_s.onnx
```

---

## Project Structure

```
YOLOX/
├── tools/
│   ├── train_distance.py      # Train distance regression head
│   ├── infer_distance.py      # Inference with distance estimation
│   ├── train.py               # Standard YOLOX training
│   ├── demo.py                # Standard detection demo
│   ├── eval.py                # COCO evaluation
│   ├── export_onnx.py         # ONNX export
│   └── export_torchscript.py  # TorchScript export
├── yolox/
│   ├── models/                # Network architecture (backbone, FPN, head)
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── stopsign_distance.py  # Distance dataset loader
│   │   │   └── ...
│   │   └── stopsigns/
│   │       └── annotations.csv       # Example annotations
│   ├── core/                  # Training loop, launcher
│   └── utils/                 # Checkpointing, visualization, etc.
├── exps/                      # Experiment config files
├── weights/                   # Store pretrained weights here
└── requirements.txt
```

## Citation
