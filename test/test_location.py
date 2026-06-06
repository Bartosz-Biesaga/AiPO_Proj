import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# CONFIG

# https://www.kaggle.com/datasets/mgmitesh/automatic-license-plate-recognition-alpr-dataset

YOLO_MODEL_PATH = "models/YOLO/weights/best.pt"

IMAGES_DIR = "test_datasets/ufpr/test/images"
LABELS_DIR = "test_datasets/ufpr/test/labels"

# IMAGES_DIR = "ufpr/test/images"
# LABELS_DIR = "ufpr/test/labels"

RESULTS_DIR = "test/results/location"


MAX_IMAGES = 300

# klasa tablicy w wytrenowanym modelu
LICENSE_PLATE_CLASS = 1

IOU_THRESHOLD = 0.7

model = YOLO(YOLO_MODEL_PATH)


def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h

    return np.array([x1, y1, x2, y2])


def load_gt_box(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()

    if len(parts) != 5:
        return None

    cls, x, y, w, h = map(float, parts)

    return yolo_to_xyxy(x, y, w, h, img_w, img_h)


def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    intersection = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# metryki

TP = 0
FP = 0
FN = 0

ious = []
confidences = []

# debug
debug_samples = []


img_names = sorted(os.listdir(IMAGES_DIR))[:MAX_IMAGES]

for img_name in img_names:

    img_path = os.path.join(IMAGES_DIR, img_name)

    label_path = os.path.join(
        LABELS_DIR,
        os.path.splitext(img_name)[0] + ".txt"
    )

    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    gt_box = load_gt_box(label_path, w, h)

    if gt_box is None:
        continue

    # przedykcja

    result = model.predict(
        image,
        verbose=False
    )[0]

    best_box = None
    best_conf = 0.0

    if result.boxes is not None:

        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):

            if int(cls) != LICENSE_PLATE_CLASS:
                continue

            if conf > best_conf:
                best_conf = float(conf)
                best_box = box

    if best_box is None:

        FN += 1

        debug_samples.append({
            "image": img_name,
            "iou": 0,
            "status": "FN"
        })

        continue

    current_iou = iou(gt_box, best_box)

    ious.append(current_iou)
    confidences.append(best_conf)

    if current_iou >= IOU_THRESHOLD:

        TP += 1

        debug_samples.append({
            "image": img_name,
            "iou": current_iou,
            "status": "TP"
        })

    else:

        FP += 1

        debug_samples.append({
            "image": img_name,
            "iou": current_iou,
            "status": "FP"
        })

    print(
        f"{img_name:30s} "
        f"IoU={current_iou:.3f} "
        f"Conf={best_conf:.3f}"
    )


# metryki

precision = TP / (TP + FP + 1e-9)

recall = TP / (TP + FN + 1e-9)

f1 = (
    2 * precision * recall
    / (precision + recall + 1e-9)
)

mean_iou = np.mean(ious) if len(ious) else 0


# wyniki

print("\n==============================")
print(" DETECTION RESULTS")
print("==============================")

print(f"Images:      {len(debug_samples)}")
print(f"TP:          {TP}")
print(f"FP:          {FP}")
print(f"FN:          {FN}")

print()
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"Mean IoU:    {mean_iou:.4f}")


# wykresy

plt.figure(figsize=(8, 5))
plt.hist(ious, bins=20)
plt.title("IoU Distribution")
plt.xlabel("IoU")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "iou_hist.png"))

plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=20)
plt.title("Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confidence_hist.png"))

plt.figure(figsize=(6, 5))
plt.bar(
    ["Precision", "Recall", "F1"],
    [precision, recall, f1]
)
plt.ylim(0, 1)
plt.title("Detection Metrics")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "metrics.png"))

plt.figure(figsize=(7, 5))
plt.scatter(confidences, ious, alpha=0.6)
plt.xlabel("Confidence")
plt.ylabel("IoU")
plt.title("IoU vs Confidence")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "iou_vs_confidence.png"))

sorted_ious = np.sort(ious)

plt.figure(figsize=(7, 5))
plt.plot(
    sorted_ious,
    np.arange(1, len(sorted_ious)+1) / len(sorted_ious)
)
plt.xlabel("IoU")
plt.ylabel("Cumulative fraction")
plt.title("Cumulative IoU Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "iou_cdf.png"))


print("\nSaved plots:")
print("- iou_hist.png")
print("- confidence_hist.png")
print("- metrics.png")


# IoU pole przeciecia przez pole sumy
