import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from rapidfuzz.distance import Levenshtein
from ultralytics import YOLO

from main.recognition import load_model, process_license_plate
from main.YOLO_utils import crop_boxes_from_image

# https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset/data?select=State-wise_OLX
# https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset/data

INDIAN_PATH = "test_datasets/indian_dataset"
POLAND_PATH = "test_datasets/polish_dataset"

YOLO_MODEL_PATH = "models/YOLO/weights/best.pt"
RESULTS_DIR = "test/results/text_recognition"

MAX_IMAGES = 1000
RANDOM_SEED = 42

yolo = YOLO(YOLO_MODEL_PATH)
reader, _ = load_model()


def sample_dataset(data, max_images=MAX_IMAGES, seed=RANDOM_SEED):
    data = list(data)
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:max_images]


def normalize_plate(text):
    return text.replace(" ", "").upper()


def parse_indian_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    obj = root.find("object")
    if obj is None:
        return None

    gt = obj.find("name").text.strip()
    return filename, gt


def load_indian(dataset_path):
    data = []
    for root_dir, _, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".xml"):
                continue
            xml_path = os.path.join(root_dir, file)
            parsed = parse_indian_xml(xml_path)
            if not parsed:
                continue
            img_name, gt = parsed
            img_path = os.path.join(root_dir, img_name)
            if os.path.isfile(img_path):
                data.append((img_path, gt.strip().upper()))
    return data


def load_poland(dataset_path):
    xml_path = os.path.join(dataset_path, "annotations.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for img in root.findall("image"):
        filename = img.attrib["name"]
        box = img.find("box")
        if box is None:
            continue

        attr = box.find("attribute")
        if attr is None:
            continue

        gt = attr.text.strip()
        img_path = os.path.join(dataset_path, "photos", filename)
        data.append((img_path, gt.strip().upper()))

    return data


def cer(gt, pred):
    gt_n = normalize_plate(gt)
    pred_n = normalize_plate(pred)
    if len(gt_n) == 0:
        return 1.0 if len(pred_n) > 0 else 0.0
    return Levenshtein.distance(gt_n, pred_n) / len(gt_n)


def build_same_length_confusion_matrix(pairs):
    confusion_matrix = Counter()
    subset_count = 0

    for gt, pred in pairs:
        gt_n = normalize_plate(gt)
        pred_n = normalize_plate(pred)
        if len(gt_n) == 0 or len(gt_n) != len(pred_n):
            continue

        subset_count += 1
        for g, p in zip(gt_n, pred_n):
            confusion_matrix[(g, p)] += 1

    return confusion_matrix, subset_count


def plot_confusion_matrix(confusion_matrix, path, title):
    chars = sorted({c for pair in confusion_matrix for c in pair})
    idx = {c: i for i, c in enumerate(chars)}
    matrix = np.zeros((len(chars), len(chars)))

    for (g, p), count in confusion_matrix.items():
        matrix[idx[g], idx[p]] = count

    fig, ax = plt.subplots(figsize=(max(8, len(chars) * 0.45), max(6, len(chars) * 0.45)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(chars)))
    ax.set_yticks(range(len(chars)))
    ax.set_xticklabels(chars)
    ax.set_yticklabels(chars)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_error_histogram(pairs, path, title):
    distances = [
        Levenshtein.distance(normalize_plate(gt), normalize_plate(pred))
        for gt, pred in pairs
    ]

    plt.figure(figsize=(8, 5))
    if distances:
        max_err = max(distances)
        bins = np.arange(0, max_err + 2) - 0.5
        plt.hist(distances, bins=bins, edgecolor="black", alpha=0.75)
        plt.xticks(range(0, max_err + 1))
    plt.xlabel("Edit distance (character errors)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return distances


def run_eval(dataset, name="dataset", seed=RANDOM_SEED):
    exact = 0
    cers = []
    detected = 0
    read = 0
    total = 0
    pairs = []

    sampled = sample_dataset(dataset, MAX_IMAGES, seed)

    for i, (img_path, gt) in enumerate(sampled):
        print(f"({i+1}/{len(sampled)})")

        img = cv2.imread(img_path)
        if img is None:
            continue

        crop_pairs = crop_boxes_from_image(yolo, img)
        read_plates = []
        plates_found = False

        for _, plate_imgs in crop_pairs:
            for plate_img in plate_imgs:
                plates_found = True
                text = process_license_plate(image=plate_img, model=reader)
                if text not in ("[BRAK ODCZYTU]", "[PROCESSING ERROR]"):
                    read_plates.append(text)

        pred = read_plates[0] if read_plates else ""
        total += 1
        pairs.append((gt, pred))

        if plates_found:
            detected += 1
        if pred:
            read += 1
        if pred == gt:
            exact += 1

        cers.append(cer(gt, pred))
        print(f"[{name}] GT:{gt} PRED:{pred}")

    acc = exact / max(total, 1)
    mean_cer = float(np.mean(cers)) if cers else 0.0
    detection_rate = detected / max(total, 1)
    plate_read_rate = read / max(total, 1)

    print("\n======================")
    print(name)
    print("======================")
    print("Accuracy:", acc)
    print("CER:", mean_cer)
    print("Detection rate:", detection_rate)
    print("Plate read rate:", plate_read_rate)

    return {
        "dataset": name,
        "accuracy": acc,
        "cer": mean_cer,
        "detection_rate": detection_rate,
        "plate_read_rate": plate_read_rate,
        "n_samples": total,
        "pairs": pairs,
    }


os.makedirs(RESULTS_DIR, exist_ok=True)

indian = load_indian(INDIAN_PATH)
poland = load_poland(POLAND_PATH)

results_india = run_eval(indian, "INDIA", seed=RANDOM_SEED)
results_poland = run_eval(poland, "POLAND", seed=RANDOM_SEED + 1)

for result in (results_india, results_poland):
    dataset_name = result["dataset"]
    confusion_matrix, same_length_count = build_same_length_confusion_matrix(result["pairs"])

    plot_confusion_matrix(
        confusion_matrix,
        os.path.join(RESULTS_DIR, f"confusion_matrix_{dataset_name}.png"),
        f"Character confusion ({dataset_name}, same length only, n={same_length_count})",
    )
    plot_error_histogram(
        result["pairs"],
        os.path.join(RESULTS_DIR, f"error_count_hist_{dataset_name}.png"),
        f"Edit distance per plate ({dataset_name})",
    )

    result["same_length_subset"] = same_length_count

metrics_df = pd.DataFrame([results_india, results_poland])
metrics_df.drop(columns=["pairs"], inplace=True)
metrics_path = os.path.join(RESULTS_DIR, "ocr_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)