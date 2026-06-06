import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from main.recognition import load_model, process_license_plate
from main.YOLO_utils import crop_boxes_from_image
import random


# https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset/data?select=State-wise_OLX
# https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset/data



# konfiguracja

INDIAN_PATH = "test_datasets/indian_dataset"
POLAND_PATH = "test_datasets/polish_dataset"

YOLO_MODEL_PATH = "models/YOLO/weights/best.pt"

RESULTS_DIR = "test/results/text_recognition"

MAX_IMAGES = 50

RANDOM_SEED = 42

def sample_dataset(data, max_images=MAX_IMAGES, seed=RANDOM_SEED):
    data = list(data)
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:max_images]

yolo = YOLO(YOLO_MODEL_PATH)
reader, _ = load_model()


# XML parser

def parse_indian_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    obj = root.find("object")

    if obj is None:
        return None

    gt = obj.find("name").text.strip()
    return filename, gt


# ładowanie datasetu

def load_indian(dataset_path):
    data = []
    for file in os.listdir(dataset_path):
        if file.endswith(".xml"):
            parsed = parse_indian_xml(os.path.join(dataset_path, file))
            if parsed:
                img, gt = parsed
                data.append((os.path.join(dataset_path, img), gt.strip().upper()))
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


# metryki

def cer(gt, pred):
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0

    dp = np.zeros((len(gt)+1, len(pred)+1))

    for i in range(len(gt)+1):
        dp[i][0] = i
    for j in range(len(pred)+1):
        dp[0][j] = j

    for i in range(1, len(gt)+1):
        for j in range(1, len(pred)+1):
            cost = 0 if gt[i-1] == pred[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[-1][-1] / max(len(gt), 1)


# OCR 

def run_eval(dataset, name="dataset", seed=RANDOM_SEED):
    exact = 0
    cers = []
    total = 0
    char_conf = Counter()

    sampled = sample_dataset(dataset, MAX_IMAGES, seed)

    for img_path, gt in sampled:

        img = cv2.imread(img_path)
        if img is None:
            continue

        pairs = crop_boxes_from_image(yolo, img)
        read_plates = []
        for _, plate_imgs in pairs:
            for plate_img in plate_imgs:
                text = process_license_plate(image=plate_img, model=reader)
                if text not in ("[BRAK ODCZYTU]", "[PROCESSING ERROR]"):
                    read_plates.append(text)
        pred = read_plates[0] if read_plates else ""

        total += 1

        if pred == gt:
            exact += 1

        cers.append(cer(gt, pred))

        for g, p in zip(gt, pred):
            if g != p:
                char_conf[(g, p)] += 1

        print(f"[{name}] GT:{gt} PRED:{pred}")

    acc = exact / max(total, 1)
    mean_cer = np.mean(cers)

    print("\n======================")
    print(name)
    print("======================")
    print("Accuracy:", acc)
    print("CER:", mean_cer)

    return acc, mean_cer, cers, char_conf


indian = load_indian(INDIAN_PATH)
poland = load_poland(POLAND_PATH)

acc_i, cer_i, _, _ = run_eval(indian, "INDIA")
acc_p, cer_p, _, _ = run_eval(poland, "POLAND")


# wykresy

plt.figure()
plt.bar(["India", "Poland"], [acc_i, acc_p])
plt.title("Accuracy comparison")
plt.ylim(0, 1)
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_compare.png"))

plt.figure()
plt.bar(["India", "Poland"], [cer_i, cer_p])
plt.title("CER comparison")
plt.savefig(os.path.join(RESULTS_DIR, "cer_compare.png"))

print("\nSaved plots:")
print("- accuracy_compare.png")
print("- cer_compare.png")