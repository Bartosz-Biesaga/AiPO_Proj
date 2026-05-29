import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO


# Przywrócona oryginalna sygnatura funkcji (GUI już nie będzie wyrzucać błędu!)
def crop_boxes_from_image(yolo, image, license_plate_car_ioa=0.85, confidence=0.25, iou=0.7, save_prediction=False):
    # Wykorzystanie parametrów z GUI
    result = yolo(image, conf=confidence, iou=iou, verbose=False)[0]

    # Przywrócona logika zapisu predykcji dla GUI
    if save_prediction is True:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(dir_path, "..", "results", "prediction.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(filename=save_path)

    cars = []
    plates = []

    # --- NOWA, BEZBŁĘDNA LOGIKA CIĘCIA I PADDINGU ---
    for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)

        if int(cls) == 0:  # Auto
            cars.append({"img": image[y1:y2, x1:x2], "box": [x1, y1, x2, y2]})
        else:  # Tablica
            w, h = x2 - x1, y2 - y1

            # Bezpieczny margines (padding), żeby nie ucinać wierzchołków
            pad_x = int(w * 0.06)
            pad_top = int(h * 0.12)
            pad_bot = int(h * 0.05)

            ny1 = max(0, y1 - pad_top)
            ny2 = min(image.shape[0], y2 + pad_bot)
            nx1 = max(0, x1 - pad_x)
            nx2 = min(image.shape[1], x2 + pad_x)

            plates.append({"img": image[ny1:ny2, nx1:nx2], "box": [x1, y1, x2, y2]})

    # --- NOWA LOGIKA PAROWANIA TABLIC Z AUTAMI ---
    pairs = []
    assigned_plates = set()

    for car in cars:
        car_plates = []
        cx1, cy1, cx2, cy2 = car["box"]

        for i, plate in enumerate(plates):
            px1, py1, px2, py2 = plate["box"]
            plate_center_x = (px1 + px2) / 2
            plate_center_y = (py1 + py2) / 2

            # Sprawdzamy czy środek tablicy jest fizycznie wewnątrz prostokąta auta
            if cx1 <= plate_center_x <= cx2 and cy1 <= plate_center_y <= cy2:
                car_plates.append(plate["img"])
                assigned_plates.add(i)

        pairs.append([car["img"], car_plates])

    # Obsługa "sierot" (tablic bez aut)
    orphan_plates = [p["img"] for i, p in enumerate(plates) if i not in assigned_plates]
    if orphan_plates:
        pairs.append([None, orphan_plates])

    return pairs