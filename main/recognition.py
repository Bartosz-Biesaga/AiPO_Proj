import cv2
import numpy as np
import easyocr
import torch


def load_model(langs=['en'], gpu=None):
    if gpu is None:
        gpu = torch.cuda.is_available()
    reader = easyocr.Reader(langs, gpu=gpu)
    return reader, None


def process_license_plate(image_path=None, image=None, model=None, **kwargs):
    try:
        input_data = image if image is not None else image_path
        if input_data is None:
            return "[BRAK ODCZYTU]"

        if isinstance(input_data, np.ndarray):
            gray = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            input_data = clahe.apply(gray)

        results = model.readtext(input_data, detail=1)

        if not results:
            return "[BRAK ODCZYTU]"

        boxes = []
        for bbox, text, prob in results:
            y_top = min(bbox[0][1], bbox[1][1])
            y_bot = max(bbox[2][1], bbox[3][1])
            x_left = min(bbox[0][0], bbox[3][0])
            x_right = max(bbox[1][0], bbox[2][0])

            h = y_bot - y_top
            cy = (y_top + y_bot) / 2
            cx = (x_left + x_right) / 2

            clean = text.replace(" ", "").upper()
            if clean:
                boxes.append({"text": clean, "h": h, "cy": cy, "cx": cx})

        if not boxes:
            return "[BRAK ODCZYTU]"

        boxes.sort(key=lambda b: b["h"], reverse=True)
        ref_h = boxes[0]["h"]
        ref_cy = boxes[0]["cy"]

        valid_boxes = []
        for b in boxes:
            if b["h"] < 0.60 * ref_h:
                continue
            if abs(b["cy"] - ref_cy) > 0.5 * ref_h:
                continue
            valid_boxes.append(b)

        valid_boxes.sort(key=lambda b: b["cx"])
        best_text = "".join([b["text"] for b in valid_boxes])

        if len(best_text) > 7:
            if best_text.startswith("1") or best_text.startswith("I"):
                best_text = best_text[1:]
            if best_text.endswith("1") or best_text.endswith("I"):
                best_text = best_text[:-1]

        if len(best_text) < 3:
            return "[BRAK ODCZYTU]"

        return best_text

    except Exception as e:
        print(f"Error processing: {e}")
        return "[PROCESSING ERROR]"