import os
import shutil
import cv2
from ultralytics import YOLO

from main.recognition import load_model, process_license_plate
from main.YOLO_utils import crop_boxes_from_image


def run_tests_and_sort(source_dir, correct_dir, incorrect_dir):
    print("Ładowanie modeli (YOLO & EasyOCR)...")
    yolo_model = YOLO("models/YOLO/weights/best.pt")
    reader, _ = load_model()

    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    print(f"\nRozpoczynam testy z folderu: {source_dir}")
    print("=" * 60)

    total_images = 0
    correct_reads = 0
    yolo_errors = 0
    ocr_errors = 0

    for image_name in os.listdir(source_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        total_images += 1
        base_name = os.path.splitext(image_name)[0].upper()
        expected_plates = base_name.split('_')
        image_path = os.path.join(source_dir, image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        print(f"Plik: {image_name}")
        print(f"Oczekiwano: {', '.join(expected_plates)}")

        # KROK 1: Detekcja (YOLO)
        pairs = crop_boxes_from_image(yolo_model, image)

        plates_found = False
        read_plates = set()

        # KROK 2: Odczyt (EasyOCR z heurystyką)
        for car_img, plate_imgs in pairs:
            for plate_img in plate_imgs:
                plates_found = True
                read_text = process_license_plate(image=plate_img, model=reader)

                if read_text != "[BRAK ODCZYTU]":
                    read_plates.add(read_text)

        print(f"Odczytano:  {', '.join(read_plates) if read_plates else '[BRAK ODCZYTU]'}")

        # KROK 3: Walidacja i sortowanie
        read_success = (len(read_plates) == len(expected_plates) 
                        and all(expected in read_plates for expected in expected_plates))

        if read_success:
            print("-> WYNIK: ZGODNY (/correct)")
            shutil.move(image_path, os.path.join(correct_dir, image_name))
            correct_reads += 1
        else:
            if not plates_found:
                print("-> WYNIK: BŁĄD YOLO (/incorrect)")
                yolo_errors += 1
            else:
                print("-> WYNIK: BŁĄD ODCZYTU (/incorrect)")
                ocr_errors += 1
            shutil.move(image_path, os.path.join(incorrect_dir, image_name))

        print("-" * 60)

    print("=" * 60)
    print("PODSUMOWANIE TESTU I SORTOWANIA:")
    print(f"Zdjęć:      {total_images}")
    print(f"Zgodne:       {correct_reads}")
    print(f"Błędy YOLO:   {yolo_errors}")
    print(f"Błędy OCR:    {ocr_errors}")
    print(f"Błędne łącznie: {total_images - correct_reads}")
    if total_images > 0:
        print(f"SKUTECZNOŚĆ: {(correct_reads / total_images) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    src = "test_data/"
    corr = "test_data/correct"
    incorr = "test_data/incorrect"

    # Automatyczny powrót zdjęć na miejsce (jeśli testujesz wielokrotnie)
    for folder in [corr, incorr]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                shutil.move(os.path.join(folder, file), os.path.join(src, file))

    run_tests_and_sort(src, corr, incorr)