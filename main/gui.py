import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from YOLO_utils import crop_boxes_from_image
from recognition import load_model, process_license_plate
import shutil


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("System Rozpoznawania Tablic Rejestracyjnych")
        self.root.geometry("1100x650")

        # Używamy nowoczesnego motywu, jeśli jest dostępny
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))

        # Inicjalizacja modeli
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(dir_path, "..", "models", "YOLO", "weights", "best.pt")
        self.yolo = YOLO(weights_path)
        self.recognition_model, _ = load_model()

        # Ścieżki
        self.results_dir = os.path.join(dir_path, "..", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.image_path = None
        self.detections = []
        self.current_car_index = 0

        # GŁÓWNY LAYOUT (Panel lewy i prawy)
        self.left_panel = ttk.Frame(self.root, padding=15, width=250, relief="flat")
        self.left_panel.pack(side="left", fill="y")

        self.right_panel = ttk.Frame(self.root, padding=15)
        self.right_panel.pack(side="right", fill="both", expand=True)

        # PANLE LEWY: Przyciski sterujące
        ttk.Label(self.left_panel, text="Narzędzia", style="Header.TLabel").pack(pady=(0, 15))

        self.select_button = ttk.Button(self.left_panel, text="Wczytaj Zdjęcie", command=self.select_image)
        self.select_button.pack(fill="x", pady=5)

        self.detect_button = ttk.Button(self.left_panel, text="Wykryj Pojazdy", command=self.detect_objects,
                                        state=tk.DISABLED)
        self.detect_button.pack(fill="x", pady=5)

        self.process_button = ttk.Button(self.left_panel, text="Rozpoznaj Tablicę", command=self.process_plate,
                                         state=tk.DISABLED)
        self.process_button.pack(fill="x", pady=5)

        self.save_button = ttk.Button(self.left_panel, text="Zapisz Wyniki", command=self.save_results,
                                      state=tk.DISABLED)
        self.save_button.pack(fill="x", pady=5)

        ttk.Separator(self.left_panel, orient="horizontal").pack(fill="x", pady=20)

        # Nawigacja między pojazdami
        ttk.Label(self.left_panel, text="Wykryte pojazdy:", style="TLabel").pack()
        self.nav_frame = ttk.Frame(self.left_panel)
        self.nav_frame.pack(fill="x", pady=5)

        self.prev_button = ttk.Button(self.nav_frame, text="◀ Poprz.", command=self.prev_car, state=tk.DISABLED,
                                      width=8)
        self.prev_button.pack(side="left", padx=(0, 5), expand=True)

        self.next_button = ttk.Button(self.nav_frame, text="Nast. ▶", command=self.next_car, state=tk.DISABLED, width=8)
        self.next_button.pack(side="right", padx=(5, 0), expand=True)

        # Wynik tekstowy
        ttk.Separator(self.left_panel, orient="horizontal").pack(fill="x", pady=20)
        ttk.Label(self.left_panel, text="Odczytany numer:", style="Header.TLabel").pack(pady=5)

        self.plate_text_box = tk.Text(self.left_panel, height=2, width=15, font=("Courier", 16, "bold"),
                                      state=tk.DISABLED, bg="#ffffff")
        self.plate_text_box.pack(fill="x", pady=5)

        # PANEL PRAWY: Obrazy
        # Wiersz górny: Oryginał i Detekcja YOLO
        self.top_images_frame = ttk.Frame(self.right_panel)
        self.top_images_frame.pack(fill="both", expand=True)

        self.orig_frame = ttk.Frame(self.top_images_frame)
        self.orig_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(self.orig_frame, text="Oryginał", style="Header.TLabel").pack()
        self.original_label = ttk.Label(self.orig_frame, background="#e0e0e0", anchor="center")
        self.original_label.pack(fill="both", expand=True, pady=5)

        self.annotated_frame = ttk.Frame(self.top_images_frame)
        self.annotated_frame.pack(side="right", fill="both", expand=True, padx=5)
        ttk.Label(self.annotated_frame, text="Detekcja YOLO", style="Header.TLabel").pack()
        self.annotated_label = ttk.Label(self.annotated_frame, background="#e0e0e0", anchor="center")
        self.annotated_label.pack(fill="both", expand=True, pady=5)

        # Wiersz dolny: Zbliżenie na samą tablicę (do sprawdzenia OCR)
        self.bottom_image_frame = ttk.Frame(self.right_panel)
        self.bottom_image_frame.pack(fill="both", expand=True, pady=(15, 0))
        ttk.Label(self.bottom_image_frame, text="Powiększenie Wyciętej Tablicy", style="Header.TLabel").pack()
        self.cropped_label = ttk.Label(self.bottom_image_frame, background="#e0e0e0", anchor="center")
        self.cropped_label.pack(fill="both", expand=True, pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz", filetypes=[("Pliki graficzne", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.detections = []
            self.current_car_index = 0
            self.reset_display()

            pil_img = Image.open(file_path)
            pil_img.thumbnail((450, 350))
            self.original_photo = ImageTk.PhotoImage(pil_img)
            self.original_label.config(image=self.original_photo)
            self.original_label.image = self.original_photo

            self.detect_button.config(state=tk.NORMAL)

    def reset_display(self):
        self.annotated_label.config(image="")
        self.cropped_label.config(image="")
        self.plate_text_box.config(state=tk.NORMAL)
        self.plate_text_box.delete("1.0", tk.END)
        self.plate_text_box.config(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)

    def detect_objects(self):
        if not self.image_path:
            return

        cv_img = cv2.imread(self.image_path)
        if cv_img is None:
            messagebox.showerror("Błąd", "Nie można odczytać obrazu.")
            return

        try:
            self.detections = crop_boxes_from_image(self.yolo, cv_img, save_prediction=True)
            self.detections = [d for d in self.detections if d[1]]

            if not self.detections:
                messagebox.showinfo("Informacja", "Nie wykryto aut z tablicami rejestracyjnymi.")
                return

            annotated_path = os.path.join(self.results_dir, "prediction.jpg")
            if os.path.exists(annotated_path):
                pil_annotated = Image.open(annotated_path)
                pil_annotated.thumbnail((450, 350))
                self.annotated_photo = ImageTk.PhotoImage(pil_annotated)
                self.annotated_label.config(image=self.annotated_photo)
                self.annotated_label.image = self.annotated_photo

            self.current_car_index = 0
            self.show_current_car()

            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            self.process_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Błąd", f"Detekcja nie powiodła się: {e}")

    def show_current_car(self):
        if not self.detections or self.current_car_index >= len(self.detections):
            return

        self.plate_text_box.config(state=tk.NORMAL)
        self.plate_text_box.delete("1.0", tk.END)
        self.plate_text_box.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)

        car, plates = self.detections[self.current_car_index]

        # Wyświetlamy samą tablicę zamiast całego auta, żeby można było zweryfikować jakość wycięcia
        if plates:
            plate_img_cv = plates[0]
            plate_img = Image.fromarray(cv2.cvtColor(plate_img_cv, cv2.COLOR_BGR2RGB))
            # Powiększamy proporcjonalnie tablicę dla lepszej widoczności w GUI
            width, height = plate_img.size
            new_width = 350
            new_height = int((new_width / width) * height)
            plate_img = plate_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.cropped_photo = ImageTk.PhotoImage(plate_img)
            self.cropped_label.config(image=self.cropped_photo)
            self.cropped_label.image = self.cropped_photo

    def prev_car(self):
        if self.current_car_index > 0:
            self.current_car_index -= 1
            self.show_current_car()

    def next_car(self):
        if self.current_car_index < len(self.detections) - 1:
            self.current_car_index += 1
            self.show_current_car()

    def process_plate(self):
        if not self.detections or self.current_car_index >= len(self.detections):
            return

        car, plates = self.detections[self.current_car_index]
        if plates:
            plate = plates[0]
            try:
                # Usunięto argument 'preprocess', ponieważ EasyOCR go nie potrzebuje
                text = process_license_plate(image=plate, model=self.recognition_model)

                self.plate_text_box.config(state=tk.NORMAL)
                self.plate_text_box.delete("1.0", tk.END)
                # Wyśrodkowanie tekstu
                self.plate_text_box.tag_configure("center", justify='center')
                self.plate_text_box.insert(tk.END, text, "center")
                self.plate_text_box.config(state=tk.DISABLED)

                self.save_button.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Błąd", f"Przetwarzanie tablicy nie powiodło się: {e}")

    def save_results(self):
        if not self.detections or self.current_car_index >= len(self.detections):
            return

        # 1. Pobranie nazwy oryginalnego pliku (np. "auto1" z "C:/obrazy/auto1.jpg")
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]

        # 2. Utworzenie dedykowanego podfolderu dla tego zdjęcia w katalogu results
        save_dir = os.path.join(self.results_dir, base_filename)
        os.makedirs(save_dir, exist_ok=True)

        car, plates = self.detections[self.current_car_index]

        # 3. Zapis wyciętego auta
        if car is not None:
            car_path = os.path.join(save_dir, f"{base_filename}_car_{self.current_car_index}.jpg")
            cv2.imwrite(car_path, car)

        # 4. Zapis wyciętej tablicy (jeśli istnieje)
        if plates:
            plate_img = plates[0]
            plate_path = os.path.join(save_dir, f"{base_filename}_plate_{self.current_car_index}.jpg")
            cv2.imwrite(plate_path, plate_img)

        # 5. Zapis rozpoznanego tekstu
        plate_text = self.plate_text_box.get("1.0", tk.END).strip()
        if plate_text:
            text_path = os.path.join(save_dir, f"{base_filename}_text_{self.current_car_index}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(plate_text)

        # 6. Skopiowanie pełnego obrazu z naniesionymi ramkami z YOLO
        annotated_path_temp = os.path.join(self.results_dir, "prediction.jpg")
        if os.path.exists(annotated_path_temp):
            annotated_save_path = os.path.join(save_dir, f"{base_filename}_full_annotated.jpg")
            shutil.copy(annotated_path_temp, annotated_save_path)

        messagebox.showinfo(
            "Zapisano",
            f"Wyniki zostały uporządkowane i zapisane w folderze:\nresults/{base_filename}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()