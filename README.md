# Car detection and license plate recognition
___
### Project description
The project is a simple desktop application that allows the user to automatically detect vehicles and license plates in the loaded images, and then read the detected license plates.

The project was written in Python using the `ultralytics` package for vehicle and license plate detection, `easyocr` for robust international license plate character recognition, and `tkinter` for the GUI.

### Authors
Bartosz Biesaga, Mateusz Wawrzyczek, Piotr Gwioździk
  
### Starting the app

#### Requirements
The GUI was created using the `tkinter` package, which in most cases is delivered with Python. In case it is not available (e.g., on some Linux distributions), it can be installed with the following command:
```shell
sudo apt-get install python3-tk
```
The application was developed and tested using Python 3.12.

#### Installing dependencies and starting the app in a virtual environment

**For Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main/gui.py
```

**For Linux / macOS:**
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main/gui.py
```


### Folder structure

```text
.
├── dokumentacja/          # Project documentation (written in Polish)
├── main/                  # Main source code of the application
│   ├── gui.py             # Graphical User Interface script (Tkinter)
│   ├── recognition.py     # License plate character recognition using EasyOCR
│   └── YOLO_utils.py      # Logic for vehicle and plate detection & cropping
├── models/                # Artificial Intelligence models and weights
│   └── YOLO/
│       ├── YOLO.ipynb     # Notebook used to train the YOLO detection model
│       └── weights/
│           └── best.pt    # Best trained weights for vehicle/plate detection
├── results/               # Destination folder for saved outputs (cropped images, text files)
├── test_data/             # Sample images used for testing the application
├── .gitignore             # Git ignore file
├── README.md              # Project description and manual
└── requirements.txt       # Unified dependencies list (ultralytics, easyocr, etc.)