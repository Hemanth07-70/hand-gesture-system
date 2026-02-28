# Hand Gesture Monitoring System

Real-time hand gesture recognition: train a model on your gestures (e.g. 👍 Thumbs Up, 😊 Happy, 😢 Sad) and see the corresponding emoji on screen.

## Features

- **Login / Register** – Simple auth, then redirect to monitor
- **Live webcam** – MediaPipe hand detection and landmark extraction
- **Data collection** – Record samples per gesture (script or API)
- **Train model** – Random Forest on 21×3 hand landmarks
- **Real-time prediction** – Gesture → emoji display with confidence
- **Retrain** – Button on monitor page to retrain from collected data

## Gestures (default)

| Gesture     | Emoji |
|------------|-------|
| thumbs_up  | 👍    |
| thumbs_down| 👎    |
| peace      | ✌️    |
| ok         | 👌    |
| wave       | 👋    |
| happy      | 😊    |
| sad        | 😢    |
| crying     | 😭    |
| rock       | 🤘    |
| fist       | ✊    |
| open_palm  | 🖐️    |

## Requirements

- Python 3.9+
- Webcam (720p+ recommended)
- 8GB RAM (16GB recommended)

## Setup

```bash
cd hand-gesture-system
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set SECRET_KEY if you like
```

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Register or login, then you’re on the monitor page with live feed and emoji result.

## Collect data and train

1. **Collect samples** (200–500 per gesture, different angles/lighting):

   ```bash
   python scripts/collect_data.py
   ```
   Enter gesture name (e.g. `thumbs_up`, `happy`, `sad`). Point your hand at the webcam and press **SPACE** to save a batch. Repeat. Press **Q** to quit.

2. **Train**  
   On the monitor page, click **Retrain model**. The app uses `dataset/*.npz` to train a classifier and saves it under `model/saved_models/`.

## Project layout

```
hand-gesture-system/
├── app.py              # Flask app (login, monitor, video_feed, API)
├── config.py           # Paths, gesture→emoji map
├── auth/               # Registration and login
├── detection/          # Camera, MediaPipe, pipeline
├── data/               # Collection, preprocessing, users
├── model/              # Train and predict
├── templates/          # Login, register, monitor
├── static/             # CSS, JS, img (vibrant background)
├── dataset/            # Per-gesture .npz files
├── scripts/            # collect_data.py
└── model/saved_models/ # Trained classifier + label encoder
```

## Optional: vibrant background image

Replace `static/img/background.svg` with a JPG/PNG for a custom background. The UI uses a gradient overlay; the CSS references `background.svg` by default.
