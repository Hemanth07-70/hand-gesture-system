"""Flask app: login, register, monitor, video feed, data collection, training."""
import io
import os
import base64
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    Response,
)
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

import config
from auth.login import register_user, check_user
from detection.pipeline import get_pipeline
from data.collect import save_samples
from data.preprocess import landmarks_to_features
from model.train import train

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
CORS(app)


def login_required(f):
    from functools import wraps
    @wraps(f)
    def inner(*args, **kwargs):
        if session.get("username") is None:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return inner


@app.route("/")
def index():
    if session.get("username"):
        return redirect(url_for("monitor"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if check_user(username, password):
            session["username"] = username
            return redirect(url_for("monitor"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error=None)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        ok, msg = register_user(username, password)
        if ok:
            session["username"] = username
            return redirect(url_for("monitor"))
        return render_template("register.html", error=msg)
    return render_template("register.html", error=None)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/monitor")
@login_required
def monitor():
    return render_template(
        "monitor.html",
        username=session.get("username"),
        gestures=list(config.GESTURE_DISPLAY_NAMES.keys()),
        gesture_emoji=config.GESTURE_EMOJI_MAP,
    )


def generate_frames():
    import cv2
    import numpy as np
    import time
    
    pipeline = get_pipeline()
    print("DEBUG: generate_frames started")
    while True:
        try:
            frame, _, _, _ = pipeline.read_frame()
        except Exception as e:
            print(f"DEBUG: Frame read exception: {e}")
            time.sleep(1)
            continue
            
        if frame is None:
            print("DEBUG: frame is None")
            # Create a black frame with error text
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error: Not found or no permission", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(error_frame, "Check System Settings > Privacy > Camera", (50, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            _, buf = cv2.imencode(".jpg", error_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(1) # Wait before retry
            continue
            
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/predict")
@login_required
def api_predict():
    """Return current prediction as JSON (label, confidence, emoji). Updated by video_feed loop."""
    pipeline = get_pipeline()
    return jsonify({
        "label": pipeline._current_label,
        "confidence": pipeline._current_conf,
        "emoji": pipeline._current_emoji,
        "landmarks": pipeline._last_landmarks,
    })


@app.route("/api/collect", methods=["POST"])
@login_required
def api_collect():
    """Accept gesture label and list of landmark arrays; save to dataset."""
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "message": "No JSON"}), 400
    label = (data.get("gesture") or data.get("label") or "").strip()
    landmarks_list = data.get("landmarks", [])
    if not label or not landmarks_list:
        return jsonify({"ok": False, "message": "gesture and landmarks required"}), 400
    try:
        import numpy as np
        arrs = [np.array(lm, dtype=np.float32) for lm in landmarks_list]
        n = save_samples(label, arrs)
        return jsonify({"ok": True, "saved": n})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    """Trigger model training."""
    ok, msg = train()
    if ok:
        get_pipeline().reload_model()
    return jsonify({"ok": ok, "message": msg})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
