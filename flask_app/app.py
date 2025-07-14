from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from multiprocessing.connection import Client
from shared_common import APP_TO_CAM_ADDRESS, AUTH_KEY
import subprocess
import atexit
import signal

# Start camera process
camera_process = subprocess.Popen(["python3", "take_image_for_detect.py"])

# Ensure the process is killed when Flask exits
def cleanup():
    try:
        print("Shutting down background camera process...")
        send_camera_command("shutdown")
        camera_process.send_signal(signal.SIGINT)
        camera_process.wait(timeout=5)
    except Exception as e:
        print("Error cleaning up camera process:", e)

atexit.register(cleanup)

# App initialization
app = Flask(__name__)

LOGO = "static/HAMK_Logo_vertical.jpg"
PREDICTED = "test/result.png"
RESULTS = "test/xyz_coordinate.txt"

def send_camera_command(command):
    try:
        conn = Client(APP_TO_CAM_ADDRESS, authkey=AUTH_KEY)
        conn.send(command)
        response = conn.recv()
        conn.close()
        return True, response
    except Exception as e:
        print(f"Camera command failed: {e}")
        return False, str(e)

@app.route("/")
def index():
    predicted_exists = os.path.exists(PREDICTED)
    colors = []
    if os.path.exists(RESULTS):
        with open(RESULTS, 'r') as f:
            lines = f.readlines()
            for line in lines:
                color = line.strip().split()[-1]
                if color not in colors:
                    colors.append(color)
    return render_template("index.html", predicted_exists=predicted_exists, colors=colors)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        conn = Client(APP_TO_CAM_ADDRESS, authkey=AUTH_KEY)
        conn.send("take_photo")
        print("[App] Sent: take_photo")
        response = conn.recv()  # Wait for the response
        print("[App] Received:", response)
        conn.close()
    except Exception as e:
        print("[App] Error communicating with camera process:", e)
    return redirect(url_for('index'))

@app.route("/pick", methods=["POST"])
def pick():
    color = request.form.get("color")
    if color:
        subprocess.Popen(["python3", "robot_matrix_transformation.py", color])
    return redirect(url_for('index'))

@app.route("/pick_all", methods=["POST"])
def pick_all():
    subprocess.Popen(["python3", "robot_matrix_transformation.py", "all"])
    return redirect(url_for('index'))

@app.route("/shutdown", methods=["POST"])
def shutdown():
    send_camera_command("shutdown")
    return "Camera shutdown signal sent."

@app.route("/image")
def image():
    if os.path.exists(PREDICTED):
        return send_file(PREDICTED, mimetype='image/png')
    return "No image", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
