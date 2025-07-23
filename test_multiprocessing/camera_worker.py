# camera_worker.py
import cv2
from multiprocessing.connection import Listener
from common import MAIN_TO_CAM_ADDRESS, AUTH_KEY
import datetime

def main():
    cap = cv2.VideoCapture(0)
    listener = Listener(MAIN_TO_CAM_ADDRESS, authkey=AUTH_KEY)
    print("Camera worker: Waiting for signal...")

    while True:
        conn = listener.accept()
        msg = conn.recv()
        if msg == "take_photo":
            ret, frame = cap.read()
            if ret:
                filename = datetime.datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved photo: {filename}")
            else:
                print("Failed to take photo.")
        conn.close()

if __name__ == "__main__":
    main()
