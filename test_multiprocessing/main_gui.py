# main_gui.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from multiprocessing.connection import Client
from common import MAIN_TO_CAM_ADDRESS, MAIN_TO_GRIPPER_ADDRESS, AUTH_KEY

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Controller")
        layout = QVBoxLayout()

        self.photo_button = QPushButton("Take Photo")
        self.photo_button.clicked.connect(self.send_take_photo)
        layout.addWidget(self.photo_button)

        self.gripper_open_button = QPushButton("Open Gripper")
        self.gripper_open_button.clicked.connect(lambda: self.send_gripper("open"))
        layout.addWidget(self.gripper_open_button)

        self.gripper_close_button = QPushButton("Close Gripper")
        self.gripper_close_button.clicked.connect(lambda: self.send_gripper("close"))
        layout.addWidget(self.gripper_close_button)

        self.setLayout(layout)

    def send_take_photo(self):
        try:
            conn = Client(MAIN_TO_CAM_ADDRESS, authkey=AUTH_KEY)
            conn.send("take_photo")
            conn.close()
        except Exception as e:
            print("Failed to send photo command:", e)

    def send_gripper(self, command):
        try:
            conn = Client(MAIN_TO_GRIPPER_ADDRESS, authkey=AUTH_KEY)
            conn.send(command)
            conn.close()
        except Exception as e:
            print("Failed to send gripper command:", e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
