import os
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QLabel, QWidget, QPushButton, \
    QRadioButton, QVBoxLayout, QFrame, QHBoxLayout, QGroupBox, QScrollArea
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

LOGO = "HAMK_Logo_vertical.jpg"
PREDICTED = "./test/result.png"
RESULTS = "./test/xyz_coordinate.txt"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spraycan Pickup App")
        self.setGeometry(0, 0, 600, 600)
        self.centerUI()
        self.initUI()

    def initUI(self):
        # Logo
        self.logo = QLabel(self)
        self.logo.setPixmap(QPixmap(LOGO).scaledToHeight(100, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignCenter)

        # Task buttons
        self.btn_detect = QPushButton("Detect", self)
        self.btn_detect.setStyleSheet("font: Inter;"
                                      "font-size: 30px;")
        self.btn_detect.clicked.connect(self.detection)

        self.btn_color = QPushButton("Colors", self)
        self.btn_color.setStyleSheet("font: Inter;"
                                     "font-size: 30px;")
        self.btn_color.clicked.connect(self.load_colors)

        self.btn_pick = QPushButton("Pick", self)
        self.btn_pick.setStyleSheet("font: Inter;"
                                    "font-size: 30px;")
        self.btn_pick.clicked.connect(self.picking)

        self.btn_pick_all = QPushButton("Pick All", self)
        self.btn_pick_all.setStyleSheet("font: Inter;"
                                    "font-size: 30px;")
        self.btn_pick_all.clicked.connect(self.picking_all)

        # Radio button container
        self.rdo_layout = QVBoxLayout()
        self.rdo_group_box = QGroupBox("Colors")
        self.rdo_group_box.setLayout(self.rdo_layout)

        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("background-color: #FFD7BE;")
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.rdo_group_box)
        scroll_area.setFixedHeight(300)

        # Left panel layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_detect)
        left_layout.addWidget(self.btn_color)
        left_layout.addWidget(scroll_area)
        left_layout.addWidget(self.btn_pick)
        left_layout.addWidget(self.btn_pick_all)
        left_layout.addStretch(1)

        # Image display
        self.image = QLabel(self)
        self.image.setFixedWidth(640)
        self.image.setFixedHeight(480)
        self.image.setPixmap(QPixmap(PREDICTED))
        self.image.setScaledContents(True)

        # Combine layout
        central_layout = QHBoxLayout()
        left_frame = QFrame()
        left_frame.setLayout(left_layout)
        left_frame.setFixedWidth(150)

        central_layout.addWidget(left_frame)
        central_layout.addWidget(self.image)

        # Team name
        self.label = QLabel("HAMK Tech Robotics Team", self)
        self.label.setFont(QFont("Inter", 20))
        self.label.setStyleSheet("color: white;"
                                 "background-color: #003755;")
        self.label.setAlignment(Qt.AlignCenter)

        # Final layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.logo)
        main_layout.addLayout(central_layout)
        main_layout.addWidget(self.label)

        self.setLayout(main_layout)

    def centerUI(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def detection(self):
        print("Detection started!")
        self.btn_detect.setText("Detecting...")
        self.btn_detect.setDisabled(True)
        subprocess.run(["python3", "take_image_for_detect.py"])

        if os.path.exists(PREDICTED):
            self.image.setPixmap(QPixmap(PREDICTED))
            self.btn_detect.setText("Detect")
            self.btn_detect.setDisabled(False)
        else:
            self.image.setText("Detection failed!")

    def load_colors(self):
        with open(RESULTS, 'r') as f:
            lines = f.readlines()
            colors = []
            for line in lines:
                colors.append(line.strip().split()[-1])

        for i in reversed(range(self.rdo_layout.count())):
            widget = self.rdo_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for color in colors:
            rdo_btn = QRadioButton(color)
            rdo_btn.setStyleSheet(f"color: {color};"
                                  "font-size: 25px;")
            rdo_btn.toggled.connect(self.rdo_btn_changed)
            self.rdo_layout.addWidget(rdo_btn)

    def rdo_btn_changed(self):
        rdo_btn_check = self.sender()
        if rdo_btn_check.isChecked():
            print(f"color {rdo_btn_check.text()} is selected")

    def picking(self):
        selected_color = None
        for i in range(self.rdo_layout.count()):
            widget = self.rdo_layout.itemAt(i).widget()
            if isinstance(widget, QRadioButton) and widget.isChecked():
                selected_color = widget.text()
        if selected_color:
            subprocess.run(["python3", "robot_matrix_transformation.py", selected_color])
            print("Process is finished!")

    def picking_all(self):
        subprocess.run(["python3", "robot_matrix_transformation.py", "all"])
        print("Process is finished!")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()