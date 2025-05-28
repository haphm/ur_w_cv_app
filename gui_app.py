import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt

LOGO = "HAMK_Logo_vertical.jpg"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Spraycan Pickup App")
        self.setGeometry(0, 0, 600, 600)
        self.centerUI()
        self.setWindowIcon(QIcon(LOGO))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        label = QLabel("HAMK Tech Robotics Team", self)
        label.setFont(QFont("Inter", 20))
        label.setStyleSheet("color: white;"
                            "background-color: #003755;")
        label.setAlignment(Qt.AlignCenter)

        logo = QLabel(self)
        logo.setFixedWidth(100)
        logo.setFixedHeight(100)
        logo.setPixmap(QPixmap(LOGO))
        logo.setScaledContents(True)

        vbox = QVBoxLayout()
        vbox.addWidget(logo, 0, Qt.AlignBottom)
        vbox.addWidget(label, 1, Qt.AlignBottom)

        central_widget.setLayout(vbox)


    def centerUI(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # def image(self, path):
    #     label = QLabel(self)
    #     label.setGeometry(self.width() - 100, self.height() - 100, 100, 100)
    #     pixmap = QPixmap(path)
    #     label.setPixmap(pixmap)
    #     label.setScaledContents(True)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()