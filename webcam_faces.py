import argparse
import sys

import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel


def main():
    parser = argparse.ArgumentParser(description="웹캠에서 얼굴 위치 표시")
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (기본값: 0)")
    args = parser.parse_args()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise SystemExit("얼굴 탐지 모델을 불러오지 못했습니다.")
    print("카메라를 여는 중입니다. macOS에서 권한을 요청하면 허용해 주세요.", flush=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit("카메라를 열지 못했습니다. 권한과 카메라 연결을 확인해 주세요.")

    app = QApplication(sys.argv[:1])

    class VideoWindow(QLabel):
        def keyPressEvent(self, event):
            if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
                self.close()

    window = VideoWindow()
    window.setWindowTitle("PickAgent - Face Detection (Q or Esc to quit)")
    timer = QTimer()

    def update_frame():
        ok, frame = camera.read()
        if not ok:
            print("카메라 영상을 읽지 못했습니다.", flush=True)
            window.close()
            return
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for x, y, width, height in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        window.setPixmap(QPixmap.fromImage(image))

    timer.timeout.connect(update_frame)
    app.aboutToQuit.connect(camera.release)
    timer.start(33)
    window.show()
    print("영상 창에서 Q 또는 Esc를 누르면 종료합니다.", flush=True)
    try:
        app.exec()
    finally:
        camera.release()


if __name__ == "__main__":
    main()
