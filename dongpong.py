import cv2

from pose_detector import poseDetector
from dong import Dong

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)
    success, initImg = cap.read()
    count = 0
    prevAnchor = 0
    prevJoints = []
    theta = [0, 0, 0]
    omega = [0, 0, 0]

    while True:
        success, img = cap.read()
        img = detector.findPose(img, draw=False)
        lmList = detector.getPosition(img)
        detector.showFps(img)
        dongPos = detector.getDongPosition(img)
        dong = Dong(dongPos, img, prevAnchor, prevJoints, count, theta, omega)
        prevAnchor = dong.anchor
        prevJoints = dong.joints
        theta = dong.theta
        omega = dong.omega
        count = 1
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()