import cv2

from pose_detector import poseDetector
from dong import Dong
from pong import Pong

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)
    success, initImg = cap.read()
    count = 0
    prevAnchor = None
    prevJoints = []
    theta = [0, 0, 0]
    omega = [0, 0, 0]

    pong = Pong((250, 250), (5, 5), 10, (255, 0, 0))

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from video feed")
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.getPosition(img)
        detector.showFps(img)
        dongPos = detector.getDongPosition(img)
        
        # Ensure dong position is detected
        if dongPos:
            dong = Dong(dongPos, img, prevAnchor, prevJoints, count, theta, omega)
            prevAnchor = dong.anchor
            prevJoints = dong.joints
            theta = dong.theta
            omega = dong.omega
            count = 1
        else:
            dong = None

        # Update Ball
        pong.update(img, dong)

        # Draw Dong and Ball
        if dong:
            dong.draw(img, dong.joints)  # Pass the 'joints' argument here
        pong.draw(img)

        cv2.imshow("Image", img)
        
        # Control frame rate (delay in milliseconds)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
