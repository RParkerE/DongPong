import cv2
import mediapipe as mp
import time
import math
import numpy as np


class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                smooth_landmarks=self.smooth,
                                min_detection_confidence=self.detectionCon,
                                min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def showFps(self, img):
        cTime = time.time()
        #print(cTime, self.pTime)
        fbs = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fbs)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # some time this angle comes zero, so below conditon we added
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 1)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 1)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 1)
            # cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 2)
        return angle

    def getDongPosition(self, img):
        left_hip = self.lmList[23][1:]
        right_hip = self.lmList[24][1:]
        dong = (int((left_hip[0] + right_hip[0]) / 2), int((left_hip[1] + right_hip[1]) / 2))
        return dong

class Joint:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
        self.color = (36,85,141)

class Dong:
    def __init__(self, pos, canvas, prevAnchor, prevJoints, count, theta, omega, dt=1, g=3.81):
        self.anchor = Joint(pos, 5)
        self.draw(canvas, [self.anchor])
        self.joints = [Joint((pos[0], pos[1] + 20), 5), Joint((pos[0], pos[1] + 40), 5), Joint((pos[0], pos[1] + 60), 5)]

        self.theta = theta
        self.omega = omega
        if count > 0:
            alpha1 = (-g / 20 * np.sin(self.theta[0]))
            self.theta[0] += self.omega[0] * dt
            self.omega[0] += alpha1 * dt
            seg1 = Joint((int(prevAnchor.position[0] - 20 * np.sin(self.theta[0])), int(prevAnchor.position[1] + 20 * np.cos(self.theta[0]))), 5)
            alpha2 = (-g / 20 * np.sin(self.theta[0]))
            self.theta[1] += self.omega[1] * dt
            self.omega[1] += alpha2 * dt
            seg2 = Joint((int(prevJoints[0].position[0] - 20 * np.sin(self.theta[1])), int(prevJoints[0].position[1] + 20 * np.cos(self.theta[1]))), 5)
            alpha3 = (-g / 20 * np.sin(self.theta[2]))
            self.theta[2] += self.omega[2] * dt
            self.omega[2] += alpha3 * dt
            seg3 = Joint((int(prevJoints[1].position[0] - 20 * np.sin(self.theta[2])), int(prevJoints[1].position[1] + 20 * np.cos(self.theta[2]))), 5)
            self.draw(canvas, [seg1, seg2, seg3])
        else:
            self.draw(canvas, self.joints)              

    def draw(self, canvas, joints):
        for joint in joints:
            cv2.circle(canvas, (joint.position[0], joint.position[1]), joint.radius, joint.color, cv2.FILLED)
        try:
            cv2.line(canvas, self.anchor.position, joints[0].position, self.anchor.color, 3)
            cv2.line(canvas, joints[0].position, joints[1].position, joints[0].color, 3)
            cv2.line(canvas, joints[1].position, joints[2].position, joints[1].color, 3)
        except:
            pass




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
        # print(lmList)
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