import cv2
import numpy as np

class Joint:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
        self.color = (36,85,141)

class Dong:
    def __init__(self, pos, canvas, prevAnchor, prevJoints, count, theta, omega, dt=1, g=3.81, damping=0.09):
        self.anchor = Joint(pos, 5)
        self.draw(canvas, [self.anchor])
        self.joints = [Joint((pos[0], pos[1] + 20), 5), Joint((pos[0], pos[1] + 40), 5), Joint((pos[0], pos[1] + 60), 5)]

        self.theta = theta
        self.omega = omega
        self.g = g
        self.damping = damping

        if count > 0:
            for i in range(3):
                alpha = (-self.g / 20 * np.sin(self.theta[i])) - (self.damping * self.omega[i])
                self.theta[i] += self.omega[i] * dt + 0.5 * alpha * dt ** 2
                self.omega[i] += alpha * dt

            seg1 = Joint((int(prevAnchor.position[0] - 20 * np.sin(self.theta[0])), int(prevAnchor.position[1] + 20 * np.cos(self.theta[0]))), 5)
            seg2 = Joint((int(prevJoints[0].position[0] - 20 * np.sin(self.theta[1])), int(prevJoints[0].position[1] + 20 * np.cos(self.theta[1]))), 5)
            seg3 = Joint((int(prevJoints[1].position[0] - 20 * np.sin(self.theta[2])), int(prevJoints[1].position[1] + 20 * np.cos(self.theta[2]))), 5)
            self.draw(canvas, [seg1, seg2, seg3])

            # Calculate velocity based on movement from previous frame
            self.velocity = ((self.anchor.position[0] - prevAnchor.position[0]) / dt, (self.anchor.position[1] - prevAnchor.position[1]) / dt)
        else:
            self.draw(canvas, self.joints)
            self.velocity = (0, 0)  # Initial velocity is zero

    def draw(self, canvas, joints):
        for joint in joints:
            cv2.circle(canvas, (joint.position[0], joint.position[1]), joint.radius, joint.color, cv2.FILLED)
        try:
            cv2.line(canvas, self.anchor.position, joints[0].position, self.anchor.color, 3)
            cv2.line(canvas, joints[0].position, joints[1].position, joints[0].color, 3)
            cv2.line(canvas, joints[1].position, joints[2].position, joints[1].color, 3)
        except:
            pass
