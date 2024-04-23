import cv2
import numpy as np

from shapely.geometry import Point, LineString

class Pong:
    def __init__(self, position, velocity, radius, color):
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color = color

    def update(self, frame, dong):
        # Update position based on velocity
        self.position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])

        # Check for collisions with frame edges
        if self.position[0] - self.radius < 0 or self.position[0] + self.radius > frame.shape[1]:
            self.velocity = (-self.velocity[0], self.velocity[1])  # Reverse horizontal velocity

        # Ensure ball doesn't go above the dong's anchor point
        if dong and self.position[1] - self.radius < dong.anchor.position[1]:
            self.position = (self.position[0], dong.anchor.position[1] + self.radius)

        if self.position[1] - self.radius < 0 or self.position[1] + self.radius > frame.shape[0]:
            self.velocity = (self.velocity[0], -self.velocity[1])  # Reverse vertical velocity

        # Check for collisions with Dong
        if dong:
            # Check for collisions with Dong's joints and lines between joints
            for i in range(len(dong.joints) - 1):
                joint1 = dong.joints[i]
                joint2 = dong.joints[i + 1]

                # Calculate vector from joint1 to joint2
                v = np.array([joint2.position[0] - joint1.position[0], joint2.position[1] - joint1.position[1]])
                
                # Calculate vector from joint1 to ball's position
                w = np.array([self.position[0] - joint1.position[0], self.position[1] - joint1.position[1]])
                
                # Calculate the dot product of w and v
                dot_product = np.dot(w, v)
                
                # Calculate the squared length of v
                length_squared = np.dot(v, v)
                
                # Calculate the projection parameter t
                t = dot_product / length_squared
                
                # Clamp t to be between 0 and 1
                t = max(0, min(1, t))
                
                # Calculate the closest point on the line segment to the ball's position
                closest_point = (joint1.position[0] + t * v[0], joint1.position[1] + t * v[1])
                
                # Check if the distance between the ball's position and the closest point is less than the ball's radius
                distance = np.linalg.norm(np.array(self.position) - np.array(closest_point))
                if distance < self.radius:
                    # Collision detected, adjust velocity
                    self.velocity = (-self.velocity[0], -self.velocity[1])  # Reverse horizontal and vertical velocity

            # Check for collisions with Dong's anchor
            distance = np.linalg.norm(np.array(self.position) - np.array(dong.anchor.position))
            if distance < self.radius + dong.anchor.radius:
                # Collision detected, change velocity based on Dong's movement
                self.velocity = (-self.velocity[0], -self.velocity[1])  # Reverse horizontal and vertical velocity

    def draw(self, frame):
        cv2.circle(frame, self.position, self.radius, self.color, cv2.FILLED)
