"""
3D Camera system for Pycraft
"""

import math
import numpy as np
from typing import Tuple

class Camera:
    """3D camera for first-person view"""
    
    def __init__(self, position: Tuple[float, float, float] = (0, 0, 0)):
        self.position = np.array(position, dtype=float)
        self.yaw = 0.0  # Rotation around Y axis
        self.pitch = 0.0  # Rotation around X axis
        
        # Camera settings
        self.fov = 70.0  # Field of view in degrees
        self.near = 0.1
        self.far = 100.0

        # Cheatsheet
        # +X: Right
        # +Y: Up
        # +Z: Forward
        # Yaw: 0 = +Z, π/2 = +X
        # Pitch: 0 = +Z, π/2 = +Y
        
        # Movement sensitivity
        self.mouse_sensitivity = 0.003  # Adjusted for better control
        
    def set_position(self, x: float, y: float, z: float):
        """Set camera position"""
        self.position = np.array([x, y, z], dtype=float)
    
    def move(self, dx: float, dy: float, dz: float):
        """Move camera relative to its current position"""
        self.position += np.array([dx, dy, dz])
    
    def rotate(self, dyaw: float, dpitch: float):
        """Rotate camera by given angles"""
        self.yaw += dyaw * self.mouse_sensitivity
        self.pitch += dpitch * self.mouse_sensitivity
        
        # Clamp pitch to prevent flipping
        self.pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.pitch))
    
    def get_forward_vector(self) -> np.ndarray:
        """Get the forward direction vector"""
        # Standard 3D coordinate system: Z+ is forward in world space
        # Pitch: negative = look down, positive = look up
        # Yaw: 0 = look toward +Z, π/2 = look toward +X
        return np.array([
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),  # Positive Y is up
            math.cos(self.pitch) * math.cos(self.yaw)
        ])
    
    def get_horizontal_forward_vector(self) -> np.ndarray:
        """Get the horizontal forward direction vector"""
        return np.array([
            math.sin(self.yaw),
            0,
            math.cos(self.yaw)
        ])
    
    def get_right_vector(self) -> np.ndarray:
        """Get the right direction vector"""
        return np.array([
            math.cos(self.yaw),
            0,
            -math.sin(self.yaw)
        ])
    
    def get_up_vector(self) -> np.ndarray:
        """Get the up direction vector"""
        return np.array([0, 1, 0])
    
    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix for rendering"""
        forward = self.get_forward_vector()
        target = self.position + forward
        up = self.get_up_vector()
        
        # Simple view matrix calculation
        return self.look_at(self.position, target, up)
    
    @staticmethod
    def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create a look-at matrix"""
        # Calculate view direction (from eye to target)
        z = target - eye  # Forward direction (opposite of standard)
        z = z / np.linalg.norm(z)
        
        # Calculate right vector
        x = np.cross(z, up)
        x = x / np.linalg.norm(x)
        
        # Calculate up vector (orthogonal to forward and right)
        y = np.cross(x, z)
        
        # Create view matrix (camera transform)
        view_matrix = np.array([
            [x[0], x[1], x[2], -np.dot(x, eye)],
            [y[0], y[1], y[2], -np.dot(y, eye)],
            [-z[0], -z[1], -z[2], np.dot(z, eye)],  # Negative Z for right-handed system
            [0, 0, 0, 1]
        ])
        
        return view_matrix