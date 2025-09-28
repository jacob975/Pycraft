"""
Player class and controls for Pycraft
"""

import pygame
import math
import numpy as np
from .camera import Camera
from .blocks import BlockType
from .world import World
from config import *

class Player:
    """Player class handling movement and interactions"""
    
    def __init__(self, world: World, spawn_position=(0, 50, 0)):
        self.world = world
        self.camera = Camera(spawn_position)
        
        # Player properties
        self.speed = PLAYER_SPEED # blocks per second
        self.fly_speed = FLY_SPEED # blocks per second when flying
        self.gravity = GRAVITY  # gravity acceleration
        self.jump_velocity = JUMP_VELOCITY  # initial jump velocity
        self.flying = True  # Start in fly mode for simplicity
        
        # Selected block type
        self.selected_block = BlockType.GRASS
        
        # Mouse control
        self.mouse_locked = False
        
    def update(self, dt: float):
        """Update player state"""
        keys = pygame.key.get_pressed()
        
        # Movement
        move_speed = self.fly_speed if self.flying else self.speed
        movement = np.array([0.0, 0.0, 0.0])
        
        # Get camera vectors
        forward = self.camera.get_horizontal_forward_vector()
        right = self.camera.get_right_vector()
        up = self.camera.get_up_vector()
        
        # WASD movement
        if keys[pygame.K_w]:
            if self.flying:
                movement += forward * move_speed * dt
            else:
                # Ground movement (ignore Y component)
                ground_forward = np.array([forward[0], 0, forward[2]])
                if np.linalg.norm(ground_forward) > 0:
                    ground_forward = ground_forward / np.linalg.norm(ground_forward)
                movement += ground_forward * move_speed * dt
        
        if keys[pygame.K_s]:
            if self.flying:
                movement -= forward * move_speed * dt
            else:
                ground_forward = np.array([forward[0], 0, forward[2]])
                if np.linalg.norm(ground_forward) > 0:
                    ground_forward = ground_forward / np.linalg.norm(ground_forward)
                movement -= ground_forward * move_speed * dt
        
        if keys[pygame.K_d]:
            movement -= right * move_speed * dt
        
        if keys[pygame.K_a]:
            movement += right * move_speed * dt
        
        # Vertical movement (only in fly mode)
        if self.flying:
            if keys[pygame.K_SPACE]:
                movement += up * move_speed * dt
            if keys[pygame.K_LSHIFT]:
                movement -= up * move_speed * dt
        
        # Apply movement
        self.camera.move(movement[0], movement[1], movement[2])
    
    def handle_mouse_motion(self, rel_x: int, rel_y: int):
        """Handle mouse movement for camera rotation"""
        if self.mouse_locked:
            # Standard FPS mouse control: right=positive yaw, down=positive pitch
            self.camera.rotate(-rel_x, -rel_y)
    
    def handle_mouse_click(self, button: int, pos: tuple):
        """Handle mouse clicks for block interaction"""
        if button == 1:  # Left click - break block
            self.break_block()
        elif button == 3:  # Right click - place block
            self.place_block()
    
    def handle_key_press(self, key: int):
        """Handle key press events"""
        # Block selection
        if key == pygame.K_1:
            self.selected_block = BlockType.GRASS
        elif key == pygame.K_2:
            self.selected_block = BlockType.DIRT
        elif key == pygame.K_3:
            self.selected_block = BlockType.STONE
        elif key == pygame.K_4:
            self.selected_block = BlockType.WOOD
        
        # Toggle mouse lock
        if key == pygame.K_TAB:
            self.toggle_mouse_lock()
    
    def toggle_mouse_lock(self):
        """Toggle mouse lock for camera control"""
        self.mouse_locked = not self.mouse_locked
        pygame.mouse.set_visible(not self.mouse_locked)
        if self.mouse_locked:
            pygame.event.set_grab(True)
        else:
            pygame.event.set_grab(False)
    
    def break_block(self):
        """Break block at target position"""
        target_pos = self.get_target_block()
        if target_pos:
            x, y, z = target_pos
            self.world.set_block(x, y, z, BlockType.AIR)
    
    def place_block(self):
        """Place block at target position"""
        target_pos = self.get_target_block(place_mode=True)
        if target_pos:
            x, y, z = target_pos
            self.world.set_block(x, y, z, self.selected_block)
    
    def get_target_block(self, place_mode=False, max_distance=5):
        """Get the position of the block the player is targeting"""
        start = self.camera.position # Eye position
        print("Camera position:", start)
        direction = self.camera.get_forward_vector()
        print("Camera direction:", direction)

        # Raycast to find target block
        for distance in range(1, max_distance * 10):  # Check every 0.1 units
            t = distance / 10.0
            pos = start + direction * t
            
            #block_pos = (int(math.floor(pos[0])), 
            #            int(math.floor(pos[1])), 
            #            int(math.floor(pos[2])))
            block_pos = np.round(pos).astype(int).tolist()
            
            if self.world.get_block(*block_pos).is_solid():
                if place_mode:
                    # Return the position just before this solid block
                    prev_t = (distance - 1) / 10.0
                    prev_pos = start + direction * prev_t
                    return np.round(prev_pos).astype(int).tolist()
                else:
                    return block_pos
        
        return None