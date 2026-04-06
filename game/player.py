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
        self.vertical_velocity = 0.0
        self.horizontal_velocity = np.array([0.0, 0.0])  # XZ velocity for realistic mode
        self.on_ground = False

        # Realistic movement tuning
        self.ground_acceleration = 56.0
        self.air_acceleration = 20.0
        self.ground_drag = 20.0
        self.air_drag = 2.0

        # Collision body settings (camera position is eye height)
        self.body_half_width = 0.3
        self.body_eye_height = 1.6
        self.body_head_margin = 0.1
        
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
            self.horizontal_velocity = np.array([0.0, 0.0])
            if keys[pygame.K_SPACE]:
                movement += up * move_speed * dt
            if keys[pygame.K_LSHIFT]:
                movement -= up * move_speed * dt

            # Apply movement directly in fly mode
            self.camera.move(movement[0], movement[1], movement[2])
            return

        # Realistic mode: acceleration-based horizontal motion + gravity + jump
        position = self.camera.position.copy()
        horizontal_dir = np.array([movement[0], 0.0, movement[2]])
        horizontal_norm = np.linalg.norm(horizontal_dir)
        if horizontal_norm > 0:
            horizontal_dir = horizontal_dir / horizontal_norm

        horizontal_vel = np.array([
            self.horizontal_velocity[0],
            0.0,
            self.horizontal_velocity[1],
        ])
        target_vel = horizontal_dir * self.speed

        accel = self.ground_acceleration if self.on_ground else self.air_acceleration
        velocity_delta = target_vel - horizontal_vel
        velocity_delta_y0 = np.array([velocity_delta[0], 0.0, velocity_delta[2]])
        velocity_delta_mag = np.linalg.norm(velocity_delta_y0)
        if velocity_delta_mag > 0:
            speed_change = min(velocity_delta_mag, accel * dt)
            horizontal_vel += (velocity_delta_y0 / velocity_delta_mag) * speed_change

        if horizontal_norm == 0:
            drag = self.ground_drag if self.on_ground else self.air_drag
            horizontal_speed = np.linalg.norm([horizontal_vel[0], horizontal_vel[2]])
            if horizontal_speed > 0:
                decel = min(horizontal_speed, drag * dt)
                horizontal_vel *= max(0.0, (horizontal_speed - decel) / horizontal_speed)

        move_x = horizontal_vel[0] * dt
        move_z = horizontal_vel[2] * dt

        position, hit_x = self._move_axis_with_collision(position, 0, move_x)
        position, hit_z = self._move_axis_with_collision(position, 2, move_z)
        if hit_x:
            horizontal_vel[0] = 0.0
        if hit_z:
            horizontal_vel[2] = 0.0

        self.horizontal_velocity = np.array([horizontal_vel[0], horizontal_vel[2]])

        self.vertical_velocity -= self.gravity * dt
        vertical_delta = self.vertical_velocity * dt
        position, hit_vertical = self._move_axis_with_collision(position, 1, vertical_delta)
        if hit_vertical:
            self.vertical_velocity = 0.0

        self.camera.set_position(position[0], position[1], position[2])

        # Keep stable ground contact for jump checks
        ground_probe = position.copy()
        ground_probe[1] -= 0.06
        self.on_ground = self._collides_at(ground_probe)

    def _move_axis_with_collision(self, position: np.ndarray, axis: int, delta: float):
        """Move along a single axis with small steps to prevent tunneling."""
        if abs(delta) < 1e-8:
            return position, False

        steps = max(1, int(abs(delta) / 0.05) + 1)
        step_delta = delta / steps

        current = position.copy()
        for _ in range(steps):
            trial = current.copy()
            trial[axis] += step_delta
            if self._collides_at(trial):
                return current, True
            current = trial

        return current, False

    def _collides_at(self, position: np.ndarray) -> bool:
        """Check if the player's collision box intersects any solid block."""
        min_x = position[0] - self.body_half_width
        max_x = position[0] + self.body_half_width
        min_y = position[1] - self.body_eye_height
        max_y = position[1] + self.body_head_margin
        min_z = position[2] - self.body_half_width
        max_z = position[2] + self.body_half_width

        x_start = math.floor(min_x) - 1
        x_end = math.ceil(max_x) + 1
        y_start = math.floor(min_y) - 1
        y_end = math.ceil(max_y) + 1
        z_start = math.floor(min_z) - 1
        z_end = math.ceil(max_z) + 1

        for bx in range(x_start, x_end + 1):
            for by in range(y_start, y_end + 1):
                for bz in range(z_start, z_end + 1):
                    block = self.world.get_block(bx, by, bz)
                    if not block.is_solid():
                        continue

                    block_min_x = bx - 0.5
                    block_max_x = bx + 0.5
                    block_min_y = by - 0.5
                    block_max_y = by + 0.5
                    block_min_z = bz - 0.5
                    block_max_z = bz + 0.5

                    overlap_x = max_x > block_min_x and min_x < block_max_x
                    overlap_y = max_y > block_min_y and min_y < block_max_y
                    overlap_z = max_z > block_min_z and min_z < block_max_z

                    if overlap_x and overlap_y and overlap_z:
                        return True

        return False

    def _resolve_initial_overlap(self):
        """If entering realistic mode inside blocks, nudge player upward until clear."""
        position = self.camera.position.copy()
        if not self._collides_at(position):
            return

        for _ in range(50):
            position[1] += 0.1
            if not self._collides_at(position):
                self.camera.set_position(position[0], position[1], position[2])
                return
    
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

        # Toggle realistic/fly movement mode
        if key == pygame.K_F5:
            self.flying = not self.flying
            self.vertical_velocity = 0.0
            self.horizontal_velocity = np.array([0.0, 0.0])
            if self.flying:
                self.on_ground = False
                print("切換為飛行模式")
            else:
                self._resolve_initial_overlap()
                ground_probe = self.camera.position.copy()
                ground_probe[1] -= 0.06
                self.on_ground = self._collides_at(ground_probe)
                print("切換為真實模式")

        # Jump in realistic mode
        if key == pygame.K_SPACE and not self.flying and self.on_ground:
            self.vertical_velocity = self.jump_velocity
            self.on_ground = False
    
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
    
    def get_target_block(self, place_mode=False, max_distance=PLAYER_HAND_REACH):
        """Get the position of the block the player is targeting"""
        start = self.camera.position # Eye position
        print("Camera position:", start)
        direction = self.camera.get_forward_vector()
        print("Camera direction:", direction)

        # Raycast to find target block
        for t in np.linspace(1, max_distance, num=int(max_distance * 10)):  # Check every 0.1 units
            pos = start + direction * t
            block_pos = np.round(pos).astype(int).tolist()

            target_block = self.world.get_block(*block_pos)
            if target_block.type != BlockType.AIR:
                if place_mode:
                    if not target_block.is_solid():
                        continue
                    # Return the position just before this solid block
                    prev_t = (t - 0.1) if (t - 0.1) > 0 else 0
                    prev_pos = start + direction * prev_t
                    return np.round(prev_pos).astype(int).tolist()
                else:
                    return block_pos
        
        return None