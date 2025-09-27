"""
3D Rendering system for Pycraft
"""

import pygame
import math
import numpy as np
from typing import List, Tuple
from .blocks import BlockType
from .world import World
from .camera import Camera
from .font_manager import get_font_manager

class Renderer:
    """Simple 3D renderer for voxel world"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Pycraft - Minecraft-like Game")
        
        # Projection settings
        self.fov = math.radians(70)
        self.aspect_ratio = screen_width / screen_height
        self.near = 0.1
        self.far = 100.0
        
        # Create projection matrix
        self.projection_matrix = self.create_projection_matrix()
        
        # Colors for different faces (for simple shading)
        self.face_brightness = {
            'top': 1.0,
            'bottom': 0.5,
            'front': 0.8,
            'back': 0.8,
            'left': 0.6,
            'right': 0.6
        }
        
        # Debug mode
        self.debug_mode = False
    
    def create_projection_matrix(self) -> np.ndarray:
        """Create perspective projection matrix"""
        f = 1.0 / math.tan(self.fov / 2)
        
        return np.array([
            [f / self.aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), 
             (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ])
    
    def world_to_screen(self, world_pos: np.ndarray, view_matrix: np.ndarray) -> Tuple[int, int, float]:
        """Transform world coordinates to screen coordinates"""
        try:
            # Convert to homogeneous coordinates
            world_pos_h = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=float)

            # View transform
            view_pos = view_matrix @ world_pos_h  # (x_v, y_v, z_v, 1)

            # In our right‑handed system camera looks down -Z; anything with z > -near is behind/too close
            if view_pos[2] > -self.near:
                return None

            # Projection (use the precomputed projection matrix for correct perspective)
            clip_pos = self.projection_matrix @ view_pos  # (x_c, y_c, z_c, w_c)
            w = clip_pos[3]
            if abs(w) < 1e-6:
                return None

            # Perspective divide -> Normalized Device Coordinates
            ndc_x = clip_pos[0] / w
            ndc_y = clip_pos[1] / w
            ndc_z = clip_pos[2] / w  # Can be used for depth sorting if needed

            # (Optional) Reject if completely outside clip volume (slight padding)
            if ndc_x < -1.2 or ndc_x > 1.2 or ndc_y < -1.2 or ndc_y > 1.2:
                # Allow slight overshoot so large nearby faces still draw
                pass

            # Convert to screen space
            screen_x = int((ndc_x * 0.5 + 0.5) * self.screen_width)
            screen_y = int((1 - (ndc_y * 0.5 + 0.5)) * self.screen_height)  # Flip Y

            depth = -view_pos[2]  # Positive depth value (distance along camera forward)

            return screen_x, screen_y, depth
            
        except Exception as e:
            return None
    
    def is_face_visible(self, face_normal: np.ndarray, view_direction: np.ndarray) -> bool:
        """Check if a face is visible (back-face culling)"""
        return np.dot(face_normal, view_direction) < 0
    
    def is_in_frustum(self, world_pos: Tuple[float, float, float], camera_pos: np.ndarray, 
                     view_matrix: np.ndarray) -> bool:
        """Check if a block is within the camera's viewing frustum"""
        x, y, z = world_pos
        
        # Convert block position to view space
        world_pos_h = np.array([x + 0.5, y + 0.5, z + 0.5, 1], dtype=float)  # Use block center
        view_pos = view_matrix @ world_pos_h
        
        # Check if behind camera (with some tolerance)
        if view_pos[2] > -self.near:  # Behind / too close
            return False
        
        # Check distance
        distance = np.linalg.norm(np.array([x, y, z]) - camera_pos)
        if distance > 40:  # Too far
            return False
        
        # Check horizontal FOV bounds
        z_depth = -view_pos[2]
        if z_depth > 0.1:  # Avoid division by zero
            fov_scale = math.tan(self.fov / 2)
            
            # Calculate horizontal and vertical bounds in view space
            x_bound = z_depth * fov_scale * self.aspect_ratio
            y_bound = z_depth * fov_scale
            
            # Check if block is within frustum with some padding
            padding = 1.5  # Block size + some margin
            if (abs(view_pos[0]) > x_bound + padding or 
                abs(view_pos[1]) > y_bound + padding):
                return False
        
        return True
    
    def is_occluded(self, world_pos: Tuple[int, int, int], camera_pos: np.ndarray, 
                   world) -> bool:
        """Simple occlusion check - see if block is hidden behind other blocks"""
        x, y, z = world_pos
        
        # Vector from camera to block
        block_center = np.array([x + 0.5, y + 0.5, z + 0.5])
        to_block = block_center - camera_pos
        distance = np.linalg.norm(to_block)
        
        # Skip occlusion check for very close blocks
        if distance < 3.0:
            return False
        
        # Normalize direction vector
        direction = to_block / distance
        
        # Check a few points along the ray to see if they're blocked
        steps = min(int(distance), 8)  # Don't check too many points
        for i in range(1, steps):
            t = (i / steps) * distance * 0.8  # Check 80% of the way
            check_pos = camera_pos + direction * t
            
            # Convert to block coordinates
            check_x = int(math.floor(check_pos[0]))
            check_y = int(math.floor(check_pos[1]))
            check_z = int(math.floor(check_pos[2]))
            
            # Check if there's a solid block at this position
            try:
                block = world.get_block(check_x, check_y, check_z)
                if block and block.is_solid():
                    return True  # Block is occluded
            except:
                continue  # Skip invalid positions
        
        return False
    
    def render_block(self, world_pos: Tuple[int, int, int], block_color: Tuple[int, int, int], 
                    view_matrix: np.ndarray, camera_pos: np.ndarray, world):
        """Render a single block with face culling"""
        x, y, z = world_pos
        base_color = np.array(block_color)
        
        # Define cube vertices (relative to block position)
        vertices = [
            np.array([x, y, z]),       # 0: bottom-left-front
            np.array([x+1, y, z]),     # 1: bottom-right-front
            np.array([x+1, y+1, z]),   # 2: top-right-front
            np.array([x, y+1, z]),     # 3: top-left-front
            np.array([x, y, z+1]),     # 4: bottom-left-back
            np.array([x+1, y, z+1]),   # 5: bottom-right-back
            np.array([x+1, y+1, z+1]), # 6: top-right-back
            np.array([x, y+1, z+1])    # 7: top-left-back
        ]
        
        # Face definitions with neighbor positions for culling
        face_definitions = [
            ([vertices[0], vertices[1], vertices[2], vertices[3]], 'front', (0, 0, -1)),   # Front face
            ([vertices[5], vertices[4], vertices[7], vertices[6]], 'back', (0, 0, 1)),     # Back face  
            ([vertices[4], vertices[0], vertices[3], vertices[7]], 'left', (-1, 0, 0)),    # Left face
            ([vertices[1], vertices[5], vertices[6], vertices[2]], 'right', (1, 0, 0)),    # Right face
            ([vertices[3], vertices[2], vertices[6], vertices[7]], 'top', (0, 1, 0)),      # Top face
            ([vertices[4], vertices[5], vertices[1], vertices[0]], 'bottom', (0, -1, 0))   # Bottom face
        ]
        
        # Only render faces that are exposed (not adjacent to solid blocks)
        for face_vertices, face_name, (dx, dy, dz) in face_definitions:
            # Check if there's a solid block adjacent to this face
            neighbor_x, neighbor_y, neighbor_z = x + dx, y + dy, z + dz
            neighbor_block = world.get_block(neighbor_x, neighbor_y, neighbor_z)
            
            # Only render face if neighbor is not solid (air, water, or out of bounds)
            if not neighbor_block.is_solid():
                self.render_face(face_vertices, base_color, face_name, view_matrix)
    
    def render_face(self, vertices: List[np.ndarray], base_color: np.ndarray, 
                   face_name: str, view_matrix: np.ndarray):
        """Render a single face of a block (simplified polygon version)"""
        # Early reject: if any vertex is behind the near plane, skip entire face to avoid distortion
        for v in vertices:
            v_h = np.array([v[0], v[1], v[2], 1.0], dtype=float)
            view_pos = view_matrix @ v_h
            if view_pos[2] > -self.near:  # Behind / too close
                return

        # Project all vertices
        screen_points = []
        for vertex in vertices:
            result = self.world_to_screen(vertex, view_matrix)
            if result is None:
                return  # Safety: skip if any failed
            x, y, depth = result
            screen_points.append((int(x), int(y)))

        # Apply lighting (simple face brightness)
        brightness = self.face_brightness.get(face_name, 0.8)
        face_color = (base_color * brightness).astype(int)
        face_color = np.clip(face_color, 0, 255)

        # Draw polygon
        try:
            pygame.draw.polygon(self.screen, tuple(face_color), screen_points)
            pygame.draw.polygon(self.screen, (0, 0, 0), screen_points, 1)
        except (ValueError, TypeError):
            pass
    
    def render_world(self, world: World, camera: Camera, performance_mode=True):
        """Render the visible world"""
        # Clear screen
        self.screen.fill((135, 206, 235))  # Sky blue
        
        # Get view matrix
        view_matrix = camera.get_view_matrix()
        
        # Get visible chunks around camera
        camera_pos = camera.position
        
        # Reduce render distance for better performance
        visible_chunks = world.get_visible_chunks(int(camera_pos[0]), int(camera_pos[2]), render_distance=2)
        
        # Collect and render all nearby visible blocks with culling
        rendered_count = 0
        culled_count = 0
        total_blocks_checked = 0
        max_blocks_per_frame = 2000 if performance_mode else 4000  # Can increase since we're culling
        
        # Collect blocks with distance for sorting
        blocks_to_render = []
        
        for chunk in visible_chunks:
            chunk_world_x = chunk.x * chunk.SIZE
            chunk_world_z = chunk.z * chunk.SIZE
            
            chunk_blocks = len(chunk.blocks)
            total_blocks_checked += chunk_blocks
            
            for (local_x, local_y, local_z), block in chunk.blocks.items():
                if block.is_solid():
                    world_x = chunk_world_x + local_x
                    world_y = local_y
                    world_z = chunk_world_z + local_z
                    
                    # Frustum culling - skip blocks outside view
                    if not self.is_in_frustum((world_x, world_y, world_z), camera_pos, view_matrix):
                        culled_count += 1
                        continue
                    
                    # Calculate distance for sorting
                    dx = world_x - camera_pos[0]
                    dy = world_y - camera_pos[1] 
                    dz = world_z - camera_pos[2]
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Occlusion culling for distant blocks
                    if distance > 8.0 and self.is_occluded((world_x, world_y, world_z), camera_pos, world):
                        culled_count += 1
                        continue
                    
                    blocks_to_render.append((distance, world_x, world_y, world_z, block))
        
        # Sort by distance (closest first for better occlusion)
        blocks_to_render.sort(key=lambda x: x[0])
        
        # Render sorted blocks
        for distance, world_x, world_y, world_z, block in blocks_to_render:
            if rendered_count >= max_blocks_per_frame:
                break
                
            block_color = block.get_color()
            try:
                self.render_block((world_x, world_y, world_z), block_color, view_matrix, camera_pos, world)
                rendered_count += 1
            except Exception as e:
                # Skip problematic blocks but continue rendering
                continue
        
        # Debug output to help diagnose rendering issues
        culling_efficiency = (culled_count / max(total_blocks_checked, 1)) * 100
        if rendered_count == 0:
            print(f"WARNING: 0 blocks rendered! Camera at {camera_pos}")
            print(f"  Total blocks: {total_blocks_checked}, Culled: {culled_count} ({culling_efficiency:.1f}%)")
            # Show first few blocks for debugging
            block_count = 0
            for chunk in visible_chunks[:3]:  # Check first 3 chunks
                chunk_world_x = chunk.x * chunk.SIZE
                chunk_world_z = chunk.z * chunk.SIZE
                for (local_x, local_y, local_z), block in list(chunk.blocks.items())[:5]:
                    world_x = chunk_world_x + local_x
                    world_y = local_y
                    world_z = chunk_world_z + local_z
                    dx = abs(world_x - camera_pos[0])
                    dy = abs(world_y - camera_pos[1])
                    dz = abs(world_z - camera_pos[2])
                    in_frustum = self.is_in_frustum((world_x, world_y, world_z), camera_pos, view_matrix)
                    print(f"  Block at ({world_x},{world_y},{world_z}) distance=({dx:.1f},{dy:.1f},{dz:.1f}) in_frustum={in_frustum} type={block.type.name}")
                    block_count += 1
                    if block_count >= 5:
                        break
                if block_count >= 5:
                    break
        elif culling_efficiency > 50:  # Only show when culling is working well
            print(f"Culling working: {rendered_count} rendered, {culled_count} culled ({culling_efficiency:.1f}%)")
        
        # Draw crosshair
        self.draw_crosshair()
        
        # Optional debug info
        if hasattr(self, 'debug_mode') and self.debug_mode:
            font_mgr = get_font_manager()
            render_text = f"渲染: {rendered_count} 方塊"
            text_surface = font_mgr.render_text(render_text, size=32, color=(255, 255, 255))
            self.screen.blit(text_surface, (10, 320))
            
            culling_text = f"剔除: {culled_count} 方塊 ({(culled_count/(culled_count+rendered_count)*100):.1f}%)"
            cull_surface = font_mgr.render_text(culling_text, size=32, color=(255, 255, 0))
            self.screen.blit(cull_surface, (10, 355))
        
        # Draw UI
        self.draw_ui(world, camera)
    
    def draw_crosshair(self):
        """Draw crosshair in center of screen"""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        size = 10
        color = (255, 255, 255)
        
        pygame.draw.line(self.screen, color, 
                        (center_x - size, center_y), 
                        (center_x + size, center_y), 2)
        pygame.draw.line(self.screen, color, 
                        (center_x, center_y - size), 
                        (center_x, center_y + size), 2)
    
    def draw_ui(self, world, camera):
        """Draw user interface elements"""
        font_mgr = get_font_manager()
        
        # Draw position info (larger and more prominent)
        pos = camera.position
        pos_text = f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        text_surface = font_mgr.render_text(pos_text, size=28, color=(255, 255, 0))
        # Add background for better visibility
        text_rect = text_surface.get_rect()
        text_rect.topleft = (10, 10)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
        self.screen.blit(text_surface, (10, 10))
        
        # Draw angle info
        angle_text = f"視角: pitch={camera.pitch:.2f}, yaw={camera.yaw:.2f}"
        angle_surface = font_mgr.render_text(angle_text, size=28, color=(255, 255, 0))
        angle_rect = angle_surface.get_rect()
        angle_rect.topleft = (10, 45)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), angle_rect.inflate(10, 5))
        self.screen.blit(angle_surface, (10, 45))
        
        # Draw controls
        controls = [
            "控制:",
            "WASD - 移動",
            "滑鼠 - 轉視角",
            "空格/Shift - 上升/下降",
            "左鍵 - 破壞方塊",
            "右鍵 - 放置方塊", 
            "1-4 - 選擇方塊",
            "Tab - 切換滑鼠",
            "ESC - 退出"
        ]
        
        for i, control in enumerate(controls):
            text_surface = font_mgr.render_text(control, size=24, color=(255, 255, 255))
            self.screen.blit(text_surface, (10, 80 + i * 28))