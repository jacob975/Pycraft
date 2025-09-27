"""
OpenGL-based GPU renderer for Pycraft.

This implementation uses PyOpenGL with a legacy immediate-mode pipeline
for simplicity. It mirrors the public API of the existing CPU `Renderer`
class so `engine.GameEngine` can seamlessly switch between them.

Key features:
  * Frustum-limited chunk selection (same as CPU renderer logic)
  * Per-face visibility (only draw faces exposed to air)
  * Simple brightness shading per face (same brightness map)
  * Crosshair + UI & debug overlay using pygame font surfaces
  * Graceful fallback: raise ImportError if OpenGL init fails

Notes:
  This is intentionally simple (no VBO batching yet). For performance
  improvements, a future pass can build meshed chunk VBOs.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple, List
from .world import World, Chunk
from .camera import Camera
from .blocks import Block

import pygame

try:
    from OpenGL.GL import (
        glClearColor, glClear, glEnable, glDisable, glViewport,
        glMatrixMode, glLoadIdentity, glBegin, glEnd, glVertex3f, glColor3f,
        glLineWidth, glVertex2f, glPushMatrix, glPopMatrix, glOrtho,
        glTranslatef, glRotatef, glFlush, glDepthMask, glHint,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
        GL_PROJECTION, GL_MODELVIEW, GL_QUADS, GL_LINES
    )
    from OpenGL.GLU import gluPerspective, gluLookAt
except Exception as e:  # Broad except so engine fallback works
    raise ImportError(f"PyOpenGL 初始化失敗: {e}")

from .blocks import BlockType
from .font_manager import get_font_manager


class GPURenderer:
    """GPU renderer using OpenGL immediate mode.

    Public API expected by engine:
      render_world(world, camera, performance_mode=True)
      draw_debug_info(debug_data: dict)
    """

    FACE_DELTAS = {
        'front':  (0, 0, -1),
        'back':   (0, 0, 1),
        'left':   (-1, 0, 0),
        'right':  (1, 0, 0),
        'top':    (0, 1, 0),
        'bottom': (0, -1, 0),
    }

    FACE_BRIGHTNESS = {
        'top': 1.0,
        'bottom': 0.5,
        'front': 0.8,
        'back': 0.8,
        'left': 0.6,
        'right': 0.6,
    }

    # Each face defined by 4 corner offsets (x,y,z) relative to block origin
    # Vertices ordered counter-clockwise when viewed from outside the cube
    FACE_VERTICES = {
        'front': [ (0,0,0), (0,1,0), (1,1,0), (1,0,0) ],            # -Z (CCW from outside)
        'back':  [ (1,0,1), (1,1,1), (0,1,1), (0,0,1) ],            # +Z (CCW from outside)
        'left':  [ (0,0,1), (0,1,1), (0,1,0), (0,0,0) ],            # -X (CCW from outside)
        'right': [ (1,0,0), (1,1,0), (1,1,1), (1,0,1) ],            # +X (CCW from outside)
        'top':   [ (0,1,0), (0,1,1), (1,1,1), (1,1,0) ],            # +Y (CCW from outside)
        'bottom':[ (0,0,0), (1,0,0), (1,0,1), (0,0,1) ],            # -Y (CCW from outside)
    }

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Create OpenGL-enabled window
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((screen_width, screen_height), flags)
        pygame.display.set_caption("Pycraft - OpenGL 渲染")

        # Basic GL state
        glViewport(0, 0, screen_width, screen_height)
        glEnable(GL_DEPTH_TEST)
        # Re-enable face culling with correct front face orientation
        glEnable(GL_CULL_FACE)
        from OpenGL.GL import glFrontFace, GL_CCW
        glFrontFace(GL_CCW)  # Counter-clockwise vertices are front-facing
        glClearColor(135/255.0, 206/255.0, 235/255.0, 1.0)  # Sky blue

        self.fov = 70.0
        self.near = 0.1
        self.far = 100.0
        self.aspect = screen_width / screen_height

        # Stats for debug
        self.last_stats = {
            'faces': 0,
            'blocks': 0,
            'culled_blocks': 0,
        }
        
        # Occlusion cache to avoid repeated neighbor checks
        self._occlusion_cache = {}
        self._cache_frame = 0

        self._init_projection()

    # ------------------------------------------------------------------
    # GL setup helpers
    # ------------------------------------------------------------------
    def _init_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect, self.near, self.far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _set_camera(self, camera: Camera):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        pos = camera.position
        forward = camera.get_forward_vector()
        target = pos + forward
        up = camera.get_up_vector()
        # gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ)
        gluLookAt(pos[0], pos[1], pos[2], target[0], target[1], target[2], up[0], up[1], up[2])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_world(self, world: World, camera: Camera, performance_mode=True):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._init_projection()
        self._set_camera(camera)
        
        # Clear occlusion cache every few frames to prevent stale data
        self._cache_frame += 1
        if self._cache_frame % 60 == 0:  # Clear cache every 60 frames
            self._occlusion_cache.clear()

        camera_pos = camera.position
        render_distance = 2  # same as CPU default
        visible_chunks = world.get_visible_chunks(int(camera_pos[0]), int(camera_pos[2]), render_distance=render_distance)

        max_blocks = 6000 if performance_mode else 12000

        blocks_rendered = 0
        faces_rendered = 0
        culled_blocks = 0

        # Iterate chunks
        for chunk in visible_chunks:
            chunk_world_x = chunk.x * chunk.SIZE
            chunk_world_z = chunk.z * chunk.SIZE
            
            # Pre-calculate camera distance to chunk center for early chunk culling
            chunk_center_x = chunk_world_x + chunk.SIZE / 2
            chunk_center_z = chunk_world_z + chunk.SIZE / 2
            chunk_dist_sq = (chunk_center_x - camera_pos[0])**2 + (chunk_center_z - camera_pos[2])**2
            
            # Skip entire chunks that are too far away
            if chunk_dist_sq > (50 * 50):  # Increased from 40*40 for chunk-level culling
                culled_blocks += len(chunk.blocks)
                continue
            
            # Sort blocks by distance for better early termination
            block_distances = []
            for (lx, ly, lz), block in chunk.blocks.items():
                if not block.is_solid():
                    continue
                    
                wx = chunk_world_x + lx
                wy = ly
                wz = chunk_world_z + lz
                
                # Quick distance calculation (no sqrt for sorting)
                dist_sq = (wx - camera_pos[0])**2 + (wy - camera_pos[1])**2 + (wz - camera_pos[2])**2
                block_distances.append((dist_sq, wx, wy, wz, block))
            
            # Sort by distance (closest first) and limit processing
            block_distances.sort(key=lambda x: x[0])
            
            for dist_sq, wx, wy, wz, block in block_distances:
                # Distance culling with pre-calculated distance
                if dist_sq > 40 * 40:
                    culled_blocks += 1
                    continue
                
                # Visibility / occlusion: skip if completely surrounded by solid blocks
                if self._is_fully_occluded(world, wx, wy, wz):
                    culled_blocks += 1
                    continue
                    
                if blocks_rendered >= max_blocks:
                    break

                faces_rendered += self._render_block_faces(world, wx, wy, wz, block)
                blocks_rendered += 1
                
            if blocks_rendered >= max_blocks:
                break

        self.last_stats.update({
            'faces': faces_rendered,
            'blocks': blocks_rendered,
            'culled_blocks': culled_blocks,
        })

        # OpenGL rendering complete - now add pygame UI overlay
        # Read OpenGL buffer to pygame surface, add UI, then swap buffers
        self._draw_ui(world, camera)
        
        # Overlay elements
        self._draw_crosshair()

    # ------------------------------------------------------------------
    def _is_fully_occluded(self, world: World, x: int, y: int, z: int) -> bool:
        # Check cache first
        cache_key = (x, y, z)
        if cache_key in self._occlusion_cache:
            return self._occlusion_cache[cache_key]
        
        # Check 6 neighbors; if all solid -> occluded
        neighbors_solid = 0
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            if world.get_block(x+dx, y+dy, z+dz).is_solid():
                neighbors_solid += 1
            else:
                # Early exit - if any neighbor is not solid, not occluded
                self._occlusion_cache[cache_key] = False
                return False
        
        # All neighbors are solid
        is_occluded = (neighbors_solid == 6)
        self._occlusion_cache[cache_key] = is_occluded
        return is_occluded

    def _render_block_faces(self, world: World, x: int, y: int, z: int, block: Block) -> int:
        color = block.get_color()
        r, g, b = [c / 255.0 for c in color]
        faces = 0
        # Only draw faces that are exposed to air (or out of bounds)
        for face, (dx, dy, dz) in self.FACE_DELTAS.items():
            nx, ny, nz = x + dx, y + dy, z + dz
            neighbor = world.get_block(nx, ny, nz)
            if neighbor.is_solid():
                continue
            brightness = self.FACE_BRIGHTNESS.get(face, 0.8)
            glBegin(GL_QUADS)
            glColor3f(r * brightness, g * brightness, b * brightness)
            for vx, vy, vz in self.FACE_VERTICES[face]:
                glVertex3f(x + vx, y + vy, z + vz)
            glEnd()
            faces += 1
        return faces

    # ------------------------------------------------------------------
    # Overlay & UI
    # ------------------------------------------------------------------
    def _enter_2d(self):
        # Setup an orthographic projection for screen-space drawing
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

    def _leave_2d(self):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _draw_crosshair(self):
        self._enter_2d()
        cx = self.screen_width // 2
        cy = self.screen_height // 2
        size = 10
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        glEnd()
        self._leave_2d()

    def _draw_ui(self, world, camera):
        """Draw UI using OpenGL text rendering"""
        try:
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            
            # Switch to 2D rendering mode
            self._enter_2d()
            
            pos = camera.position
            lines = [
                f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                f"視角: pitch={camera.pitch:.2f}, yaw={camera.yaw:.2f}",
                "控制:",
                "WASD - 移動",
                "滑鼠 - 視角", 
                "空格/Shift - 上升/下降", 
                "左鍵 - 破壞 右鍵 - 放置", 
                "1-4 - 選擇方塊 Tab - 滑鼠",
                "ESC - 退出"
            ]
            
            # Draw UI text using OpenGL
            y = 10
            for i, text in enumerate(lines):
                size = 28 if i < 2 else 22
                color = (1.0, 1.0, 0.0) if i < 2 else (1.0, 1.0, 1.0)  # OpenGL color (0-1 range)
                font_mgr.render_text_opengl(text, 10, y, size, color)
                y += size + 4
            
            self._leave_2d()
            
        except Exception as e:
            # Fallback: print to console if OpenGL text fails
            pos = camera.position
            print(f"UI INFO: 位置=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) 視角=pitch={camera.pitch:.2f}, yaw={camera.yaw:.2f}")

    # Called by GameEngine when F3 debug mode is enabled
    def draw_debug_info(self, data: Dict):
        """Draw debug info using OpenGL text rendering"""
        try:
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            
            # Switch to 2D rendering mode
            self._enter_2d()
            
            lines = [
                f"FPS: {data.get('fps', 0):.1f}",
                f"Chunk: {data.get('chunk', (0,0))}",
                f"Chunks Loaded: {data.get('chunks_loaded', 0)}", 
                f"選擇方塊: {data.get('selected_block', '')}",
                f"Performance: {'ON' if data.get('performance_mode') else 'OFF'}", 
                f"Blocks: {self.last_stats['blocks']} Faces: {self.last_stats['faces']}"
            ]
            
            # Draw debug info on right side
            x = self.screen_width - 260
            font_mgr.render_text_list_opengl(lines, x, 10, line_height=30, size=24, color=(1.0, 1.0, 0.0))
            
            self._leave_2d()
            
        except Exception as e:
            # Fallback: console output
            print(f"[DEBUG] FPS={data.get('fps', 0):.1f} Chunk={data.get('chunk', (0,0))} "
                  f"Chunks_Loaded={data.get('chunks_loaded', 0)} Selected_Block={data.get('selected_block', '')} "
                  f"Performance={'ON' if data.get('performance_mode') else 'OFF'} "
                  f"Blocks={self.last_stats['blocks']} Faces={self.last_stats['faces']}")

    def cleanup(self):
        """Cleanup OpenGL resources"""
        try:
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            font_mgr.cleanup_textures()
        except Exception:
            pass


__all__ = ["GPURenderer"]
