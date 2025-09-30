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

import time
from typing import Dict, Tuple, List, Optional
import numpy as np

from .world import World, Chunk
from .camera import Camera
from .blocks import Block
from config import *

import pygame

# ModernGL is required for this renderer
try:
    import moderngl as mgl
    import moderngl_window as mglw
    MODERNGL_AVAILABLE = True
except ImportError:
    raise ImportError("ModernGL is required for GPU rendering. Install with: pip install moderngl moderngl-window")

from .blocks import BlockType
from .font_manager import get_font_manager

# High-performance functions using pure NumPy vectorization
def calculate_distances_vectorized(block_positions: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
    """Vectorized distance calculation using pure NumPy for high performance"""
    diff = block_positions - camera_pos
    return np.sum(diff * diff, axis=1)

def frustum_cull_blocks(block_positions: np.ndarray, frustum_planes: np.ndarray) -> np.ndarray:
    """Fast frustum culling using vectorized NumPy operations"""
    visible = np.ones(len(block_positions), dtype=bool)
    
    # Add block center offset for more accurate culling
    block_centers = block_positions + 0.5
    
    # Vectorized frustum culling - much faster than loops
    for i in range(len(frustum_planes)):
        plane = frustum_planes[i]
        # Calculate distance from all block centers to this plane
        distances = np.dot(block_centers, plane[:3]) + plane[3]
        # Mark blocks as invisible if they're completely behind this plane
        # Use a tolerance of -0.866 (roughly sqrt(3)/2) to account for block size
        visible &= distances >= -0.866  # Block diagonal consideration
    
    return visible

def sort_blocks_by_distance(distances: np.ndarray) -> np.ndarray:
    """Fast sorting of blocks by distance using NumPy's optimized argsort"""
    return np.argsort(distances)

def check_occlusion_batch(positions: np.ndarray, neighbor_data: np.ndarray) -> np.ndarray:
    """Vectorized batch occlusion checking for multiple blocks"""
    # Vectorized approach: sum solid neighbors for all blocks at once
    solid_neighbor_counts = np.sum(neighbor_data, axis=1)
    # A block is occluded if all 6 neighbors are solid
    occluded = solid_neighbor_counts == 6
    return occluded

class GPURenderer:
    """Modern GPU renderer using ModernGL for all rendering operations."""

    def __init__(self, screen_width: int, screen_height: int, existing_screen: pygame.Surface = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.existing_screen = existing_screen
        
        # Initialize ModernGL rendering pipeline
        self._init_moderngl_context()
        self._create_shaders()
        self._create_geometry_buffers()
        self._setup_uniforms()
        self.aspect = screen_width / screen_height

        # Enhanced stats for debug and performance tracking
        self.last_stats = {
            'faces': 0,
            'blocks': 0,
            'culled_blocks': 0,
            'render_time_ms': 0.0,
            'frames_rendered': 0,
        }
        
        # Pre-allocated numpy arrays for batch processing
        self.max_blocks = 65536
        self.block_positions = np.zeros((self.max_blocks, 3), dtype=np.float32)
        self.block_colors = np.zeros((self.max_blocks, 3), dtype=np.float32)
        self.block_types = np.zeros(self.max_blocks, dtype=np.int32)
        
        # Frustum planes for culling
        self.frustum_planes = np.zeros((6, 4), dtype=np.float32)
        
        # Performance flags - always true for ModernGL-only renderer
        self.use_moderngl = True
        self.use_numpy_optimization = True
            
        print(f"🚀 ModernGL GPU Renderer initialized - Screen: {self.screen_width}x{self.screen_height}")

    # ------------------------------------------------------------------
    # Initialization methods
    # ------------------------------------------------------------------
    def _create_shaders(self):
        """Create comprehensive shader programs for different rendering needs"""
        # Main block rendering shader with instancing
        vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec3 instance_pos;
        layout(location = 3) in vec3 instance_color;
        
        uniform mat4 projection_matrix;
        uniform mat4 view_matrix;
        uniform vec3 light_dir;
        uniform vec3 camera_pos;
        
        out vec3 color;
        out float fog_factor;
        
        void main() {
            vec3 world_pos = position + instance_pos;
            gl_Position = projection_matrix * view_matrix * vec4(world_pos, 1.0);
            
            // Simplified lighting for better performance
            float brightness = max(0.6, abs(dot(normal, normalize(-light_dir))));
            color = instance_color * brightness;
            
            // Simplified fog calculation
        float distance = length(world_pos - camera_pos);
        fog_factor = clamp(1.0 - (distance - 40.0) / 60.0, 0.0, 1.0);
    }
        '''
        
        fragment_shader = '''
        #version 330 core
        
        in vec3 color;
        in float fog_factor;
        
        uniform vec3 fog_color;
        
        out vec4 fragColor;
        
        void main() {
            vec3 final_color = mix(fog_color, color, fog_factor);
            fragColor = vec4(final_color, 1.0);
        }
        '''
        
        self.block_shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # UI shader for crosshair and text
        ui_vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 color;
        
        uniform mat4 ortho_matrix;
        
        out vec3 vertex_color;
        
        void main() {
            gl_Position = ortho_matrix * vec4(position, 0.0, 1.0);
            vertex_color = color;
        }
        '''
        
        ui_fragment_shader = '''
        #version 330 core
        
        in vec3 vertex_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(vertex_color, 1.0);
        }
        '''
        
        self.ui_shader = self.ctx.program(
            vertex_shader=ui_vertex_shader,
            fragment_shader=ui_fragment_shader
        )
        
        print("✅ Shaders created successfully")
    
    def _create_geometry_buffers(self):
        """Create optimized geometry buffers for instanced rendering"""
        # Create cube vertices with normals (unit cube from -0.5 to 0.5)
        vertices = np.array([
            # Front face (z = 0.5)
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  # bottom-left
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  # bottom-right
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  # top-right
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  # top-left
            
            # Back face (z = -0.5)
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            
            # Left face (x = -0.5)
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
            
            # Right face (x = 0.5)
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
            
            # Top face (y = 0.5)
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
            
            # Bottom face (y = -0.5)
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
        ], dtype=np.float32)
        
        # Cube indices for triangulated faces (counter-clockwise winding)
        indices = np.array([
            0, 2, 1,  0, 3, 2,    # Front (CCW)
            4, 6, 5,  4, 7, 6,    # Back (CCW)
            8, 10, 9,  8, 11, 10,  # Left (CCW)
            12, 14, 13, 12, 15, 14, # Right (CCW)
            16, 18, 17, 16, 19, 18, # Top (CCW)
            20, 22, 21, 20, 23, 22  # Bottom (CCW)
        ], dtype=np.uint32)
        
        # Create buffers
        self.cube_vbo = self.ctx.buffer(vertices.tobytes())
        self.cube_ibo = self.ctx.buffer(indices.tobytes())
        
        # Create crosshair geometry
        crosshair_vertices = np.array([
            # Horizontal line
            -10, 0, 1.0, 1.0, 1.0,
             10, 0, 1.0, 1.0, 1.0,
            # Vertical line
             0, -10, 1.0, 1.0, 1.0,
             0,  10, 1.0, 1.0, 1.0,
        ], dtype=np.float32)
        
        self.crosshair_vbo = self.ctx.buffer(crosshair_vertices.tobytes())
        
        print("✅ Geometry buffers created")
    
    def _setup_uniforms(self):
        """Setup uniform locations and initial values"""
        self.aspect = self.screen_width / self.screen_height
        
        # Pre-calculate projection matrix
        self.projection_matrix = self._create_projection_matrix()
        
        # Enhanced stats for debug and performance tracking
        self.last_stats = {
            'faces': 0,
            'blocks': 0,
            'culled_blocks': 0,
            'render_time_ms': 0.0,
            'frames_rendered': 0,
        }
        
        # Pre-allocated numpy arrays for batch processing
        self.max_blocks = 65536
        self.block_positions = np.zeros((self.max_blocks, 3), dtype=np.float32)
        self.block_colors = np.zeros((self.max_blocks, 3), dtype=np.float32)
        self.block_types = np.zeros(self.max_blocks, dtype=np.int32)
        
        # Frustum planes for culling
        self.frustum_planes = np.zeros((6, 4), dtype=np.float32)
        
        # Performance flags
        self.use_moderngl = True  # Always true now
        self.use_numpy_optimization = True  # Always available with NumPy
        
        print(f"🚀 ModernGL GPU Renderer initialized - Screen: {self.screen_width}x{self.screen_height}")
    
    def _init_moderngl_context(self):
        """Initialize ModernGL context and pygame window"""
        if self.existing_screen is not None:
            # Reuse existing pygame screen but switch to OpenGL mode
            # We need to recreate the surface with OpenGL flags
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)
            pygame.display.set_caption("Pycraft - ModernGL GPU Renderer")
        else:
            # Create OpenGL-enabled window for ModernGL with proper depth buffer
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            # Request depth buffer
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)
            pygame.display.set_caption("Pycraft - ModernGL GPU Renderer")
        
        # Create ModernGL context
        self.ctx = mgl.create_context()
        
        # Enable depth testing but disable face culling to see all faces
        self.ctx.enable(mgl.DEPTH_TEST)
        # Face culling disabled to debug rendering issues
        # self.ctx.enable(mgl.CULL_FACE)
        
        print("✅ ModernGL context initialized")
    
    def _create_projection_matrix(self):
        """Create perspective projection matrix"""
        fovy_rad = np.radians(FOV)
        f = 1.0 / np.tan(fovy_rad / 2.0)
        
        proj = np.array([
            [f / self.aspect, 0,  0,  0],
            [0, f,  0,  0],
            [0, 0, (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE), (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE)],
            [0, 0, -1,  0]
        ], dtype=np.float32)
        
        return proj
    
    def _create_view_matrix(self, camera: Camera):
        """Create view matrix from camera"""
        pos = np.array(camera.position, dtype=np.float32)
        forward = np.array(camera.get_forward_vector(), dtype=np.float32)
        up = np.array(camera.get_up_vector(), dtype=np.float32)
        
        # Debug: Print camera vectors occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        # Calculate camera basis vectors
        z_axis = -forward  # Camera looks down -Z
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Create view matrix
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[0, 3] = -np.dot(x_axis, pos)
        view[1, 3] = -np.dot(y_axis, pos)
        view[2, 3] = -np.dot(z_axis, pos)
        
        return view

    def _calculate_frustum_planes(self, view_matrix: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        """Calculate 6 frustum planes from view and projection matrices
        Each plane is represented as (a, b, c, d) where ax + by + cz + d = 0
        """
        # Combine view and projection matrices
        mvp = projection_matrix @ view_matrix
        
        # Extract frustum planes from the combined matrix
        # Each plane: [a, b, c, d] where ax + by + cz + d = 0
        planes = np.zeros((6, 4), dtype=np.float32)
        
        # Left plane: mvp[3] + mvp[0]
        planes[0] = mvp[3, :] + mvp[0, :]
        # Right plane: mvp[3] - mvp[0]
        planes[1] = mvp[3, :] - mvp[0, :]
        # Bottom plane: mvp[3] + mvp[1]
        planes[2] = mvp[3, :] + mvp[1, :]
        # Top plane: mvp[3] - mvp[1]
        planes[3] = mvp[3, :] - mvp[1, :]
        # Near plane: mvp[3] + mvp[2]
        planes[4] = mvp[3, :] + mvp[2, :]
        # Far plane: mvp[3] - mvp[2]
        planes[5] = mvp[3, :] - mvp[2, :]
        
        # Normalize planes
        for i in range(6):
            norm = np.linalg.norm(planes[i, :3])
            if norm > 0:
                planes[i] /= norm
        
        return planes

    # ------------------------------------------------------------------
    # ModernGL Rendering Pipeline
    # ------------------------------------------------------------------
    def render_world(self, world: World, camera: Camera, performance_mode=True):
        """High-performance world rendering with ModernGL"""
        start_time = time.time()
        
        # Store world reference for face culling
        self._current_world = world
        
        # Clear screen with sky blue color and depth buffer
        self.ctx.clear(0.529, 0.808, 0.922, 1.0, 1.0)  # Clear color and depth in one call
        
        # Ensure proper GL state
        self.ctx.enable(mgl.DEPTH_TEST)
        
        # Set viewport to match screen size
        self.ctx.viewport = (0, 0, self.screen_width, self.screen_height)
        
        # Configure depth test (ModernGL uses string values)
        self.ctx.depth_func = '<'  # Less than comparison
        
        # Determine render limits based on performance mode 
        max_blocks = MAX_BLOCKS if performance_mode else MAX_BLOCKS * 2
        
        # Get visible chunks using optimized culling
        visible_chunks = self._get_optimized_visible_chunks(world, camera, RENDER_DISTANCE)
        # Batch process all blocks using NumPy
        block_data = self._prepare_block_data(visible_chunks, camera, max_blocks)
        if len(block_data['positions']) > 0:
            self._render_blocks_moderngl(block_data, camera)
        # Render UI elements
        self._render_ui_moderngl(world, camera)
        # Update performance stats
        render_time = (time.time() - start_time) * 1000
        self.last_stats['render_time_ms'] = render_time
        self.last_stats['frames_rendered'] += 1
    
    def _render_blocks_moderngl(self, block_data: Dict, camera: Camera):
        """Render blocks using ModernGL instanced rendering"""
        if len(block_data['positions']) == 0:
            return
        
        # Prepare matrices
        view_matrix = self._create_view_matrix(camera)
        
        # Set uniforms (ModernGL automatically binds the program when setting uniforms)
        self.block_shader['projection_matrix'].write(self.projection_matrix.T.astype(np.float32).tobytes())
        self.block_shader['view_matrix'].write(view_matrix.T.astype(np.float32).tobytes())
        self.block_shader['light_dir'].write(np.array([0.2, -1.0, 0.3], dtype=np.float32).tobytes())
        self.block_shader['camera_pos'].write(np.array(camera.position, dtype=np.float32).tobytes())
        self.block_shader['fog_color'].write(np.array([0.529, 0.808, 0.922], dtype=np.float32).tobytes())
        
        # Create instance data buffer (positions + colors)
        instance_data = np.column_stack([
            block_data['positions'],
            block_data['colors']
        ]).astype(np.float32)
        
        # Create instance buffer
        instance_buffer = self.ctx.buffer(instance_data.tobytes())
        
        # Create vertex array object with shader program
        vertex_attributes = [
            (self.cube_vbo, '3f 3f', 'position', 'normal'),
            (instance_buffer, '3f 3f/i', 'instance_pos', 'instance_color'),
        ]
        
        vao = self.ctx.vertex_array(self.block_shader, vertex_attributes, self.cube_ibo)
        
        # Render all instances
        vao.render(instances=len(block_data['positions']))
        
        # Cleanup
        vao.release()
        instance_buffer.release()
        
        self.last_stats['faces'] = len(block_data['positions']) * 6  # 6 faces per block
        self.last_stats['blocks'] = len(block_data['positions'])
    
    def _prepare_block_data(self, chunks: List[Chunk], camera: Camera, max_blocks: int) -> Dict:
        """Ultra-optimized batch processing for 10k+ blocks using NumPy arrays"""
        if not chunks:
            return {'positions': np.array([]), 'colors': np.array([]), 'types': np.array([])}
        
        # Pre-allocate lists for concatenation
        all_positions = []
        all_colors = []
        all_types = []
        total_blocks = 0
        
        # Batch process chunks with minimal object creation
        for chunk in chunks:
            if not chunk.blocks:
                continue
            
            # Get optimized visible faces data
            visible_data = chunk.get_visible_faces()
            
            if len(visible_data['positions']) > 0:
                all_positions.append(visible_data['positions'])
                all_colors.append(visible_data['colors'])
                all_types.append(visible_data['types'])
            
            total_blocks += len(chunk.blocks)

        # Efficiently concatenate all arrays
        if all_positions:
            final_positions = np.concatenate(all_positions, axis=0)[:max_blocks]
            final_colors = np.concatenate(all_colors, axis=0)[:max_blocks]
            final_types = np.concatenate(all_types, axis=0)[:max_blocks]
        else:
            final_positions = np.empty((0, 3), dtype=np.float32)
            final_colors = np.empty((0, 3), dtype=np.float32)
            final_types = np.empty(0, dtype=object)

        # Update stats
        visible_block_count = len(final_positions)
        self.last_stats['blocks'] = visible_block_count
        self.last_stats['culled_blocks'] = max(0, total_blocks - visible_block_count)

        return {
            'positions': final_positions,
            'colors': final_colors,
            'types': final_types
        }

    def _get_optimized_visible_chunks(self, world: World, camera: Camera, render_distance: int) -> List[Chunk]:
        # Get chunks with distance-based culling
        return world.get_visible_chunks(
            int(camera.position[0]), int(camera.position[2]), 
            render_distance=render_distance,
            to_create=False  # Only get already loaded chunks
        )

    def _render_ui_moderngl(self, world: World, camera: Camera):
        """Render UI elements using ModernGL"""
        # Render crosshair
        self._draw_crosshair_moderngl()
        
        # TODO: Render UI text (position, controls, etc)
    
    def _draw_crosshair_moderngl(self):
        """Draw crosshair using ModernGL"""
        # Create orthographic projection matrix for UI
        ortho_matrix = np.array([
            [2.0 / self.screen_width, 0, 0, -1],
            [0, -2.0 / self.screen_height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Set UI shader uniforms (automatically binds program)
        self.ui_shader['ortho_matrix'].write(ortho_matrix.T.astype(np.float32).tobytes())
        
        # Disable depth testing for UI
        self.ctx.disable(mgl.DEPTH_TEST)
        
        # Center crosshair position
        cx, cy = self.screen_width // 2, self.screen_height // 2
        size = 10
        
        # Update crosshair vertices
        crosshair_vertices = np.array([
            # Horizontal line
            cx - size, cy, 1.0, 1.0, 1.0,
            cx + size, cy, 1.0, 1.0, 1.0,
            # Vertical line
            cx, cy - size, 1.0, 1.0, 1.0,
            cx, cy + size, 1.0, 1.0, 1.0,
        ], dtype=np.float32)
        
        # Update buffer and render
        self.crosshair_vbo.write(crosshair_vertices.tobytes())
        
        crosshair_vao = self.ctx.vertex_array(
            self.ui_shader, 
            [(self.crosshair_vbo, '2f 3f', 'position', 'color')]
        )
        
        # Render as lines
        crosshair_vao.render(mode=mgl.LINES)
        crosshair_vao.release()
        
        # Re-enable depth testing
        self.ctx.enable(mgl.DEPTH_TEST)

    def _create_text_shader(self):
        """Create shader program for text rendering"""
        text_vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        
        uniform mat4 ortho_matrix;
        
        out vec2 uv;
        
        void main() {
            gl_Position = ortho_matrix * vec4(position, 0.0, 1.0);
            uv = texcoord;
        }
        '''
        
        text_fragment_shader = '''
        #version 330 core
        
        in vec2 uv;
        uniform sampler2D text_texture;
        uniform vec3 text_color;
        
        out vec4 fragColor;
        
        void main() {
            vec4 sampled = texture(text_texture, uv);
            fragColor = vec4(text_color, sampled.a);
        }
        '''
        
        self.text_shader = self.ctx.program(
            vertex_shader=text_vertex_shader,
            fragment_shader=text_fragment_shader
        )

    def _render_text_texture(self, text: str, x: int, y: int, font_size: int = 24, color: tuple = (255, 255, 255)):
        """Render text as texture using ModernGL (based on example_pygame_text.py)"""
        try:
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            
            # Create pygame surface with text
            font = font_mgr.get_font(font_size)
            img = font.render(text, True, color)
            w, h = img.get_size()
            
            if w == 0 or h == 0:
                return
            
            # Generate texture
            texture = self.ctx.texture((w, h), 4)  # RGBA format
            texture.filter = (mgl.NEAREST, mgl.NEAREST)
            
            # Convert pygame surface to texture data
            data = pygame.image.tostring(img, "RGBA", True)  # Flip vertically
            texture.write(data)
            
            # Create quad vertices for text rendering
            vertices = np.array([
                # Position  # TexCoords
                x,     y,     0.0, 1.0,  # Top-left
                x + w, y,     1.0, 1.0,  # Top-right
                x + w, y + h, 1.0, 0.0,  # Bottom-right
                x,     y + h, 0.0, 0.0,  # Bottom-left
            ], dtype=np.float32)
            
            indices = np.array([
                0, 1, 2,  # First triangle
                0, 2, 3   # Second triangle
            ], dtype=np.uint32)
            
            # Create buffers
            vbo = self.ctx.buffer(vertices.tobytes())
            ibo = self.ctx.buffer(indices.tobytes())
            
            # Set up orthographic projection
            ortho_matrix = np.array([
                [2.0 / self.screen_width, 0, 0, -1],
                [0, -2.0 / self.screen_height, 0, 1],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            # Use text shader and set uniforms
            self.text_shader['ortho_matrix'].write(ortho_matrix.T.astype(np.float32).tobytes())
            self.text_shader['text_color'].write(np.array([c/255.0 for c in color[:3]], dtype=np.float32).tobytes())
            
            # Bind texture
            texture.use(0)
            self.text_shader['text_texture'].value = 0
            
            # Create VAO and render
            vao = self.ctx.vertex_array(
                self.text_shader,
                [(vbo, '2f 2f', 'position', 'texcoord')],
                ibo
            )
            
            vao.render()
            
            # Cleanup
            vao.release()
            vbo.release()
            ibo.release()
            texture.release()
            
        except Exception as e:
            print(f"Text rendering error: {e}")
            pass

    def _format_debug_line(self, key: str, value, format_type: str = 'default') -> str:
        """Cache formatted debug strings to reduce string operations"""
        if format_type == 'fps':
            return f"FPS: {value:.1f}"
        elif format_type == 'chunk':
            return f"Chunk: {value}"
        elif format_type == 'chunks_loaded':
            return f"Chunks Loaded: {value}"
        elif format_type == 'selected_block':
            return f"選擇方塊: {value}"
        elif format_type == 'performance':
            return f"Performance: {'ON' if value else 'OFF'}"
        elif format_type == 'blocks_faces':
            return f"Blocks: {value['blocks']} Faces: {value['faces']}"
        else:
            return f"{key}: {value}"

    # Called by GameEngine when F3 debug mode is enabled
    def draw_debug_info(self, data: Dict):
        """Draw debug info using ModernGL texture-based text rendering"""
        try:
            # Disable depth testing for UI rendering
            self.ctx.disable(mgl.DEPTH_TEST)
            self.ctx.enable(mgl.BLEND)
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            
            # Create text shader if not exists
            if not hasattr(self, 'text_shader'):
                self._create_text_shader()
            
            # Render debug information on screen
            y_offset = 10
            line_height = 25
            
            # FPS
            fps_text = self._format_debug_line('fps', data.get('fps', 0), 'fps')
            self._render_text_texture(fps_text, 10, y_offset, font_size=20, color=(255, 255, 0))
            y_offset += line_height
            
            # Position
            pos = data.get('position', (0, 0, 0))
            pos_text = f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            self._render_text_texture(pos_text, 10, y_offset, font_size=20, color=(255, 255, 255))
            y_offset += line_height
            
            # Chunk info
            chunk_text = self._format_debug_line('chunk', data.get('chunk', (0,0)), 'chunk')
            self._render_text_texture(chunk_text, 10, y_offset, font_size=20, color=(255, 255, 255))
            y_offset += line_height
            
            # Chunks loaded
            chunks_text = self._format_debug_line('chunks_loaded', data.get('chunks_loaded', 0), 'chunks_loaded')
            self._render_text_texture(chunks_text, 10, y_offset, font_size=20, color=(255, 255, 255))
            y_offset += line_height
            
            # Selected block
            block_text = self._format_debug_line('selected_block', data.get('selected_block', ''), 'selected_block')
            self._render_text_texture(block_text, 10, y_offset, font_size=20, color=(255, 255, 255))
            y_offset += line_height
            
            # Performance mode
            perf_text = self._format_debug_line('performance', data.get('performance_mode'), 'performance')
            self._render_text_texture(perf_text, 10, y_offset, font_size=20, color=(255, 255, 255))
            y_offset += line_height
            
            # Block and face count
            stats_text = self._format_debug_line('blocks_faces', self.last_stats, 'blocks_faces')
            self._render_text_texture(stats_text, 10, y_offset, font_size=20, color=(0, 255, 255))
            
            # Re-enable depth testing
            self.ctx.disable(mgl.BLEND)
            self.ctx.enable(mgl.DEPTH_TEST)
            
        except Exception as e:
            # Fallback to console output if texture rendering fails
            debug_lines = [
                self._format_debug_line('fps', data.get('fps', 0), 'fps'),
                self._format_debug_line('chunk', data.get('chunk', (0,0)), 'chunk'),
                self._format_debug_line('chunks_loaded', data.get('chunks_loaded', 0), 'chunks_loaded'),
                self._format_debug_line('selected_block', data.get('selected_block', ''), 'selected_block'),
                self._format_debug_line('performance', data.get('performance_mode'), 'performance'),
                self._format_debug_line('blocks_faces', self.last_stats, 'blocks_faces')
            ]
            
            # Print debug info periodically
            if hasattr(self, '_last_debug_time'):
                if time.time() - self._last_debug_time > 0.5:  # Every 0.5 seconds
                    print(f"\r[DEBUG] {' | '.join(debug_lines)}", end='')
                    self._last_debug_time = time.time()
            else:
                self._last_debug_time = time.time()

# Factory function to create the best available renderer
def create_best_renderer(screen_width: int, screen_height: int):
    """Create the highest performance renderer available"""
    try:
        renderer = GPURenderer(screen_width, screen_height)
        return renderer
    except Exception as e:
        print(f"⚠️ High performance GPU renderer failed: {e}")
        raise

__all__ = ["GPURenderer", "create_best_renderer"]
