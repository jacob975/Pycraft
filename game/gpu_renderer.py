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
from itertools import compress
from functools import lru_cache

from .world import World, Chunk
from .camera import Camera
from .blocks import Block

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

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize ModernGL rendering pipeline
        self._init_moderngl_context()
        self._create_shaders()
        self._create_geometry_buffers()
        self._setup_uniforms()

        self.fov = 70.0
        self.near = 0.1
        self.far = 100.0
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
        self.fov = 70.0
        self.near = 0.1
        self.far = 100.0
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
        fovy_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fovy_rad / 2.0)
        
        proj = np.array([
            [f / self.aspect, 0,  0,  0],
            [0, f,  0,  0],
            [0, 0, (self.far + self.near) / (self.near - self.far), (2 * self.far * self.near) / (self.near - self.far)],
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
        max_blocks = 409600 if performance_mode else 819200
        render_distance = 4
        
        # Get visible chunks using optimized culling
        visible_chunks = self._get_optimized_visible_chunks(world, camera, render_distance)
        st1 = time.time()
        # Batch process all blocks using NumPy
        block_data = self._prepare_block_data_batch(visible_chunks, camera, max_blocks, render_distance)
        st2 = time.time()
        if len(block_data['positions']) > 0:
            self._render_blocks_moderngl(block_data, camera)
        # Render UI elements
        st3 = time.time()
        self._render_ui_moderngl(world, camera)
        st4 = time.time()
        # Update performance stats
        print(f"Chunk culling: {(st1 - start_time)*1000:.2f} ms, Block prep: {(st2 - st1)*1000:.2f} ms, Block render: {(st3 - st2)*1000:.2f} ms, UI render: {(st4 - st3)*1000:.2f} ms")
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

    def _render_test_cube(self, camera: Camera):
        """Render a single test cube at a fixed position to debug geometry"""
        # Only render if we're close to spawn
        if np.linalg.norm(np.array(camera.position) - np.array([8, 30, 8])) > 20:
            return
            
        # Prepare matrices
        view_matrix = self._create_view_matrix(camera)
        
        # Set uniforms
        self.block_shader['projection_matrix'].write(self.projection_matrix.T.astype(np.float32).tobytes())
        self.block_shader['view_matrix'].write(view_matrix.T.astype(np.float32).tobytes())
        self.block_shader['light_dir'].write(np.array([0.2, -1.0, 0.3], dtype=np.float32).tobytes())
        self.block_shader['camera_pos'].write(np.array(camera.position, dtype=np.float32).tobytes())
        self.block_shader['fog_color'].write(np.array([0.529, 0.808, 0.922], dtype=np.float32).tobytes())
        
        # Create a single test cube at position (10, 25, 10) - should be visible and below camera
        test_position = np.array([[10.0, 25.0, 10.0]], dtype=np.float32)  
        test_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Bright red for visibility
        
        # Create instance data buffer
        instance_data = np.column_stack([test_position, test_color]).astype(np.float32)
        instance_buffer = self.ctx.buffer(instance_data.tobytes())
        
        # Create vertex array object
        vertex_attributes = [
            (self.cube_vbo, '3f 3f', 'position', 'normal'),
            (instance_buffer, '3f 3f/i', 'instance_pos', 'instance_color'),
        ]
        
        vao = self.ctx.vertex_array(self.block_shader, vertex_attributes, self.cube_ibo)
        
        # Render the test cube
        vao.render(instances=1)
        
        # Cleanup
        vao.release()
        instance_buffer.release()

    def _should_render_block(self, world, x: int, y: int, z: int) -> bool:
        """Check if a block should be rendered (has at least one exposed face)"""
        try:
            # Fast check - only check 4 horizontal neighbors for better performance
            # Skip expensive vertical neighbor checks unless necessary
            neighbors = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
            
            for nx, ny, nz in neighbors:
                neighbor = world.get_block(nx, ny, nz)
                if not neighbor.is_solid():  # Exposed to air or transparent block
                    return True
            
            # Only check vertical neighbors if all horizontal are solid
            vertical_neighbors = [(x, y+1, z), (x, y-1, z)]
            for nx, ny, nz in vertical_neighbors:
                neighbor = world.get_block(nx, ny, nz)
                if not neighbor.is_solid():
                    return True
                    
            return False  # Completely surrounded by solid blocks
        except Exception:
            # If we can't check neighbors, assume the block should be rendered
            return True
    
    def _prepare_block_data_batch(self, chunks: List[Chunk], camera: Camera, max_blocks: int, render_distance: int = 2) -> Dict:
        """Ultra-optimized batch processing for 10k+ blocks"""
        if not chunks:
            return {'positions': np.array([]), 'colors': np.array([]), 'types': np.array([])}

        camera_pos = np.array(camera.position, dtype=np.float32)
        max_distance_sq = (render_distance * 16 + 16) ** 2

        # Pre-allocate arrays for maximum efficiency
        estimated_blocks = len(chunks) * 100  # Rough estimate
        positions_buffer = np.zeros((estimated_blocks, 3), dtype=np.float32)
        colors_buffer = np.zeros((estimated_blocks, 3), dtype=np.float32)

        block_count = 0
        # Batch process chunks with minimal object creation
        for chunk in chunks:
            chunk_world_x = chunk.x * chunk.SIZE
            chunk_world_z = chunk.z * chunk.SIZE

            # Convert chunk blocks to arrays in one go
            if not chunk.blocks:
                continue

            # Extract positions and blocks in vectorized operations
            local_positions = np.array(list(chunk.blocks.keys()), dtype=np.int32)
            blocks_list = np.array(list(chunk.blocks.values()))

            # Convert to world coordinates (vectorized)
            world_positions = local_positions.astype(np.float32)
            world_positions[:, 0] += chunk_world_x
            world_positions[:, 2] += chunk_world_z

            # Filter solid blocks (vectorized)
            solid_mask = np.array([block.is_solid() for block in blocks_list])
            valid_positions = world_positions[solid_mask]
            valid_blocks = list(compress(blocks_list, solid_mask))

            if len(valid_positions) == 0:
                continue
            
            # Vectorized distance calculation
            distances = np.sum((valid_positions - camera_pos)**2, axis=1)
            distance_mask = distances <= max_distance_sq

            # Apply distance culling
            culled_positions = valid_positions[distance_mask]
            culled_blocks = list(compress(valid_blocks, distance_mask))

            # Add to buffers
            chunk_count = len(culled_positions)
            if block_count + chunk_count > estimated_blocks:
                # Resize buffers if needed
                new_size = (block_count + chunk_count) * 2
                positions_buffer = np.resize(positions_buffer, (new_size, 3))
                colors_buffer = np.resize(colors_buffer, (new_size, 3))

            # Batch color extraction (vectorized where possible)
            positions_buffer[block_count:block_count + chunk_count] = culled_positions

            # Vectorized color extraction
            if culled_blocks:
                # Extract all colors at once and convert to numpy array
                colors_list = [block.get_color() for block in culled_blocks]
                colors_array = np.array(colors_list, dtype=np.float32) / 255.0
                colors_buffer[block_count:block_count + chunk_count] = colors_array

            block_count += chunk_count

            if block_count >= max_blocks:
                break
            
        if block_count == 0:
            return {'positions': np.array([]), 'colors': np.array([]), 'types': np.array([])}

        # Trim arrays to actual size
        final_positions = positions_buffer[:block_count]
        final_colors = colors_buffer[:block_count]

        # Final distance-based sorting (if needed)
        if block_count > max_blocks:
            final_distances = np.sum((final_positions - camera_pos)**2, axis=1)
            sort_indices = np.argsort(final_distances)[:max_blocks]
            final_positions = final_positions[sort_indices]
            final_colors = final_colors[sort_indices]
            block_count = max_blocks

        return {
            'positions': final_positions,
            'colors': final_colors,
            'types': np.zeros(block_count, dtype=np.int32)
        }
    
    def _get_optimized_visible_chunks(self, world: World, camera: Camera, render_distance: int) -> List[Chunk]:
        # Get chunks with distance-based culling
        return world.get_visible_chunks(
            int(camera.position[0]), int(camera.position[2]), 
            render_distance=render_distance
        )

    # ------------------------------------------------------------------
    #@lru_cache(maxsize=32768)  # Reduced size for actual usage pattern
    def _is_fully_occluded_cached(self, x: int, y: int, z: int) -> bool:
        """
        Fast cached occlusion check for static block positions.
        Cache is cleared when world is modified via clear_occlusion_cache().
        """
        return self._is_fully_occluded_internal(x, y, z)

    def _is_fully_occluded_internal(self, x: int, y: int, z: int) -> bool:
        """Internal occlusion check without caching"""
        # This will be set by the caller with the world reference
        world = getattr(self, '_current_world', None)
        if world is None:
            return False
            
        # Check 6 neighbors; if all solid -> occluded
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            if not world.get_block(x+dx, y+dy, z+dz).is_solid():
                return False
        return True

    # Removed problematic distance caching functions - camera position changes every frame!
    # Use vectorized NumPy calculations instead for better performance

    def _is_fully_occluded(self, world: World, x: int, y: int, z: int) -> bool:
        # Set world reference for internal method
        self._current_world = world
        
        # Use simplified caching - caller should clear cache when world changes
        return self._is_fully_occluded_cached(x, y, z)

    #@lru_cache(maxsize=256)  # Small cache for limited block types
    def _get_normalized_color(self, block_type: str) -> Tuple[float, float, float]:
        """Get normalized RGB color for a block type"""
        # This would need to be implemented based on your block system
        # For now, return a placeholder that works with existing code
        from .blocks import BlockType
        if hasattr(BlockType, block_type):
            color = getattr(BlockType, block_type).value.get_color()
        else:
            color = (128, 128, 128)  # Default gray
        return tuple(c / 255.0 for c in color)
    
    # Removed problematic neighbor caching - world state tracking is insufficient
    # Direct world.get_block() calls are more reliable and still reasonably fast



    def _render_ui_moderngl(self, world: World, camera: Camera):
        """Render UI elements using ModernGL"""
        # Render crosshair
        self._draw_crosshair_moderngl()
        
        # Render UI text (position, controls, etc)
        self._draw_ui_text_moderngl(world, camera)
    
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

    def _draw_ui_text_moderngl(self, world: World, camera: Camera):
        """Draw UI text using ModernGL (simplified for now)"""
        # For now, we'll use pygame surfaces and upload as textures
        # This is a simplified approach - a full implementation would use
        # bitmap fonts or text rendering shaders
        try:
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            
            pos = camera.position
            debug_text = f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | " + \
                        f"Pitch: {camera.pitch:.1f}, Yaw: {camera.yaw:.1f} | " + \
                        f"Blocks: {self.last_stats['blocks']} | " + \
                        f"FPS: {1000.0 / max(self.last_stats['render_time_ms'], 1):.1f}"
            
            # For now, print to console instead of rendering to screen
            # TODO: Implement proper text rendering with ModernGL
            if hasattr(self, '_last_debug_print_time'):
                if time.time() - self._last_debug_print_time > 1.0:  # Print once per second
                    print(f"\r{debug_text}", end='')
                    self._last_debug_print_time = time.time()
            else:
                self._last_debug_print_time = time.time()
            
        except Exception as e:
            pass  # Silently fail for UI rendering

    #@lru_cache(maxsize=128)  # Small cache for debug strings
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
        """Draw debug info using ModernGL (simplified console output for now)"""
        try:
            # For now, use console output instead of screen rendering
            # TODO: Implement proper text rendering with ModernGL
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
            
        except Exception as e:
            pass  # Silently fail for debug rendering

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
