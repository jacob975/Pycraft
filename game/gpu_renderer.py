"""
OpenGL-based GPU renderer for Pycraft.

This implementation uses PyOpenGL with a legacy immediate-mode pipeline
for simplicity and keeps a stable renderer API for `engine.GameEngine`.

Key features:
    * Frustum-limited chunk selection
  * Per-face visibility (only draw faces exposed to air)
    * Simple brightness shading per face
  * Crosshair + UI & debug overlay using pygame font surfaces
    * Clear startup errors if OpenGL/ModernGL initialization fails

Notes:
  This is intentionally simple (no VBO batching yet). For performance
  improvements, a future pass can build meshed chunk VBOs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path
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

    def __init__(self, screen_width: int, screen_height: int, existing_screen: Optional[pygame.Surface] = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.existing_screen = existing_screen

        self.max_blocks = 65536
        self.max_instances = self.max_blocks * 6

        self._cached_visible_chunks: List[Chunk] = []
        self._cached_chunk_center: Optional[Tuple[int, int]] = None
        self._cached_world_chunk_count = -1
        self._cached_render_distance = -1
        self.render_distance_chunks = int(RENDER_DISTANCE)

        # Cache debug text textures to avoid rebuilding GPU resources every frame.
        self._text_texture_cache: Dict[Tuple[str, int, Tuple[int, int, int]], Tuple[mgl.Texture, int, int, float]] = {}
        self._last_debug_text_update = 0.0
        self._debug_text_update_interval = 0.2
        self._debug_text_lines: List[Tuple[str, Tuple[int, int, int]]] = []
        self._last_text_cache_prune = 0.0
        self._text_cache_ttl_seconds = 3.0
        self._character_part_specs: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
            'head': ((0.0, 1.55, 0.0), (0.50, 0.50, 0.50)),
            'torso': ((0.0, 1.00, 0.0), (0.70, 0.60, 0.35)),
            'left_arm': ((-0.48, 1.00, 0.0), (0.20, 0.62, 0.20)),
            'right_arm': ((0.48, 1.00, 0.0), (0.20, 0.62, 0.20)),
            'left_leg': ((-0.18, 0.35, 0.0), (0.24, 0.70, 0.24)),
            'right_leg': ((0.18, 0.35, 0.0), (0.24, 0.70, 0.24)),
        }
        self.character_textures: Dict[str, mgl.Texture] = {}
        
        # Initialize ModernGL rendering pipeline
        self._init_moderngl_context()
        self._create_shaders()
        self._create_geometry_buffers()
        self._setup_uniforms()
        self.aspect = screen_width / screen_height

    # ------------------------------------------------------------------
    # Initialization methods
    # ------------------------------------------------------------------
    def _create_shaders(self):
        """Create comprehensive shader programs for different rendering needs"""
        # Main face rendering shader with instancing
        vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec2 quad_pos;
        layout(location = 1) in vec3 instance_pos;
        layout(location = 2) in vec3 instance_color;
        layout(location = 3) in float instance_face;
        layout(location = 4) in float instance_texture_layer;
        
        uniform mat4 projection_matrix;
        uniform mat4 view_matrix;
        uniform vec3 light_dir;
        uniform vec3 camera_pos;
        uniform float grass_width;
        
        out vec3 color;
        out float fog_factor;
        out vec2 uv;
        flat out float texture_layer;
        
        vec3 rotate_y(vec3 v, float a) {
            float c = cos(a);
            float s = sin(a);
            return vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
        }

        void get_axis_face_basis(int local_face, out vec3 normal, out vec3 tangent, out vec3 bitangent) {
            if (local_face == 0) { normal = vec3(0.0, 0.0, -1.0); tangent = vec3(-1.0, 0.0, 0.0); bitangent = vec3(0.0, 1.0, 0.0); }
            else if (local_face == 1) { normal = vec3(0.0, 0.0, 1.0); tangent = vec3(1.0, 0.0, 0.0); bitangent = vec3(0.0, 1.0, 0.0); }
            else if (local_face == 2) { normal = vec3(1.0, 0.0, 0.0); tangent = vec3(0.0, 0.0, -1.0); bitangent = vec3(0.0, 1.0, 0.0); }
            else if (local_face == 3) { normal = vec3(-1.0, 0.0, 0.0); tangent = vec3(0.0, 0.0, 1.0); bitangent = vec3(0.0, 1.0, 0.0); }
            else if (local_face == 4) { normal = vec3(0.0, 1.0, 0.0); tangent = vec3(1.0, 0.0, 0.0); bitangent = vec3(0.0, 0.0, -1.0); }
            else { normal = vec3(0.0, -1.0, 0.0); tangent = vec3(1.0, 0.0, 0.0); bitangent = vec3(0.0, 0.0, 1.0); }
        }

        void main() {
            int face_id = int(instance_face + 0.5);
            vec3 normal;
            vec3 tangent;
            vec3 bitangent;
            vec3 center;
            vec3 world_pos;

            if (face_id <= 5) {
                get_axis_face_basis(face_id, normal, tangent, bitangent);
                center = instance_pos + (normal * 0.5);
                world_pos = center + tangent * quad_pos.x + bitangent * quad_pos.y;
                uv = quad_pos + vec2(0.5, 0.5);
            } else {
                int encoded = face_id - 6;
                int height_variant = encoded / 2;
                int plane_idx = encoded % 2;

                float grass_height = 1.0;
                if (height_variant == 1) grass_height = 0.66;
                else if (height_variant == 2) grass_height = 0.33;

                float half_height = grass_height * 0.5;
                float angle = (plane_idx == 0) ? 0.78539816339 : -0.78539816339;
                normal = rotate_y(vec3(0.0, 0.0, 1.0), angle);
                tangent = rotate_y(vec3(1.0, 0.0, 0.0), angle);
                bitangent = vec3(0.0, 1.0, 0.0);

                center = instance_pos + vec3(0.0, -0.5 + half_height, 0.0);
                world_pos = center + tangent * (quad_pos.x * grass_width) + bitangent * (quad_pos.y * grass_height);
                uv = quad_pos + vec2(0.5, 0.5);
            }

            gl_Position = projection_matrix * view_matrix * vec4(world_pos, 1.0);
            
            // Simplified lighting for better performance
            float brightness = max(0.6, abs(dot(normal, normalize(-light_dir))));
            if (instance_texture_layer >= 0.0) {
                // Textured faces should only receive lighting, not block-color tinting.
                color = vec3(brightness);
            } else {
                color = instance_color * brightness;
            }
            
            // Simplified fog calculation
            float distance = length(world_pos - camera_pos);
            fog_factor = clamp(1.0 - (distance - 40.0) / 60.0, 0.0, 1.0);
            texture_layer = instance_texture_layer;
        }
        '''
        
        fragment_shader = '''
        #version 330 core
        
        in vec3 color;
        in float fog_factor;
        in vec2 uv;
        flat in float texture_layer;
        
        uniform vec3 fog_color;
        uniform sampler2DArray block_textures;
        
        out vec4 fragColor;
        
        void main() {
            vec4 sampled = vec4(1.0, 1.0, 1.0, 1.0);
            if (texture_layer >= 0.0) {
                sampled = texture(block_textures, vec3(uv, texture_layer));
                if (sampled.a < 0.1) {
                    discard;
                }
            }

            vec3 lit_color = color * sampled.rgb;
            vec3 final_color = mix(fog_color, lit_color, fog_factor);
            fragColor = vec4(final_color, sampled.a);
        }
        '''
        
        self.block_shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        self._load_block_textures()
        
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

        character_vertex_shader = '''
        #version 330 core

        layout(location = 0) in vec3 in_position;
        layout(location = 1) in vec3 in_normal;
        layout(location = 2) in vec2 in_uv;

        uniform mat4 projection_matrix;
        uniform mat4 view_matrix;
        uniform mat4 model_matrix;
        uniform vec3 light_dir;
        uniform vec3 camera_pos;

        out vec2 uv;
        out float lighting;
        out float fog_factor;

        void main() {
            vec4 world_pos4 = model_matrix * vec4(in_position, 1.0);
            vec3 world_pos = world_pos4.xyz;
            mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
            vec3 world_normal = normalize(normal_matrix * in_normal);

            gl_Position = projection_matrix * view_matrix * world_pos4;
            uv = in_uv;

            lighting = max(0.6, abs(dot(world_normal, normalize(-light_dir))));
            float distance = length(world_pos - camera_pos);
            fog_factor = clamp(1.0 - (distance - 40.0) / 60.0, 0.0, 1.0);
        }
        '''

        character_fragment_shader = '''
        #version 330 core

        in vec2 uv;
        in float lighting;
        in float fog_factor;

        uniform sampler2D part_texture;
        uniform vec3 fog_color;

        out vec4 fragColor;

        void main() {
            vec4 sampled = texture(part_texture, uv);
            if (sampled.a < 0.1) {
                discard;
            }

            vec3 lit_color = sampled.rgb * lighting;
            vec3 final_color = mix(fog_color, lit_color, fog_factor);
            fragColor = vec4(final_color, sampled.a);
        }
        '''

        self.character_shader = self.ctx.program(
            vertex_shader=character_vertex_shader,
            fragment_shader=character_fragment_shader
        )
        
        print("✅ Shaders created successfully")
        self._load_character_textures()
    
    def _create_geometry_buffers(self):
        """Create optimized geometry buffers for instanced rendering"""
        # Unit quad centered at origin; orientation is determined in vertex shader by face id
        quad_vertices = np.array([
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ], dtype=np.float32)

        self.face_vbo = self.ctx.buffer(quad_vertices.tobytes())

        # Reusable dynamic instance buffer: [pos.xyz, color.rgb, face_id, texture_layer]
        self.instance_stride_bytes = 8 * 4
        self.instance_buffer = self.ctx.buffer(
            reserve=self.max_instances * self.instance_stride_bytes,
            dynamic=True,
        )

        self.block_vao = self.ctx.vertex_array(
            self.block_shader,
            [
                (self.face_vbo, '2f', 'quad_pos'),
                (self.instance_buffer, '3f 3f 1f 1f/i', 'instance_pos', 'instance_color', 'instance_face', 'instance_texture_layer'),
            ],
        )
        
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
        self.crosshair_vao = self.ctx.vertex_array(
            self.ui_shader,
            [(self.crosshair_vbo, '2f 3f', 'position', 'color')]
        )
        self._last_crosshair_center: Optional[Tuple[int, int]] = None

        character_vertices = self._create_character_cube_geometry()
        self.character_vbo = self.ctx.buffer(character_vertices.tobytes())
        self.character_vao = self.ctx.vertex_array(
            self.character_shader,
            [(self.character_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')]
        )
        
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
        
        # Frustum planes for culling
        self.frustum_planes = np.zeros((6, 4), dtype=np.float32)
        
        # Performance flags
        self.use_moderngl = True  # Always true now
        self.use_numpy_optimization = True  # Always available with NumPy
        
        print(f"🚀 ModernGL GPU Renderer initialized - Screen: {self.screen_width}x{self.screen_height}")
    
    def _init_moderngl_context(self):
        """Initialize ModernGL context and pygame window"""
        reused_surface = False
        if self.existing_screen is not None:
            try:
                has_opengl = bool(self.existing_screen.get_flags() & pygame.OPENGL)
                same_size = self.existing_screen.get_size() == (self.screen_width, self.screen_height)
                if has_opengl and same_size:
                    self.screen = self.existing_screen
                    pygame.display.set_caption("Pycraft - ModernGL GPU Renderer")
                    reused_surface = True
            except pygame.error:
                reused_surface = False

        if not reused_surface:
            # Create OpenGL-enabled window for ModernGL with proper depth buffer
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            # Request depth buffer
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)
            pygame.display.set_caption("Pycraft - ModernGL GPU Renderer")
        
        # Create ModernGL context
        self.ctx = mgl.create_context()
        
        # Enable depth testing. Keep culling disabled because crossed vegetation
        # planes are intentionally single-quad billboards and must be visible
        # from both sides.
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)
        
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
    def render_world(self, world: World, camera: Camera, performance_mode=True, player_state: Optional[Dict[str, Any]] = None):
        """High-performance world rendering with ModernGL"""
        start_time = time.time()
        
        # Store world reference for face culling
        self._current_world = world
        
        # Clear screen with sky blue color and depth buffer
        self.ctx.clear(0.529, 0.808, 0.922, 1.0, 1.0)  # Clear color and depth in one call
        
        # Ensure proper GL state
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)
        
        # Set viewport to match screen size
        self.ctx.viewport = (0, 0, self.screen_width, self.screen_height)
        
        # Configure depth test (ModernGL uses string values)
        self.ctx.depth_func = '<'  # Less than comparison
        
        # Use user-selected render distance directly so pause-menu slider maps 1:1.
        render_distance = self.render_distance_chunks

        # Scale face budget with render distance area so larger distances can
        # actually display farther chunks instead of being clipped by a fixed cap.
        if performance_mode:
            base_distance = max(1, PERFORMANCE_RENDER_DISTANCE)
            base_budget = MAX_BLOCKS
        else:
            base_distance = max(1, RENDER_DISTANCE)
            base_budget = MAX_BLOCKS * 2

        area_scale = (render_distance / float(base_distance)) ** 2
        max_blocks = int(base_budget * area_scale)
        max_blocks = max(base_budget, min(max_blocks, self.max_instances))
        
        # Get visible chunks using optimized culling
        visible_chunks = self._get_optimized_visible_chunks(world, camera, render_distance)
        # Batch process all blocks using NumPy
        block_data = self._prepare_block_data(world, visible_chunks, camera, max_blocks)
        if len(block_data['positions']) > 0:
            self._render_blocks_moderngl(block_data, camera)

        self._render_player_model(player_state, camera)

        # Render UI elements
        self._render_ui_moderngl(world, camera)
        # Update performance stats
        render_time = (time.time() - start_time) * 1000
        self.last_stats['render_time_ms'] = render_time
        self.last_stats['frames_rendered'] += 1

    def set_render_distance(self, render_distance: int) -> None:
        """Set active render distance in chunks for world chunk visibility culling."""
        clamped_distance = max(MIN_RENDER_DISTANCE, min(MAX_RENDER_DISTANCE, int(render_distance)))
        if clamped_distance == self.render_distance_chunks:
            return
        self.render_distance_chunks = clamped_distance
        self._cached_render_distance = -1
    
    def _render_blocks_moderngl(self, block_data: Dict, camera: Camera):
        """Render blocks using ModernGL instanced rendering"""
        face_count = len(block_data['positions'])
        if face_count == 0:
            return
        
        # Prepare matrices
        view_matrix = self._create_view_matrix(camera)
        
        # Set uniforms (ModernGL automatically binds the program when setting uniforms)
        self.block_shader['projection_matrix'].write(self.projection_matrix.T.astype(np.float32).tobytes())
        self.block_shader['view_matrix'].write(view_matrix.T.astype(np.float32).tobytes())
        self.block_shader['light_dir'].write(np.array([0.2, -1.0, 0.3], dtype=np.float32).tobytes())
        self.block_shader['camera_pos'].write(np.array(camera.position, dtype=np.float32).tobytes())
        self.block_shader['fog_color'].write(np.array([0.529, 0.808, 0.922], dtype=np.float32).tobytes())
        self.block_shader['grass_width'].value = TALL_GRASS_WIDTH
        self.block_texture_array.use(location=0)
        self.block_shader['block_textures'].value = 0
        
        # Update reusable instance buffer: [pos.xyz, color.rgb, face_id, texture_layer]
        instance_data = np.column_stack([
            block_data['positions'],
            block_data['colors'],
            block_data['face_ids'].astype(np.float32, copy=False).reshape(-1, 1),
            block_data['texture_layers'].astype(np.float32, copy=False).reshape(-1, 1),
        ]).astype(np.float32, copy=False)

        instance_count = min(face_count, self.max_instances)
        if instance_count <= 0:
            return

        self.instance_buffer.write(instance_data[:instance_count].tobytes(), offset=0)

        # Render one quad per visible face
        self.block_vao.render(mode=mgl.TRIANGLES, vertices=6, instances=instance_count)

        self.last_stats['faces'] = instance_count
        self.last_stats['blocks'] = block_data.get('total_blocks', 0)
    
    def _prepare_block_data(self, world: World, chunks: List[Chunk], camera: Camera, max_blocks: int) -> Dict:
        """Ultra-optimized batch processing for 10k+ blocks using NumPy arrays"""
        if not chunks:
            return {
                'positions': np.empty((0, 3), dtype=np.float32),
                'colors': np.empty((0, 3), dtype=np.float32),
                'face_ids': np.empty(0, dtype=np.uint8),
                'texture_layers': np.empty(0, dtype=np.int8),
                'total_blocks': 0,
            }
        
        # Pre-allocate lists for concatenation
        all_positions = []
        all_colors = []
        all_face_ids = []
        all_texture_layers = []
        total_blocks = 0
        
        # Batch process chunks with minimal object creation
        for chunk in chunks:
            if not chunk.blocks:
                continue
            
            # Get optimized visible faces data
            visible_data = chunk.get_visible_faces(world)
            
            if len(visible_data['positions']) > 0:
                all_positions.append(visible_data['positions'])
                all_colors.append(visible_data['colors'])
                all_face_ids.append(visible_data['face_ids'])
                all_texture_layers.append(visible_data['texture_layers'])
            
            total_blocks += len(chunk.blocks)

        # Efficiently concatenate all arrays
        if all_positions:
            final_positions = np.concatenate(all_positions, axis=0)[:max_blocks]
            final_colors = np.concatenate(all_colors, axis=0)[:max_blocks]
            final_face_ids = np.concatenate(all_face_ids, axis=0)[:max_blocks]
            final_texture_layers = np.concatenate(all_texture_layers, axis=0)[:max_blocks]
        else:
            final_positions = np.empty((0, 3), dtype=np.float32)
            final_colors = np.empty((0, 3), dtype=np.float32)
            final_face_ids = np.empty(0, dtype=np.uint8)
            final_texture_layers = np.empty(0, dtype=np.int8)

        # Update stats
        visible_face_count = len(final_positions)
        self.last_stats['blocks'] = total_blocks
        self.last_stats['culled_blocks'] = max(0, total_blocks - min(total_blocks, visible_face_count))

        return {
            'positions': final_positions,
            'colors': final_colors,
            'face_ids': final_face_ids,
            'texture_layers': final_texture_layers,
            'total_blocks': total_blocks,
        }

    def _load_block_textures(self):
        """Load block textures into one texture array used by instanced face rendering."""
        texture_dir = Path(__file__).resolve().parent.parent / 'assets' / 'textures' / 'blocks'
        texture_paths = [
            texture_dir / 'tall_grass_1.png',
            texture_dir / 'tall_grass_2.png',
            texture_dir / 'tall_grass_3.png',
            texture_dir / 'grass_side.png',
            texture_dir / 'grass.png',
            texture_dir / 'dirt.png',
            texture_dir / 'stone.png',
            texture_dir / 'wood.png',
            texture_dir / 'leaf.png',
        ]

        layers = []
        for path in texture_paths:
            if path.exists():
                image = pygame.image.load(path.as_posix()).convert_alpha()
            else:
                # Fallback keeps renderer robust even when texture files are missing.
                image = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
                image.fill((255, 255, 255, 255))

            image = pygame.transform.flip(image, False, True)
            if image.get_size() != (16, 16):
                image = pygame.transform.smoothscale(image, (16, 16))
            layers.append(pygame.image.tostring(image, 'RGBA'))

        texture_array_data = b''.join(layers)
        layer_count = len(layers)
        self.block_texture_array = self.ctx.texture_array((16, 16, layer_count), 4, texture_array_data)
        self.block_texture_array.filter = (mgl.NEAREST, mgl.NEAREST)
        self.block_texture_array.repeat_x = False
        self.block_texture_array.repeat_y = False

    def _load_character_textures(self):
        """Load the per-body-part textures used by the player model renderer."""
        texture_dir = Path(__file__).resolve().parent.parent / 'assets' / 'textures' / 'characters' / 'main_character_parts'
        texture_names = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

        for name in texture_names:
            path = texture_dir / f'{name}.png'
            if path.exists():
                image = pygame.image.load(path.as_posix()).convert_alpha()
            else:
                image = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
                image.fill((255, 255, 255, 255))

            image = pygame.transform.flip(image, False, True)
            data = pygame.image.tostring(image, 'RGBA')
            texture = self.ctx.texture(image.get_size(), 4, data)
            texture.filter = (mgl.NEAREST, mgl.NEAREST)
            texture.repeat_x = False
            texture.repeat_y = False
            self.character_textures[name] = texture

    def _create_character_cube_geometry(self) -> np.ndarray:
        """Create a unit cube mesh with position/normal/UV for textured body parts."""
        # Each vertex: position.xyz, normal.xyz, uv.xy
        vertices = np.array([
            # Front (+Z)
            -0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 0.0,
             0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 1.0,

            # Back (-Z)
             0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  1.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 0.0, -1.0,  1.0, 1.0,
             0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 0.0, -1.0,  1.0, 1.0,
             0.5,  0.5, -0.5,  0.0, 0.0, -1.0,  0.0, 1.0,

            # Left (-X)
            -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 0.0,
            -0.5, -0.5,  0.5, -1.0, 0.0, 0.0,  1.0, 0.0,
            -0.5,  0.5,  0.5, -1.0, 0.0, 0.0,  1.0, 1.0,
            -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 0.0,
            -0.5,  0.5,  0.5, -1.0, 0.0, 0.0,  1.0, 1.0,
            -0.5,  0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 1.0,

            # Right (+X)
             0.5, -0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 1.0,
             0.5, -0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 1.0,

            # Top (+Y)
            -0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  0.0, 0.0,
             0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 1.0,
            -0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  0.0, 1.0,

            # Bottom (-Y)
            -0.5, -0.5, -0.5,  0.0, -1.0, 0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0, -1.0, 0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0, 0.0,  1.0, 1.0,
            -0.5, -0.5, -0.5,  0.0, -1.0, 0.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0, 0.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, -1.0, 0.0,  0.0, 1.0,
        ], dtype=np.float32)
        return vertices

    def _translation_matrix(self, tx: float, ty: float, tz: float) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = tx
        matrix[1, 3] = ty
        matrix[2, 3] = tz
        return matrix

    def _scale_matrix(self, sx: float, sy: float, sz: float) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        matrix[2, 2] = sz
        return matrix

    def _rotation_y_matrix(self, yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.array([
            [ c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _rotation_x_matrix(self, angle: float) -> np.ndarray:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _render_player_model(self, player_state: Optional[Dict[str, Any]], camera: Camera) -> None:
        """Render the main character model when third-person mode is active."""
        if not player_state or not player_state.get('visible', False):
            return

        if not self.character_textures:
            return

        pos = player_state.get('position')
        if pos is None or len(pos) != 3:
            return

        base_x = float(pos[0])
        base_y = float(pos[1]) - 1.6
        base_z = float(pos[2])
        yaw = float(player_state.get('yaw', 0.0))
        walking = bool(player_state.get('walking', False))
        move_factor = float(player_state.get('move_factor', 0.0))

        swing = 0.0
        if walking and move_factor > 0.01:
            swing = np.sin(time.time() * 8.0) * (0.7 * move_factor)

        # Swap arm textures only when the camera is behind the character so
        # front view keeps left/right correct while back view appears mirrored.
        facing = np.array([np.sin(yaw), 0.0, np.cos(yaw)], dtype=np.float32)
        character_center = np.array([base_x, base_y + 1.0, base_z], dtype=np.float32)
        to_camera = np.array(camera.position, dtype=np.float32) - character_center
        to_camera[1] = 0.0
        is_back_view = bool(np.dot(to_camera, facing) < 0.0)

        view_matrix = self._create_view_matrix(camera)
        self.character_shader['projection_matrix'].write(self.projection_matrix.T.astype(np.float32).tobytes())
        self.character_shader['view_matrix'].write(view_matrix.T.astype(np.float32).tobytes())
        self.character_shader['light_dir'].write(np.array([0.2, -1.0, 0.3], dtype=np.float32).tobytes())
        self.character_shader['camera_pos'].write(np.array(camera.position, dtype=np.float32).tobytes())
        self.character_shader['fog_color'].write(np.array([0.529, 0.808, 0.922], dtype=np.float32).tobytes())

        base_matrix = self._translation_matrix(base_x, base_y, base_z) @ self._rotation_y_matrix(yaw)

        for part_name, (center, size) in self._character_part_specs.items():
            texture_part_name = part_name
            if is_back_view and part_name == 'left_arm':
                texture_part_name = 'right_arm'
            elif is_back_view and part_name == 'right_arm':
                texture_part_name = 'left_arm'

            texture = self.character_textures.get(texture_part_name)
            if texture is None:
                continue

            if part_name == 'left_arm':
                pivot_y = size[1] * 0.5
                part_matrix = (
                    base_matrix
                    @ self._translation_matrix(center[0], center[1] + pivot_y, center[2])
                    @ self._rotation_x_matrix(-swing)
                    @ self._translation_matrix(0.0, -pivot_y, 0.0)
                    @ self._scale_matrix(size[0], size[1], size[2])
                )
            elif part_name == 'right_arm':
                pivot_y = size[1] * 0.5
                part_matrix = (
                    base_matrix
                    @ self._translation_matrix(center[0], center[1] + pivot_y, center[2])
                    @ self._rotation_x_matrix(swing)
                    @ self._translation_matrix(0.0, -pivot_y, 0.0)
                    @ self._scale_matrix(size[0], size[1], size[2])
                )
            elif part_name == 'left_leg':
                pivot_y = size[1] * 0.5
                part_matrix = (
                    base_matrix
                    @ self._translation_matrix(center[0], center[1] + pivot_y, center[2])
                    @ self._rotation_x_matrix(swing)
                    @ self._translation_matrix(0.0, -pivot_y, 0.0)
                    @ self._scale_matrix(size[0], size[1], size[2])
                )
            elif part_name == 'right_leg':
                pivot_y = size[1] * 0.5
                part_matrix = (
                    base_matrix
                    @ self._translation_matrix(center[0], center[1] + pivot_y, center[2])
                    @ self._rotation_x_matrix(-swing)
                    @ self._translation_matrix(0.0, -pivot_y, 0.0)
                    @ self._scale_matrix(size[0], size[1], size[2])
                )
            else:
                part_matrix = base_matrix @ self._translation_matrix(center[0], center[1], center[2]) @ self._scale_matrix(size[0], size[1], size[2])
            self.character_shader['model_matrix'].write(part_matrix.T.astype(np.float32).tobytes())
            texture.use(location=0)
            self.character_shader['part_texture'].value = 0
            self.character_vao.render(mode=mgl.TRIANGLES)

    def _get_optimized_visible_chunks(self, world: World, camera: Camera, render_distance: int) -> List[Chunk]:
        # Recompute only when crossing chunk boundaries or world chunk count changes
        center_chunk = world.get_chunk_coords(int(camera.position[0]), int(camera.position[2]))
        world_chunk_count = len(world.chunks)
        cache_valid = (
            self._cached_chunk_center == center_chunk
            and self._cached_world_chunk_count == world_chunk_count
            and self._cached_render_distance == render_distance
        )
        if cache_valid:
            return self._cached_visible_chunks

        visible_chunks = world.get_visible_chunks(
            int(camera.position[0]),
            int(camera.position[2]),
            render_distance=render_distance,
            to_create=False,
        )
        self._cached_visible_chunks = visible_chunks
        self._cached_chunk_center = center_chunk
        self._cached_world_chunk_count = world_chunk_count
        self._cached_render_distance = render_distance
        return visible_chunks

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
        
        # Update geometry only when center changes (e.g. window resize)
        cx, cy = self.screen_width // 2, self.screen_height // 2
        center = (cx, cy)
        if self._last_crosshair_center != center:
            size = 10
            crosshair_vertices = np.array([
                cx - size, cy, 1.0, 1.0, 1.0,
                cx + size, cy, 1.0, 1.0, 1.0,
                cx, cy - size, 1.0, 1.0, 1.0,
                cx, cy + size, 1.0, 1.0, 1.0,
            ], dtype=np.float32)
            self.crosshair_vbo.write(crosshair_vertices.tobytes())
            self._last_crosshair_center = center

        # Render as lines with reusable VAO
        self.crosshair_vao.render(mode=mgl.LINES)
        
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

            cache_key = (text, font_size, (int(color[0]), int(color[1]), int(color[2])))
            cache_entry = self._text_texture_cache.get(cache_key)
            if cache_entry is not None:
                texture, w, h, _ = cache_entry
                self._text_texture_cache[cache_key] = (texture, w, h, time.time())
            else:
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
                self._text_texture_cache[cache_key] = (texture, w, h, time.time())
            
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
            
            # Cleanup transient geometry buffers
            vao.release()
            vbo.release()
            ibo.release()
            
        except Exception as e:
            print(f"Text rendering error: {e}")
            pass

    def _prune_text_texture_cache(self):
        """Release stale text textures to keep GPU memory bounded."""
        now = time.time()
        stale_keys = [
            key
            for key, (_, _, _, last_used) in self._text_texture_cache.items()
            if now - last_used > self._text_cache_ttl_seconds
        ]
        for key in stale_keys:
            texture, _, _, _ = self._text_texture_cache.pop(key)
            texture.release()

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

            now = time.time()
            if now - self._last_debug_text_update >= self._debug_text_update_interval:
                fps_text = self._format_debug_line('fps', data.get('fps', 0), 'fps')
                pos = data.get('position', (0, 0, 0))
                # Round to reduce noisy updates and texture churn.
                pos_text = f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
                chunk_text = self._format_debug_line('chunk', data.get('chunk', (0,0)), 'chunk')
                chunks_text = self._format_debug_line('chunks_loaded', data.get('chunks_loaded', 0), 'chunks_loaded')
                block_text = self._format_debug_line('selected_block', data.get('selected_block', ''), 'selected_block')
                perf_text = self._format_debug_line('performance', data.get('performance_mode'), 'performance')
                stats_text = self._format_debug_line('blocks_faces', self.last_stats, 'blocks_faces')

                self._debug_text_lines = [
                    (fps_text, (255, 255, 0)),
                    (pos_text, (255, 255, 255)),
                    (chunk_text, (255, 255, 255)),
                    (chunks_text, (255, 255, 255)),
                    (block_text, (255, 255, 255)),
                    (perf_text, (255, 255, 255)),
                    (stats_text, (0, 255, 255)),
                ]
                self._last_debug_text_update = now

            if now - self._last_text_cache_prune >= 1.0:
                self._prune_text_texture_cache()
                self._last_text_cache_prune = now
            
            # Create text shader if not exists
            if not hasattr(self, 'text_shader'):
                self._create_text_shader()
            
            # Render debug information on screen
            y_offset = 10
            line_height = 25

            for line_text, line_color in self._debug_text_lines:
                self._render_text_texture(line_text, 10, y_offset, font_size=20, color=line_color)
                y_offset += line_height
            
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
