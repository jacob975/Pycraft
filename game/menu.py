"""
ModernGL-based menu system for Pycraft.
Provides high-performance GPU-accelerated menu rendering with OpenGL shaders.
"""

from __future__ import annotations
import pygame
import sys
import time
from datetime import datetime
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from config import *

# ModernGL is required for this menu system
try:
    import moderngl as mgl
    MODERNGL_AVAILABLE = True
except ImportError:
    raise ImportError("ModernGL is required for GPU menu rendering. Install with: pip install moderngl")

from .font_manager import get_font_manager
from .saves import SaveMetadata, list_saves

@dataclass
class ButtonState:
    """Button state data"""
    x: int
    y: int
    width: int
    height: int
    text: str
    font_size: int
    color: Tuple[int, int, int]
    bg_color: Tuple[int, int, int]
    hover_color: Tuple[int, int, int]
    is_hovered: bool = False
    is_pressed: bool = False

class ModernGLButton:
    """High-performance button using ModernGL for rendering"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255),
                 bg_color: Tuple[int, int, int] = (50, 50, 50),
                 hover_color: Tuple[int, int, int] = (80, 80, 80)):
        self.state = ButtonState(x, y, width, height, text, font_size, color, bg_color, hover_color)
        self.rect = pygame.Rect(x, y, width, height)
        
    def handle_event(self, event) -> bool:
        """Handle mouse events and return True if clicked"""
        if event.type == pygame.MOUSEMOTION:
            self.state.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.state.is_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.state.is_pressed and self.rect.collidepoint(event.pos):
                self.state.is_pressed = False
                return True
            self.state.is_pressed = False
        return False

class ModernGLMenu:
    """High-performance ModernGL-based main menu interface"""
    
    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT, screen: pygame.Surface = None):
        self.width = width
        self.height = height
        self.running = True
        self.selected_option = None
        self._text_texture_cache: Dict[Tuple[str, int, Tuple[int, int, int], bool], Tuple[mgl.Texture, int, int]] = {}
        
        # Initialize ModernGL context
        self._init_moderngl_context(screen)
        self._create_shaders()
        self._create_geometry_buffers()
        
        # Background colors
        self.bg_color = (30/255, 30/255, 50/255)  # Normalized for OpenGL
        
        # Animation state
        self.title_scale = 1.0
        self.title_time = 0.0
        
        # Create buttons with ModernGL
        self._create_buttons()
        
        self.clock = pygame.time.Clock()
        
        print("🚀 ModernGL Menu System initialized")
    
    def _init_moderngl_context(self, existing_screen: pygame.Surface = None):
        """Initialize ModernGL context and pygame OpenGL window"""

        def create_new_window():
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

            try:
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
            except pygame.error:
                print("⚠️ MSAA not supported, continuing without anti-aliasing")

            flags = pygame.OPENGL | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            pygame.display.set_caption("Pycraft - ModernGL Menu")
            pygame.display.gl_set_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT, 1)

        reused_context = False
        if existing_screen is not None:
            try:
                has_opengl = bool(existing_screen.get_flags() & pygame.OPENGL)
                same_size = existing_screen.get_size() == (self.width, self.height)
                if has_opengl and same_size:
                    self.screen = existing_screen
                    reused_context = True
                    pygame.display.set_caption("Pycraft - ModernGL Menu")
            except pygame.error:
                reused_context = False

        if not reused_context:
            create_new_window()

        try:
            self.ctx = mgl.create_context()
            print("✅ ModernGL context created successfully")
        except Exception as e:
            if reused_context:
                print("⚠️ Failed to attach to existing OpenGL context, recreating window...")
                create_new_window()
                try:
                    self.ctx = mgl.create_context()
                    print("✅ ModernGL context created successfully")
                except Exception as inner_error:
                    raise RuntimeError(f"Failed to create ModernGL context: {inner_error}") from inner_error
            else:
                raise RuntimeError(f"Failed to create ModernGL context: {e}") from e

        # Enable features for UI rendering
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        # Ensure depth testing is disabled so 2D UI elements render in draw order
        # (the pause menu shares the GPU context with the 3D renderer, which leaves depth testing on)
        self.ctx.disable(mgl.DEPTH_TEST)
        # Ensure face culling is disabled so UI quads are not discarded when reusing 3D renderer state.
        self.ctx.disable(mgl.CULL_FACE)

        print("✅ ModernGL context initialized for menu")
    
    def _create_shaders(self):
        """Create shader programs for menu rendering"""
        
        # Shader for solid color rectangles (buttons, backgrounds)
        rect_vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 color;
        
        uniform mat4 ortho_matrix;
        uniform vec2 offset;
        uniform vec2 scale;
        
        out vec3 vertex_color;
        
        void main() {
            vec2 scaled_pos = position * scale + offset;
            gl_Position = ortho_matrix * vec4(scaled_pos, 0.0, 1.0);
            vertex_color = color;
        }
        '''
        
        rect_fragment_shader = '''
        #version 330 core
        
        in vec3 vertex_color;
        out vec4 fragColor;
        
        uniform float alpha;
        
        void main() {
            fragColor = vec4(vertex_color, alpha);
        }
        '''
        
        self.rect_shader = self.ctx.program(
            vertex_shader=rect_vertex_shader,
            fragment_shader=rect_fragment_shader
        )
        
        # Shader for textured rendering (text, images)
        texture_vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        
        uniform mat4 ortho_matrix;
        uniform vec2 offset;
        uniform vec2 scale;
        
        out vec2 uv;
        
        void main() {
            vec2 scaled_pos = position * scale + offset;
            gl_Position = ortho_matrix * vec4(scaled_pos, 0.0, 1.0);
            uv = texcoord;
        }
        '''
        
        texture_fragment_shader = '''
        #version 330 core
        
        in vec2 uv;
        uniform sampler2D texture_sampler;
        uniform vec3 text_color;
        uniform float alpha;
        
        out vec4 fragColor;
        
        void main() {
            vec4 sampled = texture(texture_sampler, uv);
            fragColor = vec4(text_color * sampled.rgb, sampled.a * alpha);
        }
        '''
        
        self.texture_shader = self.ctx.program(
            vertex_shader=texture_vertex_shader,
            fragment_shader=texture_fragment_shader
        )
        
        # Background gradient shader
        gradient_vertex_shader = '''
        #version 330 core
        
        layout(location = 0) in vec2 position;
        
        uniform mat4 ortho_matrix;
        
        out vec2 screen_pos;
        
        void main() {
            gl_Position = ortho_matrix * vec4(position, 0.0, 1.0);
            screen_pos = position;
        }
        '''
        
        gradient_fragment_shader = '''
        #version 330 core
        
        in vec2 screen_pos;
        uniform vec2 screen_size;
        uniform float time;
        
        out vec4 fragColor;
        
        void main() {
            vec2 uv = screen_pos / screen_size;
            
            // Create animated background pattern
            float pattern1 = sin(uv.x * 20.0 + time * 0.5) * 0.05;
            float pattern2 = cos(uv.y * 15.0 + time * 0.3) * 0.05;
            
            vec3 base_color = vec3(0.12, 0.12, 0.2);  // Dark blue-gray
            vec3 pattern_color = base_color + vec3(pattern1 + pattern2);
            
            // Add subtle vignette effect
            float vignette = 1.0 - smoothstep(0.3, 0.8, length(uv - 0.5));
            pattern_color *= vignette;
            
            fragColor = vec4(pattern_color, 1.0);
        }
        '''
        
        self.gradient_shader = self.ctx.program(
            vertex_shader=gradient_vertex_shader,
            fragment_shader=gradient_fragment_shader
        )
        
        print("✅ Menu shaders created")
    
    def _create_geometry_buffers(self):
        """Create geometry buffers for menu elements"""
        
        # Create orthographic projection matrix for UI
        self.ortho_matrix = np.array([
            [2.0 / self.width, 0, 0, -1],
            [0, -2.0 / self.height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Create unit quad for buttons and backgrounds
        quad_vertices = np.array([
            # Position, Color (will be overridden by uniforms)
            0.0, 0.0, 1.0, 1.0, 1.0,  # Bottom-left
            1.0, 0.0, 1.0, 1.0, 1.0,  # Bottom-right
            1.0, 1.0, 1.0, 1.0, 1.0,  # Top-right
            0.0, 1.0, 1.0, 1.0, 1.0,  # Top-left
        ], dtype=np.float32)
        
        quad_indices = np.array([
            0, 1, 2,  # First triangle
            0, 2, 3   # Second triangle
        ], dtype=np.uint32)
        
        # Texture coordinates for text rendering
        tex_quad_vertices = np.array([
            # Position, TexCoord
            0.0, 0.0, 0.0, 1.0,  # Bottom-left
            1.0, 0.0, 1.0, 1.0,  # Bottom-right
            1.0, 1.0, 1.0, 0.0,  # Top-right
            0.0, 1.0, 0.0, 0.0,  # Top-left
        ], dtype=np.float32)
        
        # Create buffers
        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        self.quad_ibo = self.ctx.buffer(quad_indices.tobytes())
        self.tex_quad_vbo = self.ctx.buffer(tex_quad_vertices.tobytes())

        # Reusable VAO for textured quad rendering.
        self.texture_vao = self.ctx.vertex_array(
            self.texture_shader,
            [(self.tex_quad_vbo, '2f 2f', 'position', 'texcoord')],
            self.quad_ibo
        )

        # Reusable dynamic VBO/VAO for colored quads (buttons, borders, rects).
        self.dynamic_rect_vbo = self.ctx.buffer(reserve=4 * 5 * 4, dynamic=True)
        self.dynamic_rect_vao = self.ctx.vertex_array(
            self.rect_shader,
            [(self.dynamic_rect_vbo, '2f 3f', 'position', 'color')],
            self.quad_ibo
        )
        
        # Fullscreen quad for background
        fullscreen_vertices = np.array([
            0.0, 0.0,
            self.width, 0.0,
            self.width, self.height,
            0.0, self.height
        ], dtype=np.float32)
        
        self.fullscreen_vbo = self.ctx.buffer(fullscreen_vertices.tobytes())

        self.gradient_vao = self.ctx.vertex_array(
            self.gradient_shader,
            [(self.fullscreen_vbo, '2f', 'position')],
            self.quad_ibo
        )
        
        print("✅ Menu geometry buffers created")
    
    def _create_buttons(self):
        """Create button objects"""
        button_width = 300
        button_height = 50
        button_spacing = 20
        start_y = self.height // 2 - 30
        
        self.buttons = {
            'new_world': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y,
                button_width,
                button_height,
                "Start New World",
                font_size=28,
                bg_color=(40, 120, 40),
                hover_color=(60, 140, 60)
            ),
            'load_world': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y + button_height + button_spacing,
                button_width,
                button_height,
                "Load World",
                font_size=28,
                bg_color=(40, 80, 120),
                hover_color=(60, 100, 140)
            ),
            'exit': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y + 2 * (button_height + button_spacing),
                button_width,
                button_height,
                "Exit Game",
                font_size=28,
                bg_color=(120, 40, 40),
                hover_color=(140, 60, 60)
            )
        }
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.selected_option = 'exit'
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.selected_option = 'exit'
                    self.running = False
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    self.selected_option = 'new_world'
                    self.running = False
            
            # Handle button events
            for button_name, button in self.buttons.items():
                if button.handle_event(event):
                    self.selected_option = button_name
                    self.running = False
    
    def update(self, dt: float):
        """Update menu state"""
        self.title_time += dt
        self.title_scale = 1.0 + 0.05 * abs(np.sin(self.title_time * 2.0))
    
    def render(self):
        """Render the menu using ModernGL"""
        # Re-assert UI-safe GL state each frame in case previous renderer changed it.
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        # Clear screen
        self.ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        
        # Set viewport
        self.ctx.viewport = (0, 0, self.width, self.height)

        # Render background
        self._render_background()
        
        # Render title
        self._render_title()
        
        # Render subtitle
        self._render_subtitle()
        
        # Render buttons
        self._render_buttons()
        
        # Render UI text
        self._render_ui_info()
        
        # Swap buffers
        pygame.display.flip()
    
    def _render_background(self):
        """Render animated background pattern"""
        self.gradient_shader['ortho_matrix'].write(self.ortho_matrix.T.tobytes())
        self.gradient_shader['screen_size'].write(np.array([self.width, self.height], dtype=np.float32).tobytes())
        self.gradient_shader['time'].write(np.array([self.title_time], dtype=np.float32).tobytes())

        self.gradient_vao.render()
    
    def _render_buttons(self):
        """Render all buttons using ModernGL"""
        for button_name, button in self.buttons.items():
            self._render_button(button)
            self._render_button_text(button)
    
    def _render_button(self, button: ModernGLButton):
        """Render a single button background"""
        # Choose color based on state
        if button.state.is_pressed:
            color = tuple(max(0, c - 20) for c in button.state.hover_color)
        elif button.state.is_hovered:
            color = button.state.hover_color
        else:
            color = button.state.bg_color
        
        # Normalize color to 0-1 range
        norm_color = np.array([c / 255.0 for c in color], dtype=np.float32)
        
        # Set shader uniforms
        self.rect_shader['ortho_matrix'].write(self.ortho_matrix.T.tobytes())
        self.rect_shader['offset'].write(np.array([button.state.x, button.state.y], dtype=np.float32).tobytes())
        self.rect_shader['scale'].write(np.array([button.state.width, button.state.height], dtype=np.float32).tobytes())
        self.rect_shader['alpha'].write(np.array([1.0], dtype=np.float32).tobytes())
        
        button_vertices = np.array([
            # Position, Color
            0.0, 0.0, norm_color[0], norm_color[1], norm_color[2],
            1.0, 0.0, norm_color[0], norm_color[1], norm_color[2],
            1.0, 1.0, norm_color[0], norm_color[1], norm_color[2],
            0.0, 1.0, norm_color[0], norm_color[1], norm_color[2],
        ], dtype=np.float32)

        self.dynamic_rect_vbo.write(button_vertices.tobytes())
        self.dynamic_rect_vao.render()
        
        # Draw button border
        self._render_button_border(button)
    
    def _render_button_border(self, button: ModernGLButton):
        """Render button border"""
        border_color = np.array([0.4, 0.4, 0.4], dtype=np.float32)  # Gray border
        border_width = 2
        
        # Top border
        self._render_rect(
            button.state.x, button.state.y + button.state.height - border_width,
            button.state.width, border_width, border_color
        )
        
        # Bottom border
        self._render_rect(
            button.state.x, button.state.y,
            button.state.width, border_width, border_color
        )
        
        # Left border
        self._render_rect(
            button.state.x, button.state.y,
            border_width, button.state.height, border_color
        )
        
        # Right border
        self._render_rect(
            button.state.x + button.state.width - border_width, button.state.y,
            border_width, button.state.height, border_color
        )
    
    def _render_rect(self, x: int, y: int, width: int, height: int, color: np.ndarray):
        """Render a solid color rectangle"""
        self.rect_shader['ortho_matrix'].write(self.ortho_matrix.T.tobytes())
        self.rect_shader['offset'].write(np.array([x, y], dtype=np.float32).tobytes())
        self.rect_shader['scale'].write(np.array([width, height], dtype=np.float32).tobytes())
        self.rect_shader['alpha'].write(np.array([1.0], dtype=np.float32).tobytes())
        
        rect_vertices = np.array([
            0.0, 0.0, color[0], color[1], color[2],
            1.0, 0.0, color[0], color[1], color[2],
            1.0, 1.0, color[0], color[1], color[2],
            0.0, 1.0, color[0], color[1], color[2],
        ], dtype=np.float32)
        
        self.dynamic_rect_vbo.write(rect_vertices.tobytes())
        self.dynamic_rect_vao.render()
    
    def _render_button_text(self, button: ModernGLButton):
        """Render button text using texture"""
        try:
            texture_data = self._get_or_create_text_texture(
                button.state.text,
                button.state.font_size,
                button.state.color,
                bold=False,
            )
            if texture_data is None:
                return
            texture, text_width, text_height = texture_data
            
            # Center text on button
            text_x = button.state.x + (button.state.width - text_width) // 2
            text_y = button.state.y + (button.state.height - text_height) // 2
            
            # Render text texture
            self._render_texture(
                texture, text_x, text_y,
                text_width, text_height,
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
        except Exception as e:
            print(f"Text rendering error for button '{button.state.text}': {e}")
    
    def _render_title(self):
        """Render animated title"""
        try:
            font_mgr = get_font_manager()
            title_size = int(64 * self.title_scale)
            font = font_mgr.get_font(title_size, bold=True)
            
            # Render title text
            title_surface = font.render(GAME_TITLE, True, (255, 255, 255))
            shadow_surface = font.render(GAME_TITLE, True, (50, 50, 50))

            if title_surface.get_width() == 0:
                return
            
            # Create textures
            title_texture = self._create_texture_from_surface(title_surface)
            shadow_texture = self._create_texture_from_surface(shadow_surface)
            
            # Calculate position
            title_x = self.width // 2 - title_surface.get_width() // 2
            title_y = 100
            
            # Render shadow first
            self._render_texture(
                shadow_texture, title_x + 3, title_y + 3,
                title_surface.get_width(), title_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            # Render title
            self._render_texture(
                title_texture, title_x, title_y,
                title_surface.get_width(), title_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            title_texture.release()
            shadow_texture.release()
            
        except Exception as e:
            print(f"Title rendering error: {e}")
    
    def _render_subtitle(self):
        """Render subtitle text"""
        try:
            texture_data = self._get_or_create_text_texture(
                "A Minecraft-like Adventure",
                24,
                (200, 200, 200),
                bold=False,
            )
            if texture_data is None:
                return
            texture, subtitle_width, subtitle_height = texture_data
            
            subtitle_x = self.width // 2 - subtitle_width // 2
            subtitle_y = 200
            
            self._render_texture(
                texture, subtitle_x, subtitle_y,
                subtitle_width, subtitle_height,
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
        except Exception as e:
            print(f"Subtitle rendering error: {e}")
    
    def _render_ui_info(self):
        """Render version info and instructions"""
        try:
            # Version info
            version_text = f"Version {VERSION}"
            version_texture_data = self._get_or_create_text_texture(
                version_text,
                16,
                (150, 150, 150),
                bold=False,
            )
            if version_texture_data is not None:
                texture, version_width, version_height = version_texture_data
                self._render_texture(
                    texture,
                    self.width - version_width - 10,
                    self.height - version_height - 10,
                    version_width, version_height,
                    np.array([1.0, 1.0, 1.0], dtype=np.float32)
                )
            
            # Instructions
            instructions = [
                "Use mouse to click buttons",
                "Press Enter for New World",
                "Press ESC to exit"
            ]
            
            y_offset = self.height - 100
            
            for instruction in instructions:
                instruction_texture_data = self._get_or_create_text_texture(
                    instruction,
                    16,
                    (120, 120, 120),
                    bold=False,
                )
                if instruction_texture_data is not None:
                    texture, inst_width, inst_height = instruction_texture_data
                    inst_x = self.width // 2 - inst_width // 2

                    self._render_texture(
                        texture, inst_x, y_offset,
                        inst_width, inst_height,
                        np.array([1.0, 1.0, 1.0], dtype=np.float32)
                    )
                    y_offset += 20
            
        except Exception as e:
            print(f"UI info rendering error: {e}")
    
    def _create_texture_from_surface(self, surface: pygame.Surface):
        """Create ModernGL texture from pygame surface"""
        w, h = surface.get_size()
        texture = self.ctx.texture((w, h), 4)  # RGBA
        texture.filter = (mgl.NEAREST, mgl.NEAREST)
        
        # Convert surface to texture data
        data = pygame.image.tostring(surface, "RGBA", True)  # Flip vertically
        texture.write(data)
        
        return texture

    def _get_or_create_text_texture(
        self,
        text: str,
        font_size: int,
        color: Tuple[int, int, int],
        bold: bool = False,
    ) -> Optional[Tuple[mgl.Texture, int, int]]:
        """Create and cache static text textures to avoid per-frame GPU allocations."""
        cache_key = (text, font_size, color, bold)
        cached = self._text_texture_cache.get(cache_key)
        if cached is not None:
            return cached

        font_mgr = get_font_manager()
        font = font_mgr.get_font(font_size, bold=bold)
        text_surface = font.render(text, True, color)
        width, height = text_surface.get_size()
        if width == 0 or height == 0:
            return None

        texture = self._create_texture_from_surface(text_surface)
        cached_entry = (texture, width, height)
        self._text_texture_cache[cache_key] = cached_entry
        return cached_entry
    
    def _render_texture(self, texture, x: int, y: int, width: int, height: int, color: np.ndarray):
        """Render a texture at specified position"""
        self.texture_shader['ortho_matrix'].write(self.ortho_matrix.T.tobytes())
        self.texture_shader['offset'].write(np.array([x, y], dtype=np.float32).tobytes())
        self.texture_shader['scale'].write(np.array([width, height], dtype=np.float32).tobytes())
        self.texture_shader['text_color'].write(color.tobytes())
        self.texture_shader['alpha'].write(np.array([1.0], dtype=np.float32).tobytes())
        
        # Bind texture
        texture.use(0)
        self.texture_shader['texture_sampler'].value = 0

        # Render quad
        self.texture_vao.render()
    
    def run(self) -> Optional[str]:
        """Run the menu and return the selected option"""
        print("🚀 ModernGL Menu System started")
        print("GPU-accelerated rendering active")
        
        last_time = time.time()
        
        while self.running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            self.handle_events()
            
            # Update
            self.update(dt)
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(60)
        
        # Cleanup
        self._cleanup()
        
        return self.selected_option
    
    def _cleanup(self):
        """Clean up ModernGL resources"""
        try:
            for texture, _, _ in self._text_texture_cache.values():
                texture.release()
            self._text_texture_cache.clear()
            if hasattr(self, 'texture_vao'):
                self.texture_vao.release()
            if hasattr(self, 'dynamic_rect_vao'):
                self.dynamic_rect_vao.release()
            if hasattr(self, 'gradient_vao'):
                self.gradient_vao.release()
            if hasattr(self, 'quad_vbo'):
                self.quad_vbo.release()
            if hasattr(self, 'quad_ibo'):
                self.quad_ibo.release()
            if hasattr(self, 'tex_quad_vbo'):
                self.tex_quad_vbo.release()
            if hasattr(self, 'dynamic_rect_vbo'):
                self.dynamic_rect_vbo.release()
            if hasattr(self, 'fullscreen_vbo'):
                self.fullscreen_vbo.release()
            print("✅ ModernGL resources cleaned up")
        except Exception as e:
            print(f"Cleanup warning: {e}")

class ModernGLPauseMenu(ModernGLMenu):
    """High-performance ModernGL-based pause menu interface"""
    
    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT, screen: pygame.Surface = None):
        # Initialize the parent class
        super().__init__(width, height, screen)
    
    def _create_buttons(self):
        """Create pause menu button objects"""
        button_width = 250
        button_height = 50
        button_spacing = 15
        start_y = self.height // 2 - 60
        
        self.buttons = {
            'resume': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y,
                button_width,
                button_height,
                "Resume Game",
                font_size=26,
                bg_color=(40, 120, 40),
                hover_color=(60, 140, 60)
            ),
            'settings': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y + button_height + button_spacing,
                button_width,
                button_height,
                "Settings",
                font_size=26,
                bg_color=(80, 80, 120),
                hover_color=(100, 100, 140)
            ),
            'save_quit': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y + 2 * (button_height + button_spacing),
                button_width,
                button_height,
                "Save & Quit",
                font_size=26,
                bg_color=(120, 80, 40),
                hover_color=(140, 100, 60)
            ),
            'main_menu': ModernGLButton(
                self.width // 2 - button_width // 2,
                start_y + 3 * (button_height + button_spacing),
                button_width,
                button_height,
                "Exit to Main Menu",
                font_size=26,
                bg_color=(120, 40, 40),
                hover_color=(140, 60, 60)
            )
        }
    
    def handle_events(self):
        """Handle pygame events for pause menu"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.selected_option = 'main_menu'
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # ESC key resumes the game
                    self.selected_option = 'resume'
                    self.running = False
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Enter key resumes the game by default
                    self.selected_option = 'resume'
                    self.running = False
            
            # Handle button events
            for button_name, button in self.buttons.items():
                if button.handle_event(event):
                    self.selected_option = button_name
                    self.running = False
    
    def _render_background(self):
        return  # Override to skip animated background
    
    def _render_title(self):
        """Render animated title"""
        try:
            font_mgr = get_font_manager()
            title_size = int(64 * self.title_scale)
            font = font_mgr.get_font(title_size, bold=True)
            
            # Render title text
            title_surface = font.render("PAUSE", True, (255, 255, 255))
            shadow_surface = font.render("PAUSE", True, (50, 50, 50))

            if title_surface.get_width() == 0:
                return
            
            # Create textures
            title_texture = self._create_texture_from_surface(title_surface)
            shadow_texture = self._create_texture_from_surface(shadow_surface)
            
            # Calculate position
            title_x = self.width // 2 - title_surface.get_width() // 2
            title_y = 100
            
            # Render shadow first
            self._render_texture(
                shadow_texture, title_x + 3, title_y + 3,
                title_surface.get_width(), title_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            # Render title
            self._render_texture(
                title_texture, title_x, title_y,
                title_surface.get_width(), title_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            title_texture.release()
            shadow_texture.release()
            
        except Exception as e:
            print(f"Title rendering error: {e}")
    
    def _render_subtitle(self):
        """Render pause menu subtitle"""
        # Get font manager
        font_manager = get_font_manager()
        
        # Create subtitle text
        subtitle_font_size = 18
        subtitle_text = "Press ESC or click Resume to continue"
        subtitle_surface = font_manager.get_font(subtitle_font_size).render(
            subtitle_text, True, (180, 180, 180)
        )
        
        # Create texture from surface
        subtitle_texture = self._create_texture_from_surface(subtitle_surface)
        
        # Calculate position
        subtitle_width = subtitle_surface.get_width()
        subtitle_height = subtitle_surface.get_height()
        subtitle_x = (self.width - subtitle_width) // 2
        subtitle_y = self.height // 4 + 70
        
        # Render subtitle
        subtitle_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self._render_texture(subtitle_texture, subtitle_x, subtitle_y, subtitle_width, subtitle_height, subtitle_color)
        
        # Clean up texture
        subtitle_texture.release()


class ModernGLPauseSettingsMenu(ModernGLMenu):
    """ModernGL settings panel for pause menu options."""

    def __init__(
        self,
        width: int,
        height: int,
        current_render_distance: int,
        current_fog_distance: float,
        min_render_distance: int,
        max_render_distance: int,
        min_fog_distance: float,
        max_fog_distance: float,
        screen: pygame.Surface = None,
    ):
        self.render_distance_value = int(current_render_distance)
        self.fog_distance_value = float(current_fog_distance)
        self.min_render_distance = int(min_render_distance)
        self.max_render_distance = int(max_render_distance)
        self.min_fog_distance = float(min_fog_distance)
        self.max_fog_distance = float(max_fog_distance)
        self.dragging_slider: Optional[str] = None
        self.active_slider = 'render'
        self.render_slider_rect: Optional[pygame.Rect] = None
        self.fog_slider_rect: Optional[pygame.Rect] = None
        self.back_rect: Optional[pygame.Rect] = None
        super().__init__(width, height, screen)
        self.render_distance_value = max(self.min_render_distance, min(self.max_render_distance, self.render_distance_value))
        self.fog_distance_value = max(self.min_fog_distance, min(self.max_fog_distance, self.fog_distance_value))

    def _create_buttons(self):
        self.buttons = {}

    def _value_to_x(self, value: float, slider_rect: Optional[pygame.Rect], minimum: float, maximum: float) -> int:
        if slider_rect is None:
            return 0
        if maximum == minimum:
            return slider_rect.x
        ratio = (value - minimum) / float(maximum - minimum)
        return int(slider_rect.x + ratio * slider_rect.width)

    def _x_to_value(self, x: int, slider_rect: Optional[pygame.Rect], minimum: float, maximum: float, as_int: bool) -> float:
        if slider_rect is None:
            return minimum
        clamped_x = max(slider_rect.x, min(slider_rect.right, x))
        if slider_rect.width <= 0:
            return minimum
        ratio = (clamped_x - slider_rect.x) / float(slider_rect.width)
        value = minimum + ratio * (maximum - minimum)
        if as_int:
            return float(int(round(value)))
        return float(round(value, 1))

    def _adjust_active_slider(self, delta: int) -> None:
        if self.active_slider == 'fog':
            step = 5.0
            self.fog_distance_value = max(
                self.min_fog_distance,
                min(self.max_fog_distance, self.fog_distance_value + delta * step),
            )
            self.fog_distance_value = float(round(self.fog_distance_value, 1))
        else:
            self.render_distance_value = max(
                self.min_render_distance,
                min(self.max_render_distance, self.render_distance_value + delta),
            )

    def _set_slider_rects(self) -> pygame.Rect:
        panel_rect = pygame.Rect(self.width // 2 - 320, self.height // 2 - 200, 640, 380)
        self.render_slider_rect = pygame.Rect(panel_rect.x + 70, panel_rect.y + 140, panel_rect.width - 140, 12)
        self.fog_slider_rect = pygame.Rect(panel_rect.x + 70, panel_rect.y + 228, panel_rect.width - 140, 12)
        self.back_rect = pygame.Rect(panel_rect.centerx - 90, panel_rect.bottom - 76, 180, 46)
        return panel_rect

    def handle_events(self):
        self._set_slider_rects()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.running = False
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self.active_slider = 'render'
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.active_slider = 'fog'
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self._adjust_active_slider(-1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self._adjust_active_slider(1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                render_knob_x = self._value_to_x(
                    float(self.render_distance_value),
                    self.render_slider_rect,
                    float(self.min_render_distance),
                    float(self.max_render_distance),
                )
                render_knob_rect = pygame.Rect(render_knob_x - 14, self.render_slider_rect.centery - 14, 28, 28)

                fog_knob_x = self._value_to_x(
                    self.fog_distance_value,
                    self.fog_slider_rect,
                    self.min_fog_distance,
                    self.max_fog_distance,
                )
                fog_knob_rect = pygame.Rect(fog_knob_x - 14, self.fog_slider_rect.centery - 14, 28, 28)

                if self.back_rect.collidepoint(event.pos):
                    self.running = False
                elif render_knob_rect.collidepoint(event.pos) or self.render_slider_rect.inflate(0, 24).collidepoint(event.pos):
                    self.active_slider = 'render'
                    self.dragging_slider = 'render'
                    self.render_distance_value = int(self._x_to_value(
                        event.pos[0],
                        self.render_slider_rect,
                        float(self.min_render_distance),
                        float(self.max_render_distance),
                        as_int=True,
                    ))
                elif fog_knob_rect.collidepoint(event.pos) or self.fog_slider_rect.inflate(0, 24).collidepoint(event.pos):
                    self.active_slider = 'fog'
                    self.dragging_slider = 'fog'
                    self.fog_distance_value = self._x_to_value(
                        event.pos[0],
                        self.fog_slider_rect,
                        self.min_fog_distance,
                        self.max_fog_distance,
                        as_int=False,
                    )
            elif event.type == pygame.MOUSEMOTION and self.dragging_slider:
                if self.dragging_slider == 'render':
                    self.render_distance_value = int(self._x_to_value(
                        event.pos[0],
                        self.render_slider_rect,
                        float(self.min_render_distance),
                        float(self.max_render_distance),
                        as_int=True,
                    ))
                elif self.dragging_slider == 'fog':
                    self.fog_distance_value = self._x_to_value(
                        event.pos[0],
                        self.fog_slider_rect,
                        self.min_fog_distance,
                        self.max_fog_distance,
                        as_int=False,
                    )
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging_slider = None

    def _render_background(self):
        return

    def _render_title(self):
        return

    def _render_subtitle(self):
        return

    def render(self):
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        self.ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        self.ctx.viewport = (0, 0, self.width, self.height)

        panel_rect = self._set_slider_rects()

        # Dark pause overlay
        self._render_rect(0, 0, self.width, self.height, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.rect_shader['alpha'].write(np.array([0.45], dtype=np.float32).tobytes())
        overlay_vertices = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        self.dynamic_rect_vbo.write(overlay_vertices.tobytes())
        self.rect_shader['ortho_matrix'].write(self.ortho_matrix.T.tobytes())
        self.rect_shader['offset'].write(np.array([0, 0], dtype=np.float32).tobytes())
        self.rect_shader['scale'].write(np.array([self.width, self.height], dtype=np.float32).tobytes())
        self.dynamic_rect_vao.render()

        self._render_rect(panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height, np.array([0.125, 0.157, 0.266], dtype=np.float32))
        self._render_rect(panel_rect.x, panel_rect.y, panel_rect.width, 2, np.array([0.6, 0.66, 0.82], dtype=np.float32))
        self._render_rect(panel_rect.x, panel_rect.bottom - 2, panel_rect.width, 2, np.array([0.6, 0.66, 0.82], dtype=np.float32))

        title_data = self._get_or_create_text_texture("SETTINGS", 64, (245, 245, 255), bold=True)
        if title_data is not None:
            texture, tw, th = title_data
            self._render_texture(texture, self.width // 2 - tw // 2, panel_rect.y + 16, tw, th, np.array([1.0, 1.0, 1.0], dtype=np.float32))

        render_label_text = f"Render Distance: {self.render_distance_value} chunks"
        render_label_data = self._get_or_create_text_texture(render_label_text, 36, (235, 240, 255), bold=False)
        if render_label_data is not None:
            texture, lw, lh = render_label_data
            self._render_texture(texture, self.width // 2 - lw // 2, panel_rect.y + 90, lw, lh, np.array([1.0, 1.0, 1.0], dtype=np.float32))

        fog_label_text = f"Fog Distance: {self.fog_distance_value:.1f}"
        fog_label_data = self._get_or_create_text_texture(fog_label_text, 36, (235, 240, 255), bold=False)
        if fog_label_data is not None:
            texture, lw, lh = fog_label_data
            self._render_texture(texture, self.width // 2 - lw // 2, panel_rect.y + 178, lw, lh, np.array([1.0, 1.0, 1.0], dtype=np.float32))

        render_track_color = np.array([0.31, 0.35, 0.47], dtype=np.float32)
        fog_track_color = np.array([0.31, 0.35, 0.47], dtype=np.float32)
        if self.active_slider == 'render':
            render_track_color = np.array([0.36, 0.43, 0.59], dtype=np.float32)
        if self.active_slider == 'fog':
            fog_track_color = np.array([0.36, 0.43, 0.59], dtype=np.float32)

        self._render_rect(self.render_slider_rect.x, self.render_slider_rect.y, self.render_slider_rect.width, self.render_slider_rect.height, render_track_color)
        render_knob_x = self._value_to_x(
            float(self.render_distance_value),
            self.render_slider_rect,
            float(self.min_render_distance),
            float(self.max_render_distance),
        )
        render_filled_width = max(1, render_knob_x - self.render_slider_rect.x)
        self._render_rect(self.render_slider_rect.x, self.render_slider_rect.y, render_filled_width, self.render_slider_rect.height, np.array([0.38, 0.58, 0.93], dtype=np.float32))
        self._render_rect(render_knob_x - 10, self.render_slider_rect.centery - 10, 20, 20, np.array([0.93, 0.95, 1.0], dtype=np.float32))

        self._render_rect(self.fog_slider_rect.x, self.fog_slider_rect.y, self.fog_slider_rect.width, self.fog_slider_rect.height, fog_track_color)
        fog_knob_x = self._value_to_x(
            self.fog_distance_value,
            self.fog_slider_rect,
            self.min_fog_distance,
            self.max_fog_distance,
        )
        fog_filled_width = max(1, fog_knob_x - self.fog_slider_rect.x)
        self._render_rect(self.fog_slider_rect.x, self.fog_slider_rect.y, fog_filled_width, self.fog_slider_rect.height, np.array([0.38, 0.58, 0.93], dtype=np.float32))
        self._render_rect(fog_knob_x - 10, self.fog_slider_rect.centery - 10, 20, 20, np.array([0.93, 0.95, 1.0], dtype=np.float32))

        hint_text = (
            f"Up/Down select slider, Left/Right adjust | Render {self.min_render_distance}-{self.max_render_distance}, "
            f"Fog {self.min_fog_distance:.0f}-{self.max_fog_distance:.0f}"
        )
        hint_data = self._get_or_create_text_texture(hint_text, 20, (188, 194, 220), bold=False)
        if hint_data is not None:
            texture, hw, hh = hint_data
            self._render_texture(texture, self.width // 2 - hw // 2, panel_rect.y + 274, hw, hh, np.array([1.0, 1.0, 1.0], dtype=np.float32))

        mouse_pos = pygame.mouse.get_pos()
        back_hovered = self.back_rect.collidepoint(mouse_pos)
        back_bg = np.array([0.32, 0.42, 0.64], dtype=np.float32) if back_hovered else np.array([0.24, 0.30, 0.47], dtype=np.float32)
        self._render_rect(self.back_rect.x, self.back_rect.y, self.back_rect.width, self.back_rect.height, back_bg)
        back_data = self._get_or_create_text_texture("Back", 34, (255, 255, 255), bold=False)
        if back_data is not None:
            texture, bw, bh = back_data
            self._render_texture(
                texture,
                self.back_rect.centerx - bw // 2,
                self.back_rect.centery - bh // 2,
                bw,
                bh,
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )

        pygame.display.flip()

    def run(self) -> Dict[str, float]:
        last_time = time.time()
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            self.handle_events()
            self.update(dt)
            self.render()
            self.clock.tick(60)
        self._cleanup()
        return {
            'render_distance': float(self.render_distance_value),
            'fog_distance': float(self.fog_distance_value),
        }


class ModernGLLoadMenu(ModernGLMenu):
    """ModernGL-powered load world selector."""

    def __init__(self, width: int, height: int, saves: List[SaveMetadata], screen: pygame.Surface = None):
        self.saves = saves
        self.selected_index = 0 if saves else -1
        self.hover_index = -1
        self.visible_offset = 0
        self.max_visible = 6
        self._item_rects: List[Tuple[pygame.Rect, int]] = []
        super().__init__(width, height, screen)
        pygame.display.set_caption("Pycraft - Load World")
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        self._ensure_selection_visible()

    def _create_buttons(self):
        # No traditional buttons; navigation handled via list and keyboard/mouse.
        self.buttons = {}

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.selected_option = None
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    self.selected_option = None
                    self.running = False
                elif self.saves and event.key in (pygame.K_UP, pygame.K_w):
                    self._change_selection(-1)
                elif self.saves and event.key in (pygame.K_DOWN, pygame.K_s):
                    self._change_selection(1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self.saves and self.selected_index >= 0:
                        self.selected_option = self.saves[self.selected_index].identifier
                    else:
                        self.selected_option = None
                    self.running = False
            elif event.type == pygame.MOUSEWHEEL and self.saves:
                direction = -event.y
                if direction != 0:
                    self._change_selection(direction)
            elif event.type == pygame.MOUSEMOTION and self.saves:
                self.hover_index = -1
                for rect, idx in self._item_rects:
                    if rect.collidepoint(event.pos):
                        self.hover_index = idx
                        break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.saves:
                for rect, idx in self._item_rects:
                    if rect.collidepoint(event.pos):
                        self.selected_index = idx
                        self._ensure_selection_visible()
                        self.selected_option = self.saves[idx].identifier
                        self.running = False
                        break

    def _change_selection(self, delta: int):
        if not self.saves:
            return

        if self.selected_index < 0:
            self.selected_index = 0
        else:
            self.selected_index = (self.selected_index + delta) % len(self.saves)
        self._ensure_selection_visible()

    def _ensure_selection_visible(self):
        if not self.saves:
            self.visible_offset = 0
            return

        if self.selected_index < 0:
            self.selected_index = 0

        if self.selected_index < self.visible_offset:
            self.visible_offset = self.selected_index
        elif self.selected_index >= self.visible_offset + self.max_visible:
            self.visible_offset = self.selected_index - self.max_visible + 1

        max_offset = max(0, len(self.saves) - self.max_visible)
        self.visible_offset = max(0, min(self.visible_offset, max_offset))

    def render(self):
        self.ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        self.ctx.viewport = (0, 0, self.width, self.height)

        self._render_background()
        self._render_title()

        if self.saves:
            self._render_save_list()
        else:
            self._render_empty_state()

        self._render_footer()
        pygame.display.flip()

    def _render_title(self):
        try:
            font_mgr = get_font_manager()
            title_size = int(52 * self.title_scale)
            font = font_mgr.get_font(title_size, bold=True)
            title_surface = font.render("Load Saved World", True, (255, 255, 255))
            shadow_surface = font.render("Load Saved World", True, (50, 50, 60))

            if title_surface.get_width() == 0:
                return

            shadow_texture = self._create_texture_from_surface(shadow_surface)
            title_texture = self._create_texture_from_surface(title_surface)

            title_x = self.width // 2 - title_surface.get_width() // 2
            title_y = 90

            self._render_texture(
                shadow_texture,
                title_x + 3,
                title_y + 3,
                title_surface.get_width(),
                title_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )

            self._render_texture(
                title_texture,
                title_x,
                title_y,
                title_surface.get_width(),
                title_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )

            shadow_texture.release()
            title_texture.release()
        except Exception as exc:
            print(f"Title rendering error for load menu: {exc}")

    def _render_save_list(self):
        font_mgr = get_font_manager()
        name_font = font_mgr.get_font(28, bold=True)
        meta_font = font_mgr.get_font(18)

        list_width = min(int(self.width * 0.65), self.width - 120)
        item_height = 74
        spacing = 12

        visible_count = min(self.max_visible, len(self.saves) - self.visible_offset)
        total_height = visible_count * item_height + max(0, visible_count - 1) * spacing
        start_y = max(160, (self.height - total_height) // 2)
        start_x = (self.width - list_width) // 2

        base_color = np.array([36 / 255.0, 48 / 255.0, 80 / 255.0], dtype=np.float32)
        hover_color = np.array([48 / 255.0, 64 / 255.0, 104 / 255.0], dtype=np.float32)
        selected_color = np.array([72 / 255.0, 100 / 255.0, 168 / 255.0], dtype=np.float32)
        border_color = np.array([150 / 255.0, 160 / 255.0, 200 / 255.0], dtype=np.float32)

        self._item_rects = []

        for row in range(visible_count):
            idx = self.visible_offset + row
            save = self.saves[idx]
            item_y = start_y + row * (item_height + spacing)
            rect = pygame.Rect(start_x, item_y, list_width, item_height)
            self._item_rects.append((rect, idx))

            is_selected = idx == self.selected_index
            is_hovered = idx == self.hover_index and not is_selected
            fill_color = selected_color if is_selected else hover_color if is_hovered else base_color

            self._render_rect(rect.x, rect.y, rect.width, rect.height, fill_color)
            self._render_rect(rect.x, rect.y, rect.width, 3, border_color)
            self._render_rect(rect.x, rect.y + rect.height - 3, rect.width, 3, border_color)

            if is_selected:
                glow_color = np.array([120 / 255.0, 170 / 255.0, 255 / 255.0], dtype=np.float32)
                self._render_rect(rect.x + rect.width - 6, rect.y, 6, rect.height, glow_color)
                self._render_selection_cursor(rect)

            name_surface = name_font.render(save.display_name, True, (255, 255, 255))
            if name_surface.get_width() > 0:
                name_texture = self._create_texture_from_surface(name_surface)
                self._render_texture(
                    name_texture,
                    rect.x + 24,
                    rect.y + 14,
                    name_surface.get_width(),
                    name_surface.get_height(),
                    np.array([1.0, 1.0, 1.0], dtype=np.float32)
                )
                name_texture.release()

            meta_text = f"Updated {_format_timestamp(save.updated_at)}"
            if save.identifier:
                meta_text += f"  •  ID: {save.identifier}"
            meta_surface = meta_font.render(meta_text, True, (205, 210, 235))
            if meta_surface.get_width() > 0:
                meta_texture = self._create_texture_from_surface(meta_surface)
                self._render_texture(
                    meta_texture,
                    rect.x + 24,
                    rect.y + rect.height - meta_surface.get_height() - 12,
                    meta_surface.get_width(),
                    meta_surface.get_height(),
                    np.array([1.0, 1.0, 1.0], dtype=np.float32)
                )
                meta_texture.release()

    def _render_empty_state(self):
        font_mgr = get_font_manager()
        message_font = font_mgr.get_font(32, bold=True)
        hint_font = font_mgr.get_font(22)

        message_surface = message_font.render("No saved worlds found", True, (235, 235, 255))
        hint_surface = hint_font.render("Press ESC to return to the main menu", True, (190, 195, 215))

        for surface, y_offset in ((message_surface, -20), (hint_surface, 30)):
            if surface.get_width() == 0:
                continue
            texture = self._create_texture_from_surface(surface)
            self._render_texture(
                texture,
                self.width // 2 - surface.get_width() // 2,
                self.height // 2 + y_offset,
                surface.get_width(),
                surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            texture.release()

    def _render_selection_cursor(self, rect: pygame.Rect) -> None:
        pointer_width = 28
        pointer_surface = pygame.Surface((pointer_width, rect.height), pygame.SRCALPHA)

        pulse = 0.5 + 0.5 * np.sin(self.title_time * 4.0)
        base_color = np.array([0.40, 0.65, 1.0])
        highlight_color = np.clip(base_color + pulse * 0.15, 0.0, 1.0)
        rgba = tuple(int(c * 255) for c in highlight_color) + (int(180 + 60 * pulse),)

        pygame.draw.polygon(
            pointer_surface,
            rgba,
            [
                (0, rect.height // 2),
                (pointer_width, 6),
                (pointer_width, rect.height - 6),
            ],
        )

        pointer_texture = self._create_texture_from_surface(pointer_surface)
        pointer_x = max(rect.x - pointer_width - 12, 0)
        self._render_texture(
            pointer_texture,
            pointer_x,
            rect.y,
            pointer_surface.get_width(),
            pointer_surface.get_height(),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        pointer_texture.release()

    def _render_footer(self):
        font_mgr = get_font_manager()
        hint_font = font_mgr.get_font(18)

        hints = [
            "Use ↑/↓ or the mouse wheel to navigate",
            "Press Enter or click to load the selected world",
            "Press ESC to cancel"
        ]

        start_y = self.height - 110
        for i, text in enumerate(hints):
            surface = hint_font.render(text, True, (180, 185, 210))
            if surface.get_width() == 0:
                continue
            texture = self._create_texture_from_surface(surface)
            self._render_texture(
                texture,
                self.width // 2 - surface.get_width() // 2,
                start_y + i * 26,
                surface.get_width(),
                surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            texture.release()


def _format_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError):
        return "unknown"


def _show_load_world_menu(width: int, height: int, screen: pygame.Surface) -> Optional[str]:
    """Show the load-world selector using ModernGL with pygame fallback."""

    saves: List[SaveMetadata] = list_saves()

    try:
        menu = ModernGLLoadMenu(width, height, saves, screen=screen)
        return menu.run()
    except Exception as exc:
        print(f"⚠️ ModernGL load menu failed: {exc}")
        print("📱 Falling back to simple pygame load menu")
        return _show_simple_load_world_menu(width, height, screen)


def _show_simple_load_world_menu(width: int, height: int, screen: pygame.Surface) -> Optional[str]:
    """Fallback pygame load menu when ModernGL is unavailable."""

    saves: List[SaveMetadata] = list_saves()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pycraft - Load World")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)

    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 64)
    item_font = pygame.font.Font(None, 36)
    meta_font = pygame.font.Font(None, 24)
    hint_font = pygame.font.Font(None, 24)

    if not saves:
        message_surface = item_font.render("No saved worlds found.", True, (220, 220, 230))
        hint_surface = hint_font.render("Press ESC to return to the main menu.", True, (180, 180, 190))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER):
                        return None

            screen.fill((18, 20, 30))
            title_surface = title_font.render("Load Saved World", True, (240, 240, 255))
            screen.blit(title_surface, title_surface.get_rect(center=(width // 2, height // 3)))
            screen.blit(message_surface, message_surface.get_rect(center=(width // 2, height // 2)))
            screen.blit(hint_surface, hint_surface.get_rect(center=(width // 2, height // 2 + 50)))

            pygame.display.flip()
            clock.tick(60)

    selected_index = 0
    max_visible = 6

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    return None
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected_index = (selected_index - 1) % len(saves)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected_index = (selected_index + 1) % len(saves)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return saves[selected_index].identifier

        screen.fill((15, 18, 30))

        title_surface = title_font.render("Load Saved World", True, (235, 235, 255))
        screen.blit(title_surface, title_surface.get_rect(center=(width // 2, 100)))

        hint_text = "Use ↑/↓ to choose a save, Enter to load, ESC to cancel"
        hint_surface = hint_font.render(hint_text, True, (170, 170, 190))
        screen.blit(hint_surface, hint_surface.get_rect(center=(width // 2, height - 60)))

        if len(saves) <= max_visible:
            start = 0
            end = len(saves)
        else:
            start = max(0, selected_index - max_visible // 2)
            end = start + max_visible
            if end > len(saves):
                end = len(saves)
                start = max(0, end - max_visible)

        list_width = 520
        item_height = 70
        top_offset = 170
        item_x = width // 2 - list_width // 2

        for visible_idx, save_idx in enumerate(range(start, end)):
            save = saves[save_idx]
            item_y = top_offset + visible_idx * (item_height + 10)
            rect = pygame.Rect(item_x, item_y, list_width, item_height)
            is_selected = save_idx == selected_index

            bg_color = (60, 80, 140) if is_selected else (40, 50, 80)
            border_color = (255, 255, 255) if is_selected else (120, 120, 150)

            pygame.draw.rect(screen, bg_color, rect, border_radius=10)
            pygame.draw.rect(screen, border_color, rect, 2, border_radius=10)

            name_surface = item_font.render(save.display_name, True, (255, 255, 255))
            screen.blit(name_surface, (rect.x + 20, rect.y + 12))

            meta_text = f"Updated { _format_timestamp(save.updated_at) }"
            meta_surface = meta_font.render(meta_text, True, (200, 200, 220))
            screen.blit(meta_surface, (rect.x + 20, rect.y + 40))

        pygame.display.flip()
        clock.tick(60)


def show_main_menu(width: int = 1024, height: int = 768, screen: Optional[pygame.Surface] = None) -> Optional[str]:
    """Show the main menu and return the selected option with automatic fallback"""

    while True:
        try:
            print("🚀 Attempting ModernGL GPU-accelerated menu...")
            menu = ModernGLMenu(width, height, screen)
            result = menu.run()
        except ImportError as e:
            print(f"⚠️ ModernGL not available: {e}")
            print("📱 Falling back to standard pygame menu")
            return None
        except RuntimeError as e:
            if "OpenGL" in str(e):
                print(f"⚠️ OpenGL context error: {e}")
                print("📱 Falling back to standard pygame menu")
                return None
            raise e
        except Exception as e:
            print(f"⚠️ ModernGL menu failed: {e}")
            print("📱 Falling back to standard pygame menu")
            return None

        if result == 'load_world':
            current_screen = pygame.display.get_surface() or screen
            selected_save = _show_load_world_menu(width, height, current_screen)
            if selected_save:
                return f"load_world:{selected_save}"
            # User cancelled save selection; restart the main menu loop
            continue

        return result


def _run_simple_pause_menu(width: int, height: int) -> Optional[str]:
    """Fallback pygame pause menu when ModernGL path is unavailable."""
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pycraft - Pause Menu")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)

    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 72)
    button_font = pygame.font.Font(None, 42)
    hint_font = pygame.font.Font(None, 28)

    options = [
        ("resume", "Resume Game"),
        ("settings", "Settings"),
        ("save_quit", "Save & Quit"),
        ("main_menu", "Exit to Main Menu"),
    ]
    selected_index = 0

    button_width = 360
    button_height = 64
    button_spacing = 18
    start_y = height // 2 - 80
    start_x = width // 2 - button_width // 2

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "main_menu"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "resume"
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected_index = (selected_index - 1) % len(options)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected_index = (selected_index + 1) % len(options)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return options[selected_index][0]

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, (option_key, _label) in enumerate(options):
                    rect = pygame.Rect(
                        start_x,
                        start_y + i * (button_height + button_spacing),
                        button_width,
                        button_height,
                    )
                    if rect.collidepoint(event.pos):
                        return option_key

        screen.fill((16, 18, 30))

        # Dimmed overlay look for pause state.
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        screen.blit(overlay, (0, 0))

        title_surface = title_font.render("PAUSED", True, (245, 245, 255))
        screen.blit(title_surface, title_surface.get_rect(center=(width // 2, 120)))

        hint_surface = hint_font.render("ESC to resume", True, (185, 185, 200))
        screen.blit(hint_surface, hint_surface.get_rect(center=(width // 2, height - 70)))

        for i, (_key, label) in enumerate(options):
            rect = pygame.Rect(
                start_x,
                start_y + i * (button_height + button_spacing),
                button_width,
                button_height,
            )

            hovered = rect.collidepoint(mouse_pos)
            selected = i == selected_index
            active = hovered or selected

            bg_color = (68, 96, 160) if active else (44, 52, 86)
            border_color = (245, 245, 255) if active else (140, 150, 180)

            pygame.draw.rect(screen, bg_color, rect, border_radius=10)
            pygame.draw.rect(screen, border_color, rect, 2, border_radius=10)

            label_surface = button_font.render(label, True, (255, 255, 255))
            screen.blit(label_surface, label_surface.get_rect(center=rect.center))

        pygame.display.flip()
        clock.tick(60)


def _show_pause_settings_menu(
    width: int,
    height: int,
    current_render_distance: int,
    current_fog_distance: float,
    min_render_distance: int,
    max_render_distance: int,
    min_fog_distance: float,
    max_fog_distance: float,
) -> Dict[str, float]:
    """Fallback pygame settings page with render-distance and fog sliders."""
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pycraft - Settings")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)

    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 72)
    label_font = pygame.font.Font(None, 44)
    hint_font = pygame.font.Font(None, 28)
    button_font = pygame.font.Font(None, 36)

    render_value = max(min_render_distance, min(max_render_distance, int(current_render_distance)))
    fog_value = max(min_fog_distance, min(max_fog_distance, float(current_fog_distance)))
    active_slider = 'render'

    panel_rect = pygame.Rect(width // 2 - 320, height // 2 - 200, 640, 380)
    render_slider_rect = pygame.Rect(panel_rect.x + 70, panel_rect.y + 140, panel_rect.width - 140, 12)
    fog_slider_rect = pygame.Rect(panel_rect.x + 70, panel_rect.y + 228, panel_rect.width - 140, 12)
    knob_radius = 14
    dragging_slider: Optional[str] = None

    back_rect = pygame.Rect(panel_rect.centerx - 90, panel_rect.bottom - 76, 180, 46)

    def _value_to_x(value: float, slider_rect: pygame.Rect, minimum: float, maximum: float) -> int:
        if maximum == minimum:
            return slider_rect.x
        ratio = (value - minimum) / float(maximum - minimum)
        return int(slider_rect.x + ratio * slider_rect.width)

    def _x_to_value(x: int, slider_rect: pygame.Rect, minimum: float, maximum: float, as_int: bool) -> float:
        clamped_x = max(slider_rect.x, min(slider_rect.right, x))
        if slider_rect.width <= 0:
            return minimum
        ratio = (clamped_x - slider_rect.x) / float(slider_rect.width)
        value = minimum + ratio * (maximum - minimum)
        if as_int:
            return float(int(round(value)))
        return float(round(value, 1))

    def _result() -> Dict[str, float]:
        return {
            'render_distance': float(render_value),
            'fog_distance': float(fog_value),
        }

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return _result()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER):
                    return _result()
                if event.key in (pygame.K_UP, pygame.K_w):
                    active_slider = 'render'
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    active_slider = 'fog'
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    if active_slider == 'fog':
                        fog_value = max(min_fog_distance, fog_value - 5.0)
                    else:
                        render_value = max(min_render_distance, render_value - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    if active_slider == 'fog':
                        fog_value = min(max_fog_distance, fog_value + 5.0)
                    else:
                        render_value = min(max_render_distance, render_value + 1)
                fog_value = float(round(fog_value, 1))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                render_knob_x = _value_to_x(float(render_value), render_slider_rect, float(min_render_distance), float(max_render_distance))
                render_knob_rect = pygame.Rect(render_knob_x - knob_radius, render_slider_rect.centery - knob_radius, knob_radius * 2, knob_radius * 2)
                fog_knob_x = _value_to_x(fog_value, fog_slider_rect, min_fog_distance, max_fog_distance)
                fog_knob_rect = pygame.Rect(fog_knob_x - knob_radius, fog_slider_rect.centery - knob_radius, knob_radius * 2, knob_radius * 2)

                if back_rect.collidepoint(event.pos):
                    return _result()

                if render_knob_rect.collidepoint(event.pos) or render_slider_rect.inflate(0, 24).collidepoint(event.pos):
                    active_slider = 'render'
                    dragging_slider = 'render'
                    render_value = int(_x_to_value(
                        event.pos[0],
                        render_slider_rect,
                        float(min_render_distance),
                        float(max_render_distance),
                        as_int=True,
                    ))
                elif fog_knob_rect.collidepoint(event.pos) or fog_slider_rect.inflate(0, 24).collidepoint(event.pos):
                    active_slider = 'fog'
                    dragging_slider = 'fog'
                    fog_value = _x_to_value(
                        event.pos[0],
                        fog_slider_rect,
                        min_fog_distance,
                        max_fog_distance,
                        as_int=False,
                    )

            if event.type == pygame.MOUSEMOTION and dragging_slider:
                if dragging_slider == 'render':
                    render_value = int(_x_to_value(
                        event.pos[0],
                        render_slider_rect,
                        float(min_render_distance),
                        float(max_render_distance),
                        as_int=True,
                    ))
                elif dragging_slider == 'fog':
                    fog_value = _x_to_value(
                        event.pos[0],
                        fog_slider_rect,
                        min_fog_distance,
                        max_fog_distance,
                        as_int=False,
                    )

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_slider = None

        screen.fill((14, 17, 30))

        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        screen.blit(overlay, (0, 0))

        pygame.draw.rect(screen, (32, 40, 68), panel_rect, border_radius=16)
        pygame.draw.rect(screen, (155, 170, 210), panel_rect, 2, border_radius=16)

        title_surface = title_font.render("SETTINGS", True, (245, 245, 255))
        screen.blit(title_surface, title_surface.get_rect(center=(width // 2, panel_rect.y + 46)))

        render_label_text = f"Render Distance: {render_value} chunks"
        render_label_surface = label_font.render(render_label_text, True, (235, 240, 255))
        screen.blit(render_label_surface, render_label_surface.get_rect(center=(width // 2, panel_rect.y + 110)))

        fog_label_text = f"Fog Distance: {fog_value:.1f}"
        fog_label_surface = label_font.render(fog_label_text, True, (235, 240, 255))
        screen.blit(fog_label_surface, fog_label_surface.get_rect(center=(width // 2, panel_rect.y + 198)))

        render_track_color = (92, 110, 156) if active_slider == 'render' else (78, 88, 120)
        fog_track_color = (92, 110, 156) if active_slider == 'fog' else (78, 88, 120)

        pygame.draw.rect(screen, render_track_color, render_slider_rect, border_radius=6)
        render_knob_x = _value_to_x(float(render_value), render_slider_rect, float(min_render_distance), float(max_render_distance))
        render_filled_rect = pygame.Rect(render_slider_rect.x, render_slider_rect.y, max(1, render_knob_x - render_slider_rect.x), render_slider_rect.height)
        pygame.draw.rect(screen, (96, 148, 238), render_filled_rect, border_radius=6)
        pygame.draw.circle(screen, (236, 243, 255), (render_knob_x, render_slider_rect.centery), knob_radius)
        pygame.draw.circle(screen, (94, 120, 186), (render_knob_x, render_slider_rect.centery), knob_radius, 2)

        pygame.draw.rect(screen, fog_track_color, fog_slider_rect, border_radius=6)
        fog_knob_x = _value_to_x(fog_value, fog_slider_rect, min_fog_distance, max_fog_distance)
        fog_filled_rect = pygame.Rect(fog_slider_rect.x, fog_slider_rect.y, max(1, fog_knob_x - fog_slider_rect.x), fog_slider_rect.height)
        pygame.draw.rect(screen, (96, 148, 238), fog_filled_rect, border_radius=6)
        pygame.draw.circle(screen, (236, 243, 255), (fog_knob_x, fog_slider_rect.centery), knob_radius)
        pygame.draw.circle(screen, (94, 120, 186), (fog_knob_x, fog_slider_rect.centery), knob_radius, 2)

        hint_text = (
            f"Up/Down select slider, Left/Right adjust | Render {min_render_distance}-{max_render_distance}, "
            f"Fog {min_fog_distance:.0f}-{max_fog_distance:.0f}"
        )
        hint_surface = hint_font.render(hint_text, True, (188, 194, 220))
        screen.blit(hint_surface, hint_surface.get_rect(center=(width // 2, panel_rect.y + 282)))

        back_hovered = back_rect.collidepoint(mouse_pos)
        back_bg = (82, 106, 162) if back_hovered else (60, 76, 120)
        back_border = (235, 240, 255) if back_hovered else (145, 158, 196)
        pygame.draw.rect(screen, back_bg, back_rect, border_radius=10)
        pygame.draw.rect(screen, back_border, back_rect, 2, border_radius=10)
        back_surface = button_font.render("Back", True, (255, 255, 255))
        screen.blit(back_surface, back_surface.get_rect(center=back_rect.center))

        pygame.display.flip()
        clock.tick(60)


def show_pause_menu(
    width: int = 1024,
    height: int = 768,
    screen: Optional[pygame.Surface] = None,
    current_render_distance: int = RENDER_DISTANCE,
    current_fog_distance: float = FOG_DISTANCE,
    min_render_distance: int = MIN_RENDER_DISTANCE,
    max_render_distance: int = MAX_RENDER_DISTANCE,
    min_fog_distance: float = MIN_FOG_DISTANCE,
    max_fog_distance: float = MAX_FOG_DISTANCE,
) -> Optional[Dict[str, object]]:
    """Show the pause menu and return action + settings values."""
    render_distance = max(min_render_distance, min(max_render_distance, int(current_render_distance)))
    fog_distance = max(min_fog_distance, min(max_fog_distance, float(current_fog_distance)))

    while True:
        menu_result: Optional[str]
        try:
            print("🚀 Attempting ModernGL GPU-accelerated pause menu...")
            pause_menu = ModernGLPauseMenu(width, height, screen)
            menu_result = pause_menu.run()
        except RuntimeError as e:
            if "OpenGL" in str(e):
                print(f"⚠️ OpenGL context error in pause menu: {e}")
                print("📱 Falling back to simple pause menu")
            else:
                raise e
            menu_result = _run_simple_pause_menu(width, height)
        except ImportError as e:
            print(f"⚠️ ModernGL not available for pause menu: {e}")
            print("📱 Falling back to simple pause menu")
            menu_result = _run_simple_pause_menu(width, height)
        except Exception as e:
            print(f"⚠️ ModernGL pause menu failed: {e}")
            print("📱 Falling back to simple pause menu")
            menu_result = _run_simple_pause_menu(width, height)

        if menu_result == 'settings':
            current_surface = pygame.display.get_surface()
            is_opengl_surface = bool(current_surface and (current_surface.get_flags() & pygame.OPENGL))
            if is_opengl_surface:
                try:
                    settings_menu = ModernGLPauseSettingsMenu(
                        width,
                        height,
                        current_render_distance=render_distance,
                        current_fog_distance=fog_distance,
                        min_render_distance=min_render_distance,
                        max_render_distance=max_render_distance,
                        min_fog_distance=min_fog_distance,
                        max_fog_distance=max_fog_distance,
                        screen=current_surface,
                    )
                    settings_values = settings_menu.run()
                    render_distance = int(settings_values.get('render_distance', render_distance))
                    fog_distance = float(settings_values.get('fog_distance', fog_distance))
                except Exception as settings_error:
                    print(f"⚠️ ModernGL settings menu failed: {settings_error}")
                    print("📱 Falling back to simple settings menu")
                    settings_values = _show_pause_settings_menu(
                        width,
                        height,
                        current_render_distance=render_distance,
                        current_fog_distance=fog_distance,
                        min_render_distance=min_render_distance,
                        max_render_distance=max_render_distance,
                        min_fog_distance=min_fog_distance,
                        max_fog_distance=max_fog_distance,
                    )
                    render_distance = int(settings_values.get('render_distance', render_distance))
                    fog_distance = float(settings_values.get('fog_distance', fog_distance))
            else:
                settings_values = _show_pause_settings_menu(
                    width,
                    height,
                    current_render_distance=render_distance,
                    current_fog_distance=fog_distance,
                    min_render_distance=min_render_distance,
                    max_render_distance=max_render_distance,
                    min_fog_distance=min_fog_distance,
                    max_fog_distance=max_fog_distance,
                )
                render_distance = int(settings_values.get('render_distance', render_distance))
                fog_distance = float(settings_values.get('fog_distance', fog_distance))
            continue

        return {
            'action': menu_result,
            'render_distance': render_distance,
            'fog_distance': fog_distance,
        }