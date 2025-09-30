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
        
        # Fullscreen quad for background
        fullscreen_vertices = np.array([
            0.0, 0.0,
            self.width, 0.0,
            self.width, self.height,
            0.0, self.height
        ], dtype=np.float32)
        
        self.fullscreen_vbo = self.ctx.buffer(fullscreen_vertices.tobytes())
        
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
        
        # Create VAO for fullscreen quad
        vao = self.ctx.vertex_array(
            self.gradient_shader,
            [(self.fullscreen_vbo, '2f', 'position')],
            self.quad_ibo
        )
        
        vao.render()
        vao.release()
    
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
        
        # Create VAO with color data
        button_vertices = np.array([
            # Position, Color
            0.0, 0.0, norm_color[0], norm_color[1], norm_color[2],
            1.0, 0.0, norm_color[0], norm_color[1], norm_color[2],
            1.0, 1.0, norm_color[0], norm_color[1], norm_color[2],
            0.0, 1.0, norm_color[0], norm_color[1], norm_color[2],
        ], dtype=np.float32)
        
        button_vbo = self.ctx.buffer(button_vertices.tobytes())
        
        vao = self.ctx.vertex_array(
            self.rect_shader,
            [(button_vbo, '2f 3f', 'position', 'color')],
            self.quad_ibo
        )
        
        vao.render()
        vao.release()
        button_vbo.release()
        
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
        self.rect_shader['offset'].write(np.array([x, y], dtype=np.float32).tobytes())
        self.rect_shader['scale'].write(np.array([width, height], dtype=np.float32).tobytes())
        
        rect_vertices = np.array([
            0.0, 0.0, color[0], color[1], color[2],
            1.0, 0.0, color[0], color[1], color[2],
            1.0, 1.0, color[0], color[1], color[2],
            0.0, 1.0, color[0], color[1], color[2],
        ], dtype=np.float32)
        
        rect_vbo = self.ctx.buffer(rect_vertices.tobytes())
        vao = self.ctx.vertex_array(
            self.rect_shader,
            [(rect_vbo, '2f 3f', 'position', 'color')],
            self.quad_ibo
        )
        
        vao.render()
        vao.release()
        rect_vbo.release()
    
    def _render_button_text(self, button: ModernGLButton):
        """Render button text using texture"""
        try:
            font_mgr = get_font_manager()
            font = font_mgr.get_font(button.state.font_size)
            text_surface = font.render(button.state.text, True, button.state.color)
            
            if text_surface.get_width() == 0 or text_surface.get_height() == 0:
                return
            
            # Create texture from text surface
            texture = self._create_texture_from_surface(text_surface)
            
            # Center text on button
            text_x = button.state.x + (button.state.width - text_surface.get_width()) // 2
            text_y = button.state.y + (button.state.height - text_surface.get_height()) // 2
            
            # Render text texture
            self._render_texture(
                texture, text_x, text_y,
                text_surface.get_width(), text_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            texture.release()
            
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
            font_mgr = get_font_manager()
            font = font_mgr.get_font(24)
            subtitle_surface = font.render("A Minecraft-like Adventure", True, (200, 200, 200))
            
            if subtitle_surface.get_width() == 0:
                return
            
            texture = self._create_texture_from_surface(subtitle_surface)
            
            subtitle_x = self.width // 2 - subtitle_surface.get_width() // 2
            subtitle_y = 200
            
            self._render_texture(
                texture, subtitle_x, subtitle_y,
                subtitle_surface.get_width(), subtitle_surface.get_height(),
                np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            texture.release()
            
        except Exception as e:
            print(f"Subtitle rendering error: {e}")
    
    def _render_ui_info(self):
        """Render version info and instructions"""
        try:
            font_mgr = get_font_manager()
            
            # Version info
            version_font = font_mgr.get_font(16)
            version_surface = version_font.render(f"Version {VERSION}", True, (150, 150, 150))
            
            if version_surface.get_width() > 0:
                texture = self._create_texture_from_surface(version_surface)
                self._render_texture(
                    texture,
                    self.width - version_surface.get_width() - 10,
                    self.height - version_surface.get_height() - 10,
                    version_surface.get_width(), version_surface.get_height(),
                    np.array([1.0, 1.0, 1.0], dtype=np.float32)
                )
                texture.release()
            
            # Instructions
            instructions = [
                "Use mouse to click buttons",
                "Press Enter for New World",
                "Press ESC to exit"
            ]
            
            inst_font = font_mgr.get_font(16)
            y_offset = self.height - 100
            
            for instruction in instructions:
                inst_surface = inst_font.render(instruction, True, (120, 120, 120))
                if inst_surface.get_width() > 0:
                    texture = self._create_texture_from_surface(inst_surface)
                    inst_x = self.width // 2 - inst_surface.get_width() // 2
                    
                    self._render_texture(
                        texture, inst_x, y_offset,
                        inst_surface.get_width(), inst_surface.get_height(),
                        np.array([1.0, 1.0, 1.0], dtype=np.float32)
                    )
                    texture.release()
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
        vao = self.ctx.vertex_array(
            self.texture_shader,
            [(self.tex_quad_vbo, '2f 2f', 'position', 'texcoord')],
            self.quad_ibo
        )
        
        vao.render()
        vao.release()
    
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
            if hasattr(self, 'quad_vbo'):
                self.quad_vbo.release()
            if hasattr(self, 'quad_ibo'):
                self.quad_ibo.release()
            if hasattr(self, 'tex_quad_vbo'):
                self.tex_quad_vbo.release()
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


def show_main_menu(width: int = 1024, height: int = 768, screen: pygame.Surface = None) -> Optional[str]:
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


def show_pause_menu(width: int = 1024, height: int = 768, screen: pygame.Surface = None) -> Optional[str]:
    """Show the pause menu and return the selected option with automatic fallback"""
    try:
        print("🚀 Attempting ModernGL GPU-accelerated pause menu...")
        pause_menu = ModernGLPauseMenu(width, height, screen)
        return pause_menu.run()
    except ImportError as e:
        print(f"⚠️ ModernGL not available for pause menu: {e}")
        print("📱 Falling back to simple pause menu")