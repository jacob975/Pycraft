"""
ModernGL-based menu system for Pycraft.
Provides high-performance GPU-accelerated menu rendering with OpenGL shaders.
"""

from __future__ import annotations
import pygame
import sys
import time
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
        # Always create a new OpenGL-enabled window for ModernGL
        # ModernGL requires a proper OpenGL context which might not exist with existing screens
        
        # Set OpenGL attributes before creating the display
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        
        # Try to enable MSAA, but don't fail if not supported
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        except pygame.error:
            print("⚠️ MSAA not supported, continuing without anti-aliasing")
        
        # Create OpenGL-enabled window
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("Pycraft - ModernGL Menu")
        
        # Ensure OpenGL context is active
        pygame.display.gl_set_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT, 1)
        
        # Create ModernGL context
        try:
            self.ctx = mgl.create_context()
            print("✅ ModernGL context created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create ModernGL context: {e}")
        
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

def show_main_menu(width: int = 1024, height: int = 768, screen: pygame.Surface = None) -> Optional[str]:
    """Show the main menu and return the selected option with automatic fallback"""
    try:
        print("🚀 Attempting ModernGL GPU-accelerated menu...")
        menu = ModernGLMenu(width, height, screen)
        return menu.run()
    except ImportError as e:
        print(f"⚠️ ModernGL not available: {e}")
        print("📱 Falling back to standard pygame menu")
    except RuntimeError as e:
        if "OpenGL" in str(e):
            print(f"⚠️ OpenGL context error: {e}")
            print("📱 Falling back to standard pygame menu")
        else:
            raise e
    except Exception as e:
        print(f"⚠️ ModernGL menu failed: {e}")
        print("📱 Falling back to standard pygame menu")


def show_pause_menu(width: int = 1024, height: int = 768, screen: pygame.Surface = None) -> Optional[str]:
    """Show the pause menu and return the selected option with automatic fallback"""
    try:
        print("🚀 Attempting ModernGL GPU-accelerated pause menu...")
        pause_menu = ModernGLPauseMenu(width, height, screen)
        return pause_menu.run()
    except ImportError as e:
        print(f"⚠️ ModernGL not available for pause menu: {e}")
        print("📱 Falling back to simple pause menu")
        return _show_simple_pause_menu(width, height, screen)
    except RuntimeError as e:
        if "OpenGL" in str(e):
            print(f"⚠️ OpenGL context error for pause menu: {e}")
            print("📱 Falling back to simple pause menu")
            return _show_simple_pause_menu(width, height, screen)
        else:
            raise e
    except Exception as e:
        print(f"⚠️ ModernGL pause menu failed: {e}")
        print("📱 Falling back to simple pause menu")
        return _show_simple_pause_menu(width, height, screen)


def _show_simple_pause_menu(width: int, height: int, screen: pygame.Surface) -> Optional[str]:
    """Simple fallback pause menu using basic pygame rendering"""
    if not screen:
        screen = pygame.display.set_mode((width, height))
    
    pygame.font.init()
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 32)
    
    # Semi-transparent overlay surface
    overlay = pygame.Surface((width, height))
    overlay.set_alpha(200)  # Semi-transparent
    overlay.fill((20, 20, 20))
    
    clock = pygame.time.Clock()
    running = True
    selected_option = None
    
    # Button dimensions
    button_width = 250
    button_height = 50
    button_spacing = 15
    start_y = height // 2 - 60
    
    buttons = [
        {"text": "Resume Game", "action": "resume", "color": (40, 120, 40)},
        {"text": "Settings", "action": "settings", "color": (80, 80, 120)},
        {"text": "Save & Quit", "action": "save_quit", "color": (120, 80, 40)},
        {"text": "Exit to Main Menu", "action": "main_menu", "color": (120, 40, 40)}
    ]
    
    # Create button rectangles
    button_rects = []
    for i, button in enumerate(buttons):
        rect = pygame.Rect(
            width // 2 - button_width // 2,
            start_y + i * (button_height + button_spacing),
            button_width,
            button_height
        )
        button_rects.append(rect)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                selected_option = 'main_menu'
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    selected_option = 'resume'
                    running = False
                elif event.key == pygame.K_RETURN:
                    selected_option = 'resume'
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = event.pos
                    for i, rect in enumerate(button_rects):
                        if rect.collidepoint(mouse_pos):
                            selected_option = buttons[i]["action"]
                            running = False
        
        # Render
        screen.blit(overlay, (0, 0))
        
        # Title
        title_text = font_large.render("GAME PAUSED", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(width // 2, height // 4))
        screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = font_medium.render("Press ESC or click Resume to continue", True, (180, 180, 180))
        subtitle_rect = subtitle_text.get_rect(center=(width // 2, height // 4 + 50))
        screen.blit(subtitle_text, subtitle_rect)
        
        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        for i, (button, rect) in enumerate(zip(buttons, button_rects)):
            # Button background
            color = button["color"]
            if rect.collidepoint(mouse_pos):
                color = tuple(min(255, c + 40) for c in color)  # Hover effect
            
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)  # Border
            
            # Button text
            button_text = font_medium.render(button["text"], True, (255, 255, 255))
            text_rect = button_text.get_rect(center=rect.center)
            screen.blit(button_text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    return selected_option