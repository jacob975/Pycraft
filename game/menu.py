"""
Minecraft-like main menu interface for Pycraft
"""

import pygame
import sys
import time
from typing import Optional, Tuple
from .font_manager import get_font_manager

class Button:
    """Simple button class for the menu"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255),
                 bg_color: Tuple[int, int, int] = (50, 50, 50),
                 hover_color: Tuple[int, int, int] = (80, 80, 80)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font_size = font_size
        self.color = color
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.is_hovered = False
        self.is_pressed = False
        
    def handle_event(self, event) -> bool:
        """Handle mouse events and return True if clicked"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.is_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_pressed and self.rect.collidepoint(event.pos):
                self.is_pressed = False
                return True
            self.is_pressed = False
        return False
    
    def draw(self, screen: pygame.Surface):
        """Draw the button"""
        # Choose color based on state
        current_bg_color = self.hover_color if self.is_hovered else self.bg_color
        if self.is_pressed:
            current_bg_color = tuple(max(0, c - 20) for c in current_bg_color)
        
        # Draw button background
        pygame.draw.rect(screen, current_bg_color, self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)  # Border
        
        # Draw text
        font_mgr = get_font_manager()
        text_surface = font_mgr.render_text(self.text, self.font_size, self.color)
        
        # Center text on button
        text_rect = text_surface.get_rect()
        text_rect.center = self.rect.center
        screen.blit(text_surface, text_rect)

class MainMenu:
    """Main menu interface for Pycraft"""
    
    def __init__(self, width: int = 1024, height: int = 768, screen: pygame.Surface = None):
        self.width = width
        self.height = height
        
        # Use provided screen or create new one
        if screen is None:
            # Initialize Pygame if not already done
            if not pygame.get_init():
                pygame.init()
                pygame.font.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Pycraft - Main Menu")
            self.owns_screen = True
        else:
            self.screen = screen
            self.owns_screen = False
            pygame.display.set_caption("Pycraft - Main Menu")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.selected_option = None
        
        # Background color (dark blue/gray like Minecraft)
        self.bg_color = (30, 30, 50)
        
        # Title animation
        self.title_scale = 1.0
        self.title_time = 0.0
        
        # Create buttons
        button_width = 300
        button_height = 50
        button_spacing = 20
        start_y = height // 2 - 30
        
        self.buttons = {
            'new_world': Button(
                width // 2 - button_width // 2,
                start_y,
                button_width,
                button_height,
                "Start New World",
                font_size=28,
                bg_color=(40, 120, 40),
                hover_color=(60, 140, 60)
            ),
            'load_world': Button(
                width // 2 - button_width // 2,
                start_y + button_height + button_spacing,
                button_width,
                button_height,
                "Load World",
                font_size=28,
                bg_color=(40, 80, 120),
                hover_color=(60, 100, 140)
            ),
            'exit': Button(
                width // 2 - button_width // 2,
                start_y + 2 * (button_height + button_spacing),
                button_width,
                button_height,
                "Exit Game",
                font_size=28,
                bg_color=(120, 40, 40),
                hover_color=(140, 60, 60)
            )
        }
        
        print("Pycraft Main Menu initialized")
        print("Choose an option to continue...")
    
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
                    # Default action is start new world
                    self.selected_option = 'new_world'
                    self.running = False
            
            # Handle button events
            for button_name, button in self.buttons.items():
                if button.handle_event(event):
                    self.selected_option = button_name
                    self.running = False
    
    def update(self, dt: float):
        """Update menu state"""
        # Animate title
        self.title_time += dt
        self.title_scale = 1.0 + 0.05 * abs(pygame.math.Vector2(0, 1).rotate(self.title_time * 50).y)
    
    def draw(self):
        """Draw the menu"""
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Draw background pattern (simple)
        for i in range(0, self.width, 50):
            for j in range(0, self.height, 50):
                if (i + j) % 100 == 0:
                    pygame.draw.rect(self.screen, (35, 35, 55), (i, j, 50, 50))
        
        # Draw title
        font_mgr = get_font_manager()
        title_size = int(64 * self.title_scale)
        title_surface = font_mgr.render_text("PYCRAFT", title_size, (255, 255, 255), bold=True)
        
        title_rect = title_surface.get_rect()
        title_rect.centerx = self.width // 2
        title_rect.y = 100
        
        # Draw title shadow
        shadow_surface = font_mgr.render_text("PYCRAFT", title_size, (50, 50, 50), bold=True)
        shadow_rect = title_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        self.screen.blit(shadow_surface, shadow_rect)
        
        # Draw title
        self.screen.blit(title_surface, title_rect)
        
        # Draw subtitle
        subtitle_surface = font_mgr.render_text("A Minecraft-like Adventure", 24, (200, 200, 200))
        subtitle_rect = subtitle_surface.get_rect()
        subtitle_rect.centerx = self.width // 2
        subtitle_rect.y = title_rect.bottom + 10
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)
        
        # Draw version info
        version_surface = font_mgr.render_text("Version 1.0.0", 16, (150, 150, 150))
        version_rect = version_surface.get_rect()
        version_rect.bottomright = (self.width - 10, self.height - 10)
        self.screen.blit(version_surface, version_rect)
        
        # Draw instructions
        instructions = [
            "Use mouse to click buttons",
            "Press Enter for New World",
            "Press ESC to exit"
        ]
        
        y_offset = self.height - 100
        for instruction in instructions:
            inst_surface = font_mgr.render_text(instruction, 16, (120, 120, 120))
            inst_rect = inst_surface.get_rect()
            inst_rect.centerx = self.width // 2
            inst_rect.y = y_offset
            self.screen.blit(inst_surface, inst_rect)
            y_offset += 20
        
        pygame.display.flip()
    
    def run(self) -> Optional[str]:
        """Run the menu and return the selected option"""
        print("Pycraft Main Menu started")
        print("Options: New World, Load World, Exit")
        
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
            
            # Draw
            self.draw()
            
            # Control frame rate
            self.clock.tick(60)
        
        return self.selected_option

def show_main_menu(width: int = 1024, height: int = 768, screen: pygame.Surface = None) -> Optional[str]:
    """Show the main menu and return the selected option"""
    menu = MainMenu(width, height, screen)
    return menu.run()