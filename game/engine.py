"""
Core game engine and main game loop for Pycraft
"""

import pygame
import sys
import time
from .world import World
from .player import Player
from .renderer import Renderer
from .blocks import BlockType

# Try to import GPU renderer
try:
    from .gpu_renderer import GPURenderer
    GPU_AVAILABLE = True
    print("GPU渲染器可用 - OpenGL硬體加速已啟用")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"GPU渲染器不可用: {e}")
    print("使用CPU渲染器")

class GameEngine:
    """Main game engine handling the game loop and coordination"""
    
    def __init__(self, width: int = 1024, height: int = 768, use_gpu: bool = True):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.running = True
        self.clock = pygame.time.Clock()
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Initialize game components
        self.world = World()
        
        # Choose renderer based on availability and preference
        if self.use_gpu:
            self.renderer = GPURenderer(width, height)
            print("使用GPU渲染器 - OpenGL硬體加速")
        else:
            self.renderer = Renderer(width, height)
            print("使用CPU渲染器 - 軟體渲染")
        # Find a good spawn position at ground level
        spawn_x, spawn_z = 8, 8
        ground_y = 30  # Start with a reasonable height
        
        # Try to find the actual ground level
        for y in range(60, 20, -1):
            block = self.world.get_block(spawn_x, y, spawn_z)
            if block.is_solid():
                ground_y = y + 2  # Spawn 2 blocks above solid ground
                break
        
        self.player = Player(self.world, spawn_position=(spawn_x, ground_y, spawn_z))
        # Set camera to look slightly down to see the ground
        self.player.camera.pitch = -0.4  # Look down about 23 degrees
        self.player.camera.yaw = 0.0     # Face forward
        
        print(f"玩家生成位置: ({spawn_x}, {ground_y}, {spawn_z})")
        print("方塊渲染已修復！您現在應該能看到方塊了。")
        
        # Don't enable mouse lock by default - let user press Tab to enable
        # self.player.toggle_mouse_lock()
        
        # Game state - optimized for performance
        self.fps_target = 120 if self.use_gpu else 60  # Higher FPS targets
        self.debug_mode = False
        # Always start with performance mode for better FPS
        self.performance_mode = True  # Always start in performance mode
        self.startup_time = 0.0  # Track startup time to ignore early ESC
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        # Ephemeral message overlay (e.g., for F3/F4 feedback)
        self._message_text = None
        self._message_expire = 0.0
        # Renderer preference flag
        self.renderer_preference = 'gpu' if self.use_gpu else 'cpu'
    
    def handle_events(self):
        """Handle all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and self.startup_time > 1.0:
                    # Only allow ESC to quit after 1 second to avoid startup issues
                    self.running = False
                elif event.key == pygame.K_F3:
                    self.debug_mode = not self.debug_mode
                    self.show_message(f"Debug: {'ON' if self.debug_mode else 'OFF'}")
                elif event.key == pygame.K_F4:
                    self.performance_mode = not self.performance_mode
                    mode = 'ON' if self.performance_mode else 'OFF'
                    print(f"Performance mode: {mode}")
                    self.show_message(f"Performance {mode}")
                else:
                    self.player.handle_key_press(event.key)
            
            elif event.type == pygame.MOUSEMOTION:
                self.player.handle_mouse_motion(event.rel[0], event.rel[1])
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.player.handle_mouse_click(event.button, event.pos)
    
    def update(self, dt: float):
        """Update game state"""
        self.startup_time += dt
        self.player.update(dt)
    
    def render(self):
        """Render the current frame"""
        self.renderer.render_world(self.world, self.player.camera, performance_mode=self.performance_mode)
        
        if self.debug_mode:
            self.draw_debug_info()

        # Draw ephemeral message if active
        self.draw_message_overlay()
        
        pygame.display.flip()
        
        # Update frame count for performance tracking
        self.frame_count += 1
    
    def draw_debug_info(self):
        """Draw debug information"""
        # Check if renderer supports debug info drawing
        if hasattr(self.renderer, 'draw_debug_info'):
            # Let the renderer handle debug info (for GPU renderer)
            pos = self.player.camera.position
            chunk_x, chunk_z = self.world.get_chunk_coords(int(pos[0]), int(pos[2]))
            chunks_loaded = len(self.world.chunks)
            fps = self.clock.get_fps()
            block_name = self.player.selected_block.name
            
            debug_data = {
                'fps': fps,
                'position': pos,
                'chunk': (chunk_x, chunk_z),
                'chunks_loaded': chunks_loaded,
                'selected_block': block_name,
                'performance_mode': self.performance_mode
            }
            
            self.renderer.draw_debug_info(debug_data)
            
        elif hasattr(self.renderer, 'screen'):
            # Legacy CPU renderer with screen attribute
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            
            # FPS
            fps = self.clock.get_fps()
            fps_text = f"FPS: {fps:.1f}"
            text_surface = font_mgr.render_text(fps_text, size=24, color=(255, 255, 0))
            self.renderer.screen.blit(text_surface, (self.width - 100, 10))
            
            # Chunk info
            pos = self.player.camera.position
            chunk_x, chunk_z = self.world.get_chunk_coords(int(pos[0]), int(pos[2]))
            chunk_text = f"區塊: ({chunk_x}, {chunk_z})"
            text_surface = font_mgr.render_text(chunk_text, size=24, color=(255, 255, 0))
            self.renderer.screen.blit(text_surface, (self.width - 150, 35))
            
            # Performance info
            chunks_loaded = len(self.world.chunks)
            perf_text = f"已載入: {chunks_loaded}"
            text_surface = font_mgr.render_text(perf_text, size=24, color=(255, 255, 0))
            self.renderer.screen.blit(text_surface, (self.width - 120, 60))
            
            # Selected block
            block_name = self.player.selected_block.name
            block_text = f"方塊: {block_name}"
            text_surface = font_mgr.render_text(block_text, size=24, color=(255, 255, 0))
            self.renderer.screen.blit(text_surface, (self.width - 120, 85))

    # --------------------------------------------------------------
    # Ephemeral message overlay helpers
    # --------------------------------------------------------------
    def show_message(self, text: str, duration: float = 2.0):
        self._message_text = text
        self._message_expire = time.time() + duration

    def draw_message_overlay(self):
        if not self._message_text:
            return
        if time.time() > self._message_expire:
            self._message_text = None
            return
        try:
            from .font_manager import get_font_manager
            font_mgr = get_font_manager()
            surf = font_mgr.render_text(self._message_text, size=30, color=(255, 255, 255))
            # Center top
            screen = pygame.display.get_surface()
            if screen:
                rect = surf.get_rect()
                rect.centerx = self.width // 2
                rect.top = 10
                # Background box
                bg_rect = rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0, 160), bg_rect)
                screen.blit(surf, rect)
        except Exception:
            pass
    
    def run(self):
        """Main game loop"""
        print("啟動 Pycraft...")
        print("控制說明:")
        print("  WASD - 移動")
        print("  滑鼠 - 轉視角")
        print("  空格鍵/Shift - 上升/下降")
        print("  左鍵 - 破壞方塊")
        print("  右鍵 - 放置方塊")
        print("  1-4 - 選擇方塊類型")
        print("  Tab - 切換滑鼠捕獲")
        print("  F3 - 切換調試信息")
        print("  F4 - 切換性能模式")
        print("  ESC - 退出遊戲")
        print("\n注意: 按Tab鍵啟用滑鼠控制!")
        
        last_time = time.time()
        
        while self.running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            self.handle_events()
            
            # Update game state
            self.update(dt)
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(self.fps_target)
            # Reduce console output frequency for better performance
            if self.frame_count % 60 == 0:  # Every 1 second instead of every frame
                print(f"FPS: {self.clock.get_fps():.1f} | Frames: {self.frame_count}", end='\r')
        
        # Cleanup
        pygame.quit()
        print("遊戲結束。感謝遊玩 Pycraft!")