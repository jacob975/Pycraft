"""
Core game engine and main game loop for Pycraft
"""

import pygame
import sys
import time
from typing import Any, Callable, Dict, Optional
import threading
import logging
from .world import World
from .player import Player
from .blocks import BlockType
from .menu import show_pause_menu
from .saves import apply_player_state, apply_world_state, save_game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import *

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
    
    def __init__(self, width: int = 1024, height: int = 768, use_gpu: bool = True,
                 screen: pygame.Surface = None, load_state: Optional[Dict[str, Any]] = None,
                 progress_callback: Optional[Callable[[str], None]] = None):
        # Initialize Pygame if not already done
        if not pygame.get_init():
            pygame.init()
            pygame.font.init()
        
        self.width = width
        self.height = height
        self.running = True
        self.pause = False
        self.clock = pygame.time.Clock()
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Store screen reference for potential reuse
        self.external_screen = screen
        self._load_state: Optional[Dict[str, Any]] = load_state
        self.loaded_metadata: Optional[Dict[str, Any]] = load_state.get("metadata") if load_state else None
        self._progress_callback = progress_callback

        # Initialize game components
        world_seed = None
        if load_state and isinstance(load_state.get("world"), dict):
            world_seed = load_state["world"].get("seed")
        self.world = World(seed=world_seed, use_multiprocessing=True)
        self._report_progress("World generator ready")

        world_state = load_state.get("world") if load_state else None
        if world_state:
            apply_world_state(self.world, world_state)
            world_message = "Saved terrain restored"
        else:
            world_message = "Preparing initial terrain"
        self._report_progress(world_message)

        self._chunk_reload_distance = max(0, RELOAD_DISTANCE)
        player_state = load_state.get("player") if load_state else None

        spawn_x, spawn_z = 8, 8
        ground_y = 30
        position = player_state.get("position") if player_state else None
        if isinstance(position, (list, tuple)) and len(position) == 3:
            spawn_position = (float(position[0]), float(position[1]), float(position[2]))
        else:
            for y in range(60, 20, -1):
                block = self.world.get_block(spawn_x, y, spawn_z)
                if block.is_solid():
                    ground_y = y + 2  # Spawn 2 blocks above solid ground
                    break
            spawn_position = (spawn_x, ground_y, spawn_z)
            print(f"玩家生成位置: ({spawn_x}, {ground_y}, {spawn_z})")
        self._report_progress("Spawn point locked")

        self.player = Player(self.world, spawn_position=spawn_position)
        self._report_progress("Player initialized")

        if player_state:
            apply_player_state(self.player, player_state)
            player_message = "Player state restored"
        else:
            # Set camera to look slightly down to see the ground
            self.player.camera.pitch = -0.4  # Look down about 23 degrees
            self.player.camera.yaw = 0.0     # Face forward
            player_message = "Calibrated player view"
        self._report_progress(player_message)

        if self.loaded_metadata:
            save_name = self.loaded_metadata.get("name") or self.loaded_metadata.get("id")
            print(f"載入存檔: {save_name}")
        elif load_state:
            print("載入存檔: 未命名存檔")

        self._chunk_reload_thread = threading.Thread(
            target=self._preload_chunks_around_player, 
            args=(self._chunk_reload_distance,)
        )
        self._chunk_reload_thread.start()

        self.renderer_preference = 'gpu' if self.use_gpu else 'cpu'
        if self.loaded_metadata and self.loaded_metadata.get("renderer"):
            self.renderer_preference = self.loaded_metadata.get("renderer")
        
        # Don't enable mouse lock by default - let user press Tab to enable
        # self.player.toggle_mouse_lock()
        
        # Game state - optimized for performance
        self.fps_target = FPS * 2 if self.use_gpu else FPS  # Higher FPS targets
        self.debug_mode = False
        # Always start with performance mode for better FPS
        self.performance_mode = True  # Always start in performance mode
        self.startup_time = 0.0  # Track startup time to ignore early ESC

        if load_state:
            engine_state = load_state.get("engine") or {}
            self.debug_mode = bool(engine_state.get("debug_mode", self.debug_mode))
            self.performance_mode = bool(engine_state.get("performance_mode", self.performance_mode))
            if "fps_target" in engine_state:
                try:
                    self.fps_target = int(engine_state["fps_target"])
                except (TypeError, ValueError):
                    pass
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        # Ephemeral message overlay (e.g., for F3/F4 feedback)
        self._message_text = None
        self._message_expire = 0.0
        
        self._report_progress("Configuring renderer")
        # Choose renderer based on availability and preference
        if self.use_gpu:
            self.renderer = GPURenderer(width, height, self.external_screen)
            print("使用GPU渲染器 - OpenGL硬體加速")
        else:
            raise NotImplementedError("CPU渲染器尚未實作")

        # Loading UI is no longer needed once initialization completes
        self._progress_callback = None
    
    def _report_progress(self, message: str) -> None:
        if not self._progress_callback:
            return
        try:
            self._progress_callback(message)
        except Exception:
            # Loading UI is non-critical; ignore reporting failures
            pass

    def handle_events(self):
        """Handle all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Kill the thread anyway
                self._chunk_reload_thread.join(timeout=1.0)
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and self.startup_time > 1.0:
                    # Pause the game and show settings menu
                    self.pause = not self.pause
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
        #self.draw_message_overlay()
        
        pygame.display.flip()
        
        # Update frame count for performance tracking
        self.frame_count += 1

    def _preload_chunks_around_player(self, reload_distance: int = 2) -> None:
        """Ensure the player's current and surrounding chunks stay loaded."""
        while True:
            st_time = time.time()

            # End condition
            if self.running is False:
                return

            pos = self.player.camera.position
            #logging.info(f"Player position: {pos}")
            chunk_x, chunk_z = self.world.get_chunk_coords(int(pos[0]), int(pos[2]))

            for dx in range(-reload_distance, reload_distance + 1):
                for dz in range(-reload_distance, reload_distance + 1):
                    cx, cz = chunk_x + dx, chunk_z + dz
                    if (cx, cz) not in self.world.chunks:
                        #logger.info(f"Loading chunk ({cx}, {cz}) around player")
                        self.world.get_or_create_chunk(cx, cz)
            elapsed = time.time() - st_time
            time.sleep(1 - min(elapsed, 1.0))  # Ensure at least 1 second interval
    
    def draw_debug_info(self):
        """Draw debug information"""
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
            'performance_mode': self.performance_mode,
        }
        
        self.renderer.draw_debug_info(debug_data)

    # --------------------------------------------------------------
    # Ephemeral message overlay helpers
    # --------------------------------------------------------------
    def show_message(self, text: str, duration: float = 2.0):
        self._message_text = text
        self._message_expire = time.time() + duration

    def show_pause_menu(self):
        """Show pause menu and handle user input"""
        # Unlock mouse when showing menu
        mouse_lock_state = False
        if self.player.mouse_locked:
            mouse_lock_state = True
            self.player.toggle_mouse_lock()
        selected_option = show_pause_menu(width=self.width, height=self.height, screen=self.renderer.screen)
        # Restore mouse lock state
        if mouse_lock_state:
            self.player.toggle_mouse_lock()
        
        if selected_option == 'resume' or selected_option is None:
            self.pause = False
            print("繼續遊戲...")
        elif selected_option == 'save_quit':
            display_name = None
            overwrite = False
            if self.loaded_metadata:
                display_name = self.loaded_metadata.get("name")
                overwrite = True
            metadata = save_game(self, save_name=display_name, overwrite=overwrite)
            self.loaded_metadata = {
                "id": metadata.identifier,
                "name": metadata.display_name,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "renderer": self.renderer_preference,
            }
            print(f"存檔完成: {metadata.display_name} ({metadata.identifier})")
            self.running = False
            self.pause = False
        elif selected_option == 'exit' or selected_option == 'main_menu':
            print("退出遊戲...")
            self.running = False
            self.pause = False  # Ensure we exit the pause state
    
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
        print("  ESC - 暫停/返回選單")
        print("\n注意: 按Tab鍵啟用滑鼠控制!")
        
        last_time = time.time()
        
        while self.running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            self.handle_events()
            
            if self.pause:
                # Show pause menu
                self.show_pause_menu()
            else:
                # Update game state
                self.update(dt)
                # Render
                self.render()
            
            # Control frame rate
            self.clock.tick(self.fps_target)
        
        # Cleanup
        pygame.quit()
        print("遊戲結束。感謝遊玩 Pycraft!")