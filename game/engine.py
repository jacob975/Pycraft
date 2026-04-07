"""
Core game engine and main game loop for Pycraft
"""

import pygame
import time
import numpy as np
from typing import Any, Callable, Dict, Optional
import threading
import logging
from .world import World
from .player import Player
from .camera import Camera
from .blocks import BlockType
from .menu import show_pause_menu
from .saves import apply_player_state, apply_world_state, save_game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import *

# GPU renderer is required for this project.
try:
    from .gpu_renderer import GPURenderer
except ImportError as e:
    raise RuntimeError("GPU renderer is required, but it could not be imported.") from e


def build_game_bootstrap(
    load_state: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Build heavy world/player startup data that is safe to prepare off the main render path."""

    def report(message: str) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(message)
        except Exception:
            pass

    world_seed = None
    if load_state and isinstance(load_state.get("world"), dict):
        world_seed = load_state["world"].get("seed")

    world = World(seed=world_seed, use_multiprocessing=True)
    report("World generator ready")

    world_state = load_state.get("world") if load_state else None
    if world_state:
        apply_world_state(world, world_state)
        world_message = "Saved terrain restored"
    else:
        world_message = "Preparing initial terrain"
    report(world_message)

    player_state = load_state.get("player") if load_state else None

    spawn_x, spawn_z = 8, 8
    ground_y = 30
    position = player_state.get("position") if player_state else None
    if isinstance(position, (list, tuple)) and len(position) == 3:
        spawn_position = (float(position[0]), float(position[1]), float(position[2]))
    else:
        for y in range(60, 20, -1):
            block = world.get_block(spawn_x, y, spawn_z)
            if block.is_solid():
                ground_y = y + 2  # Spawn 2 blocks above solid ground
                break
        spawn_position = (spawn_x, ground_y, spawn_z)
        print(f"玩家生成位置: ({spawn_x}, {ground_y}, {spawn_z})")
    report("Spawn point locked")

    player = Player(world, spawn_position=spawn_position)
    report("Player initialized")

    if player_state:
        apply_player_state(player, player_state)
        player_message = "Player state restored"
    else:
        # Set camera to look slightly down to see the ground
        player.camera.pitch = -0.4  # Look down about 23 degrees
        player.camera.yaw = 0.0     # Face forward
        player_message = "Calibrated player view"
    report(player_message)

    loaded_metadata: Optional[Dict[str, Any]] = load_state.get("metadata") if load_state else None
    if loaded_metadata:
        save_name = loaded_metadata.get("name") or loaded_metadata.get("id")
        print(f"載入存檔: {save_name}")
    elif load_state:
        print("載入存檔: 未命名存檔")

    return {
        "world": world,
        "player": player,
        "loaded_metadata": loaded_metadata,
    }

class GameEngine:
    """Main game engine handling the game loop and coordination"""
    
    def __init__(self, width: int = 1024, height: int = 768,
                 screen: Optional[pygame.Surface] = None, load_state: Optional[Dict[str, Any]] = None,
                 progress_callback: Optional[Callable[[str], None]] = None,
                 bootstrap_data: Optional[Dict[str, Any]] = None):
        # Initialize Pygame if not already done
        if not pygame.get_init():
            pygame.init()
            pygame.font.init()
        
        self.width = width
        self.height = height
        self.running = True
        self.pause = False
        self.clock = pygame.time.Clock()
        
        # Store screen reference for potential reuse
        self.external_screen = screen
        self._load_state: Optional[Dict[str, Any]] = load_state
        self._progress_callback = progress_callback

        self._chunk_reload_distance = max(0, RELOAD_DISTANCE)
        if bootstrap_data is None:
            bootstrap_data = build_game_bootstrap(
                load_state=load_state,
                progress_callback=self._report_progress,
            )

        self.world = bootstrap_data["world"]
        self.player = bootstrap_data["player"]
        self.loaded_metadata = bootstrap_data.get("loaded_metadata")

        self._chunk_reload_thread = threading.Thread(
            target=self._preload_chunks_around_player, 
            args=(self._chunk_reload_distance,)
        )
        self._chunk_reload_thread.start()
        
        # Enable mouse lock by default so camera look works immediately
        self.player.toggle_mouse_lock()
        
        # Game state - optimized for performance
        self.fps_target = FPS * 2  # Higher FPS target for GPU rendering
        self.debug_mode = False
        # Always start with performance mode for better FPS
        self.performance_mode = True  # Always start in performance mode
        self.third_person_mode = False
        self.selfie_mode = False
        self.third_person_distance = 4.0
        self.third_person_height = 1.4
        self.startup_time = 0.0
        self.render_distance_chunks = int(RENDER_DISTANCE)
        self.fog_distance = float(FOG_DISTANCE)

        if load_state:
            engine_state = load_state.get("engine") or {}
            self.debug_mode = bool(engine_state.get("debug_mode", self.debug_mode))
            self.performance_mode = bool(engine_state.get("performance_mode", self.performance_mode))
            if "fps_target" in engine_state:
                try:
                    self.fps_target = int(engine_state["fps_target"])
                except (TypeError, ValueError):
                    pass
            if "render_distance" in engine_state:
                try:
                    parsed_distance = int(engine_state["render_distance"])
                    self.render_distance_chunks = max(MIN_RENDER_DISTANCE, min(MAX_RENDER_DISTANCE, parsed_distance))
                except (TypeError, ValueError):
                    pass
            if "fog_distance" in engine_state:
                try:
                    parsed_fog_distance = float(engine_state["fog_distance"])
                    self.fog_distance = max(float(MIN_FOG_DISTANCE), min(float(MAX_FOG_DISTANCE), parsed_fog_distance))
                except (TypeError, ValueError):
                    pass
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        # Ephemeral message overlay (e.g., for F3/F4 feedback)
        self._message_text = None
        self._message_expire = 0.0
        
        self._report_progress("Configuring renderer")
        self.renderer = GPURenderer(width, height, self.external_screen)
        self.renderer.set_render_distance(self.render_distance_chunks)
        self.renderer.set_fog_distance(self.fog_distance)
        print("使用GPU渲染器 - OpenGL硬體加速")

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
                if event.key == pygame.K_ESCAPE:
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
                elif event.key == pygame.K_F6:
                    self.third_person_mode = not self.third_person_mode
                    view_mode = 'Third-person' if self.third_person_mode else 'First-person'
                    print(f"視角模式: {view_mode}")
                    self.show_message(view_mode)
                elif event.key == pygame.K_r:
                    self.selfie_mode = True
                else:
                    self.player.handle_key_press(event.key)

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_r:
                    self.selfie_mode = False
            
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
        render_camera = self._get_render_camera()
        self.renderer.render_world(
            self.world,
            render_camera,
            performance_mode=self.performance_mode,
            player_state=self._get_player_render_state(),
        )
        
        if self.debug_mode:
            self.draw_debug_info()

        # Draw ephemeral message if active
        #self.draw_message_overlay()
        
        pygame.display.flip()
        
        # Update frame count for performance tracking
        self.frame_count += 1

    def _get_player_render_state(self) -> Dict[str, Any]:
        """Provide player transform data for optional model rendering."""
        horizontal_speed = float(np.linalg.norm(self.player.horizontal_velocity))
        move_factor = 0.0
        if self.player.speed > 1e-6:
            move_factor = min(1.0, horizontal_speed / float(self.player.speed))

        return {
            "position": self.player.camera.position.copy(),
            "yaw": float(self.player.camera.yaw),
            "visible": bool(self.third_person_mode or self.selfie_mode),
            "walking": bool((not self.player.flying) and self.player.on_ground and horizontal_speed > 0.05),
            "move_factor": float(move_factor),
        }

    def _get_render_camera(self) -> Camera:
        """Return the camera used for world rendering (first- or third-person)."""
        if self.selfie_mode:
            forward = self.player.camera.get_horizontal_forward_vector()
            render_pos = (
                self.player.camera.position
                + forward * self.third_person_distance
                + np.array([0.0, self.third_person_height, 0.0], dtype=float)
            )
            render_camera = Camera(tuple(render_pos.tolist()))
            render_camera.yaw = self.player.camera.yaw + np.pi
            render_camera.pitch = -0.2
            return render_camera

        if not self.third_person_mode:
            return self.player.camera

        forward = self.player.camera.get_horizontal_forward_vector()
        render_pos = (
            self.player.camera.position
            - forward * self.third_person_distance
            + np.array([0.0, self.third_person_height, 0.0], dtype=float)
        )

        render_camera = Camera(tuple(render_pos.tolist()))
        render_camera.yaw = self.player.camera.yaw
        render_camera.pitch = self.player.camera.pitch
        return render_camera

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
            'render_distance': self.render_distance_chunks,
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
        menu_result = show_pause_menu(
            width=self.width,
            height=self.height,
            screen=self.renderer.screen,
            current_render_distance=self.render_distance_chunks,
            current_fog_distance=self.fog_distance,
            min_render_distance=MIN_RENDER_DISTANCE,
            max_render_distance=MAX_RENDER_DISTANCE,
            min_fog_distance=MIN_FOG_DISTANCE,
            max_fog_distance=MAX_FOG_DISTANCE,
        )
        selected_option = None
        if isinstance(menu_result, dict):
            selected_option = menu_result.get('action')
            requested_render_distance = menu_result.get('render_distance')
            if isinstance(requested_render_distance, (int, float)):
                self.set_render_distance(int(requested_render_distance))
            requested_fog_distance = menu_result.get('fog_distance')
            if isinstance(requested_fog_distance, (int, float)):
                self.set_fog_distance(float(requested_fog_distance))
        else:
            selected_option = menu_result
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
            }
            print(f"存檔完成: {metadata.display_name} ({metadata.identifier})")
            self.running = False
            self.pause = False
        elif selected_option == 'exit' or selected_option == 'main_menu':
            print("退出遊戲...")
            self.running = False
            self.pause = False  # Ensure we exit the pause state

    def set_render_distance(self, render_distance: int) -> None:
        """Set the user-selected render distance and apply it to the renderer."""
        clamped_distance = max(MIN_RENDER_DISTANCE, min(MAX_RENDER_DISTANCE, int(render_distance)))
        if clamped_distance == self.render_distance_chunks:
            return
        self.render_distance_chunks = clamped_distance
        self.renderer.set_render_distance(clamped_distance)
        self.show_message(f"Render distance: {clamped_distance}")

    def set_fog_distance(self, fog_distance: float) -> None:
        """Set the user-selected fog distance and apply it to the renderer."""
        clamped_distance = max(float(MIN_FOG_DISTANCE), min(float(MAX_FOG_DISTANCE), float(fog_distance)))
        if abs(clamped_distance - self.fog_distance) < 0.001:
            return
        self.fog_distance = clamped_distance
        self.renderer.set_fog_distance(clamped_distance)
        self.show_message(f"Fog distance: {clamped_distance:.1f}")
    
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
        print("\n注意: 視角控制已啟用，Tab 可切換滑鼠捕獲。")
        
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