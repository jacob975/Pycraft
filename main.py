"""
Pycraft - A Minecraft-like Game in Python

Main entry point for the game.
"""

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
import pygame

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *


def _create_loading_screen(title: str, screen: pygame.Surface | None):
    from game.loading import LoadingScreen

    return LoadingScreen(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        title=title,
        total_steps=6,
        surface=screen,
    )


def _load_state_with_feedback(load_identifier: str, loader) -> dict | None:
    from game.saves import load_game

    dots = [".", "..", "..."]
    index = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_game, load_identifier)
        while not future.done():
            loader.set_status(f"Loading '{load_identifier}'{dots[index]}")
            index = (index + 1) % len(dots)
            time.sleep(0.08)

        return future.result()


def _build_bootstrap_with_feedback(load_state: dict | None, loader, title: str, use_gpu: bool = True) -> dict:
    from game.engine import build_game_bootstrap

    dots = [".", "..", "..."]
    index = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(build_game_bootstrap, load_state, use_gpu, None)
        while not future.done():
            loader.set_status(f"{title}{dots[index]}")
            index = (index + 1) % len(dots)
            time.sleep(0.08)

        return future.result()

def main():
    """Main function to start the game"""
    try:
        # Initialize pygame once at the beginning
        pygame.init()
        pygame.font.init()

        from game.menu import show_main_menu

        # Show main menu first
        print("Starting Pycraft...")
        selected_option = show_main_menu(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, screen=None)
        screen = pygame.display.get_surface()

        action = selected_option
        load_identifier = None

        if isinstance(selected_option, str) and selected_option.startswith('load_world:'):
            action = 'load_world'
            load_identifier = selected_option.split(':', 1)[1]

        if action == 'exit' or action is None:
            print("Exiting Pycraft. Thanks for playing!")
            return

        elif action == 'new_world':
            from game.engine import GameEngine

            print("Starting new world...")
            loader = _create_loading_screen("Pycraft - Creating World", screen)
            loader.set_status("Preparing world generator...")

            def report_progress(message: str) -> None:
                loader.advance(message)

            bootstrap_data = _build_bootstrap_with_feedback(
                load_state=None,
                loader=loader,
                title="Preparing world data",
                use_gpu=True,
            )
            loader.advance("World data prepared")

            screen = pygame.display.get_surface() or screen
            game = GameEngine(
                width=SCREEN_WIDTH,
                height=SCREEN_HEIGHT,
                use_gpu=True,
                screen=screen,
                bootstrap_data=bootstrap_data,
                progress_callback=report_progress,
            )
            loader.finish()
            game.run()

        elif action == 'load_world':
            from game.engine import GameEngine

            if not load_identifier:
                print("No save slot selected. Returning to menu...")
                return

            print(f"Loading world '{load_identifier}'...")
            loader = _create_loading_screen("Pycraft - Loading World", screen)
            loader.set_status(f"Loading '{load_identifier}'...")

            state = _load_state_with_feedback(load_identifier, loader)
            if not state:
                print("Failed to load save. Starting a new world instead...")
                loader.set_status("Save missing. Creating new world...")
                state = None

            def report_progress(message: str) -> None:
                loader.advance(message)

            bootstrap_data = _build_bootstrap_with_feedback(
                load_state=state,
                loader=loader,
                title="Preparing saved world",
                use_gpu=True,
            )
            loader.advance("Save data prepared")

            screen = pygame.display.get_surface() or screen
            game = GameEngine(
                width=SCREEN_WIDTH,
                height=SCREEN_HEIGHT,
                use_gpu=True,
                screen=screen,
                load_state=state,
                bootstrap_data=bootstrap_data,
                progress_callback=report_progress,
            )
            loader.finish()
            game.run()

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()