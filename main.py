"""
Pycraft - A Minecraft-like Game in Python

Main entry point for the game.
"""

import sys
import os
import pygame

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.engine import GameEngine
from game.menu import show_main_menu
from game.saves import load_game
from config import *

def main():
    """Main function to start the game"""
    try:
        # Initialize pygame once at the beginning
        pygame.init()
        pygame.font.init()
        
        # Show main menu first
        print("Starting Pycraft...")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        selected_option = show_main_menu(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, screen=screen)

        action = selected_option
        load_identifier = None

        if isinstance(selected_option, str) and selected_option.startswith('load_world:'):
            action = 'load_world'
            load_identifier = selected_option.split(':', 1)[1]

        if action == 'exit' or action is None:
            print("Exiting Pycraft. Thanks for playing!")
            return
        
        elif action == 'new_world':
            print("Starting new world...")
            # Reuse the same pygame context and screen
            game = GameEngine(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, use_gpu=True, screen=screen)
            game.run()
            
        elif action == 'load_world':
            if not load_identifier:
                print("No save slot selected. Returning to menu...")
                return

            print(f"Loading world '{load_identifier}'...")
            state = load_game(load_identifier)
            if not state:
                print("Failed to load save. Starting a new world instead...")
                game = GameEngine(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, use_gpu=True, screen=screen)
            else:
                game = GameEngine(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, use_gpu=True, screen=screen, load_state=state)
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