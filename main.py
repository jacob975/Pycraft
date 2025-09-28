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

def main():
    """Main function to start the game"""
    try:
        # Initialize pygame once at the beginning
        pygame.init()
        pygame.font.init()
        
        # Show main menu first
        print("Starting Pycraft...")
        screen = pygame.display.set_mode((1024, 768))
        selected_option = show_main_menu(width=1024, height=768, screen=screen)
        
        if selected_option == 'exit' or selected_option is None:
            print("Exiting Pycraft. Thanks for playing!")
            return
        
        elif selected_option == 'new_world':
            print("Starting new world...")
            # Reuse the same pygame context and screen
            game = GameEngine(width=1024, height=768, use_gpu=True, screen=screen)
            game.run()
            
        elif selected_option == 'load_world':
            print("Load world feature not implemented yet.")
            print("Starting new world instead...")
            # Reuse the same pygame context and screen
            game = GameEngine(width=1024, height=768, use_gpu=True, screen=screen)
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