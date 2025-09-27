"""
Pycraft - A Minecraft-like Game in Python

Main entry point for the game.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.engine import GameEngine

def main():
    """Main function to start the game"""
    try:
        # Create and run the game with GPU acceleration enabled by default
        game = GameEngine(width=1024, height=768, use_gpu=True)
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()