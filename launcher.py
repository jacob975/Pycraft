#!/usr/bin/env python3
"""
Pycraft Launcher - Choose rendering mode with ModernGL support
"""

import sys
import argparse
import pygame
from game.engine import GameEngine
from game.menu import show_main_menu

def main():
    parser = argparse.ArgumentParser(description='Pycraft - Minecraft-like Game with GPU Acceleration')
    parser.add_argument('--width', type=int, default=1024, 
                       help='Window width (default: 1024)')
    parser.add_argument('--height', type=int, default=768, 
                       help='Window height (default: 768)')
    parser.add_argument('--menu-mode', choices=['auto', 'moderngl', 'pygame'], 
                       default='auto',
                       help='Menu rendering mode: auto (try ModernGL first), moderngl (force GPU), pygame (force CPU)')
    
    args = parser.parse_args()
        
    try:
        # Initialize pygame once at the beginning
        pygame.init()
        pygame.font.init()
        
        print(f"🎮 Pycraft Launcher - {args.width}x{args.height}")
        print(f"🎨 Menu mode: {args.menu_mode}")
        # Show main menu with specified mode
        if args.menu_mode == 'moderngl':
            print("🚀 Forcing ModernGL GPU menu...")
            selected_option = show_main_menu(args.width, args.height, prefer_moderngl=True)
        elif args.menu_mode == 'pygame':
            print("📱 Forcing standard pygame menu...")
            selected_option = show_main_menu(args.width, args.height, prefer_moderngl=False)
        else:  # auto mode
            print("🎯 Auto-selecting best available menu system...")
            selected_option = show_main_menu(args.width, args.height, prefer_moderngl=True)
        
        # Handle menu selection
        if selected_option == 'exit' or selected_option is None:
            print("👋 Exiting Pycraft. Thanks for playing!")
            return
        
        elif selected_option == 'new_world':
            print("🌍 Starting new world...")
            engine = GameEngine(width=args.width, height=args.height, use_gpu=True)
            engine.run()
            
        elif selected_option == 'load_world':
            print("📁 Load world feature not implemented yet.")
            print("🔄 Starting new world instead...")
            engine = GameEngine(width=args.width, height=args.height, use_gpu=True)
            engine.run()
    except Exception as e:
        print(f"❌ Game startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()