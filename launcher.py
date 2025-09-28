#!/usr/bin/env python3
"""
Pycraft Launcher - Choose rendering mode
"""

import sys
import argparse
import pygame
from game.engine import GameEngine
from game.menu import show_main_menu

def main():
    parser = argparse.ArgumentParser(description='Pycraft - Minecraft-like Game')
    parser.add_argument('--width', type=int, default=1024, 
                       help='Window width (default: 1024)')
    parser.add_argument('--height', type=int, default=768, 
                       help='Window height (default: 768)')
    parser.add_argument('--skip-menu', action='store_true',
                       help='Skip main menu and start game directly')
    
    args = parser.parse_args()
        
    try:
        # Initialize pygame once at the beginning
        pygame.init()
        pygame.font.init()
        
        if args.skip_menu:
            # Skip menu and start game directly
            print("跳過主選單，直接開始遊戲...")
            engine = GameEngine(width=args.width, height=args.height, use_gpu=True)
            engine.run()
        else:
            # Show main menu first
            screen = pygame.display.set_mode((args.width, args.height))
            selected_option = show_main_menu(width=args.width, height=args.height, screen=screen)
            
            if selected_option == 'exit' or selected_option is None:
                print("退出 Pycraft。感謝遊玩!")
                return
            
            elif selected_option == 'new_world':
                print("開始新世界...")
                # Reuse the same pygame context and screen
                engine = GameEngine(width=args.width, height=args.height, use_gpu=True, screen=screen)
                engine.run()
                
            elif selected_option == 'load_world':
                print("載入世界功能尚未實作。")
                print("改為開始新世界...")
                # Reuse the same pygame context and screen
                engine = GameEngine(width=args.width, height=args.height, use_gpu=True, screen=screen)
                engine.run()
    except Exception as e:
        print(f"遊戲啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()