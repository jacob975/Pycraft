#!/usr/bin/env python3
"""
Pycraft Launcher - Choose rendering mode
"""

import sys
import argparse
from game.engine import GameEngine

def main():
    parser = argparse.ArgumentParser(description='Pycraft - Minecraft-like Game')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU rendering (software)')
    parser.add_argument('--gpu', action='store_true', 
                       help='Force GPU rendering (OpenGL)')
    parser.add_argument('--width', type=int, default=1024, 
                       help='Window width (default: 1024)')
    parser.add_argument('--height', type=int, default=768, 
                       help='Window height (default: 768)')
    
    args = parser.parse_args()
    
    # Determine rendering mode
    if args.cpu and args.gpu:
        print("錯誤: 不能同時指定 --cpu 和 --gpu")
        sys.exit(1)
    elif args.cpu:
        use_gpu = False
        print("強制使用CPU渲染")
    elif args.gpu:
        use_gpu = True
        print("強制使用GPU渲染")
    else:
        # Auto-detect (prefer GPU if available)
        use_gpu = True
        print("自動選擇渲染模式...")
    
    # Create and run game
    try:
        engine = GameEngine(width=args.width, height=args.height, use_gpu=use_gpu)
        engine.run()
    except Exception as e:
        print(f"遊戲啟動失敗: {e}")
        if use_gpu:
            print("嘗試使用 --cpu 參數以CPU模式啟動")
        sys.exit(1)

if __name__ == "__main__":
    main()