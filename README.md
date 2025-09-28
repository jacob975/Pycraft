# Pycraft - A Minecraft-like Game in Python

A simple Minecraft-like voxel game built with Python and Pygame.

## Features

- 3D voxel world with simple graphics
- Procedural terrain generation
- Block placing and breaking
- First-person camera movement
- Basic chunk system
- Save and load worlds via the main menu

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
python main.py
```

## Controls

- **WASD**: Move around
- **Mouse**: Look around
- **Space**: Move up (fly)
- **Shift**: Move down (fly)
- **Left Click**: Break block
- **Right Click**: Place block
- **1-4**: Select block type
- **ESC**: Open pause menu (resume, save & quit)

## Project Structure

- `main.py`: Entry point of the game
- `game/`: Main game module
  - `engine.py`: Core game engine and loop
  - `world.py`: World and chunk management
  - `player.py`: Player class and controls
  - `blocks.py`: Block types and definitions
  - `camera.py`: 3D camera system
  - `renderer.py`: 3D rendering system
  - `gpu_renderer.py`: GPU-accelerated rendering
  - `font_manager.py`: Font loading and text rendering
- `assets/`: Game assets (if any)

## Requirements

- Python 3.7+
- Pygame 2.0+
- NumPy
- Noise (for terrain generation)
- ModernGL (for 3D rendering)
- Numba (for performance optimization)
- Cython Imaging Library (Pillow)