# Pycraft - A Minecraft-like Game in Python

A simple Minecraft-like voxel game built with Python and Pygame.

## Features

- 3D voxel world with simple graphics
- Procedural terrain generation
- Block placing and breaking
- First-person camera movement
- Basic chunk system

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
- **ESC**: Exit game

## Project Structure

- `main.py`: Entry point of the game
- `game/`: Main game module
  - `engine.py`: Core game engine and loop
  - `world.py`: World and chunk management
  - `player.py`: Player class and controls
  - `blocks.py`: Block types and definitions
  - `camera.py`: 3D camera system
  - `renderer.py`: 3D rendering system
- `assets/`: Game assets (if any)

## Requirements

- Python 3.7+
- Pygame 2.0+
- NumPy
- Noise (for terrain generation)
- PyOpenGL (for 3D rendering)
- PyOpenGL_accelerate (optional, for performance)