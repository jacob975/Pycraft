"""
Game constants and configuration
"""
from enum import Enum

# General settings
GAME_TITLE = "Pycraft"
VERSION = "0.0.1"

# Screen settings
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60

# World settings
CHUNK_SIZE = 16  # Width and depth of a chunk in blocks
RENDER_DISTANCE = 8  # How far the player can see in blocks
RELOAD_DISTANCE = 12  # Distance in chunks to trigger loading/unloading
MAX_BLOCKS = 40960  # Max blocks to render for performance
PLAYER_HAND_REACH = 5  # How far the player can reach to place/break blocks
PLAYER_SPEED = 7.0  # Blocks per second # TODO: TO BE IMPLEMENTED
FLY_SPEED = 10.0  # Blocks per second when flying
GRAVITY = 9.81  # Gravity acceleration # TODO: TO BE IMPLEMENTED
JUMP_VELOCITY = 5.0  # Initial jump velocity # TODO: TO BE IMPLEMENTED
MOUSE_SENSITIVITY = 0.003  # Mouse look sensitivity
FOV = 70.0  # Field of view in degrees
NEAR_PLANE = 0.1  # Near clipping plane
FAR_PLANE = 100.0  # Far clipping plane

# Blocks
class BlockType(Enum):
    """Enumeration of different block types"""
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    WOOD = 4
    WATER = 5
    SAND = 6
    LEAVES = 7

BLOCK_COLORS = {
    BlockType.AIR: (0, 0, 0),
    BlockType.GRASS: (34/255, 139/255, 34/255),
    BlockType.DIRT: (139/255, 69/255, 19/255),
    BlockType.STONE: (128/255, 128/255, 128/255),
    BlockType.WOOD: (160/255, 82/255, 45/255),
    BlockType.WATER: (0, 0, 1),
    BlockType.SAND: (194/255, 178/255, 128/255),
    BlockType.LEAVES: (34/255, 139/255, 34/255),
}