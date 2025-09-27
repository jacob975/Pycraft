"""
Block types and definitions for Pycraft
"""

from enum import Enum
from typing import Tuple

class BlockType(Enum):
    """Enumeration of different block types"""
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    WOOD = 4

class Block:
    """Represents a single block in the world"""
    _COLORS = {
        BlockType.AIR: (0, 0, 0),
        BlockType.GRASS: (34, 139, 34),
        BlockType.DIRT: (139, 69, 19),
        BlockType.STONE: (128, 128, 128),
        BlockType.WOOD: (160, 82, 45),
    }
    _DEFAULT_COLOR = (255, 255, 255)
    
    def __init__(self, block_type: BlockType = BlockType.AIR):
        self.type = block_type
        self.solid = block_type != BlockType.AIR
    
    def is_solid(self) -> bool:
        """Check if the block is solid (not air)"""
        return self.solid
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get the color of the block for rendering"""
        return self._COLORS.get(self.type, self._DEFAULT_COLOR)

    def __str__(self):
        return f"Block({self.type.name})"