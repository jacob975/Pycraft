"""
Block types and definitions for Pycraft
"""

from typing import Tuple
from config import *

class Block:
    """Represents a single block in the world"""
    _COLORS = BLOCK_COLORS
    _DEFAULT_COLOR = (1, 1, 1)
    
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