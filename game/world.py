"""
World generation and management for Pycraft
"""

import chunk
import math
import random
from typing import Dict, Tuple, Optional, List
import numpy as np
import time
try:
    import noise
except ImportError:
    noise = None

from .blocks import Block, BlockType
from config import *

class Chunk:
    """A chunk of blocks in the world"""
    
    SIZE = 16  # 16x16x16 blocks per chunk
    
    def __init__(self, x: int, z: int):
        self.x = x
        self.z = z
        self.blocks: Dict[Tuple[int, int, int], Block] = {}  # to store only non-air blocks
        self.generated = False

        # Cache for visible faces to improve performance
        self._visible_faces_cache = None
        self._cache_dirty = True
    
    def get_block(self, x: int, y: int, z: int) -> Block:
        """Get block at local coordinates"""
        pos = (x, y, z)
        return self.blocks.get(pos, Block(BlockType.AIR))
    
    def set_block(self, x: int, y: int, z: int, block_type: BlockType):
        """Set block at local coordinates"""
        pos = (x, y, z)
        if block_type == BlockType.AIR:
            # Remove air blocks from storage to save memory
            if pos in self.blocks:
                del self.blocks[pos]
        else:
            self.blocks[pos] = Block(block_type)
        
        # Invalidate visible blocks cache when blocks change
        self._cache_dirty = True

    def get_visible_faces(self) -> Dict[str, np.ndarray]:
        """Get optimized arrays of visible block data for rendering (cached)"""
        # Return cached result if available and valid
        if not self._cache_dirty and self._visible_faces_cache is not None:
            return self._visible_faces_cache
        st = time.time()
        
        # Pre-allocate lists for better performance
        positions = []
        colors = []
        block_types = []
        
        chunk_world_x = self.x * self.SIZE
        chunk_world_z = self.z * self.SIZE
        
        # Pre-compute direction vectors for neighbor checking
        directions = np.array([
            (0, 0, -1),  # north
            (0, 0, 1),   # south  
            (1, 0, 0),   # east
            (-1, 0, 0),  # west
            (0, 1, 0),   # up
            (0, -1, 0)   # down
        ])
        
        # Pre-compute block colors to avoid repeated object creation
        color_cache = Block._COLORS
        
        # Iterate through all solid blocks in this chunk
        for (x, y, z), block in self.blocks.items():
            if not block.is_solid():
                continue
            
            # Check if any face is visible (not blocked by adjacent solid block)
            has_visible_face = False
            
            for dx, dy, dz in directions:
                neighbor_x, neighbor_y, neighbor_z = x + dx, y + dy, z + dz
                
                # Check if neighbor position is within chunk bounds
                if (0 <= neighbor_x < self.SIZE and 
                    0 <= neighbor_y < 256 and  # World height limit
                    0 <= neighbor_z < self.SIZE):
                    # Get the neighboring block within this chunk
                    neighbor_block = self.get_block(neighbor_x, neighbor_y, neighbor_z)
                else:
                    # If neighbor is outside chunk bounds, assume it's air (visible face)
                    neighbor_block = Block(BlockType.AIR)
                
                # Face is visible if neighboring block is not solid
                if not neighbor_block.is_solid():
                    has_visible_face = True
                    break  # Found at least one visible face, that's enough
            
            # Only add block if it has at least one visible face
            if has_visible_face:
                world_pos = (x + chunk_world_x, y, z + chunk_world_z)
                positions.append(world_pos)
                colors.append(color_cache.get(block.type, Block._DEFAULT_COLOR))
                block_types.append(block.type)
        
        # Convert to optimized NumPy arrays
        result = {
            'positions': np.array(positions, dtype=np.float32) if positions else np.empty((0, 3), dtype=np.float32),
            'colors': np.array(colors, dtype=np.float32) if colors else np.empty((0, 3), dtype=np.float32),
            'types': np.array(block_types, dtype=object) if block_types else np.empty(0, dtype=object)
        }
        
        # Cache the result
        self._visible_faces_cache = result
        self._cache_dirty = False
        print(f"Chunk ({self.x}, {self.z}) visible faces computed in {time.time() - st:.3f}s, {len(positions)} blocks")
        return result

    def generate_terrain(self):
        """Generate terrain for this chunk"""
        if self.generated:
            return
        
        world_x = self.x * self.SIZE
        world_z = self.z * self.SIZE
        
        blocks_generated = 0
        
        for x in range(self.SIZE):
            for z in range(self.SIZE):
                # Simple height map generation
                if noise:
                    height = self._get_height_at(world_x + x, world_z + z)
                else:
                    # Fallback to simple sine wave pattern
                    height = int(30 + 10 * math.sin((world_x + x) * 0.1) * math.cos((world_z + z) * 0.1))
                
                # Generate terrain layers with simpler structure for performance
                # Reduced height range for fewer blocks
                for y in range(max(15, height - 8), height + 1):  # Smaller range
                    if y <= 10:  # Bedrock layer (reduced)
                        self.set_block(x, y, z, BlockType.STONE)
                        blocks_generated += 1
                    elif y <= height - 3:  # Shallow stone layer (reduced)
                        self.set_block(x, y, z, BlockType.STONE)
                        blocks_generated += 1
                    elif y < height:  # Dirt layer (1-2 blocks thick)
                        self.set_block(x, y, z, BlockType.DIRT)
                        blocks_generated += 1
                    elif y == height:  # Surface layer
                        # Use grass for most surfaces
                        if height > 40:
                            self.set_block(x, y, z, BlockType.STONE)
                        else:
                            self.set_block(x, y, z, BlockType.GRASS)
                        blocks_generated += 1
                
                # Reduce tree generation for better performance
                if (height < 35 and height > 25 and  # Smaller height range
                    world_z > 10 and  # Further from spawn
                    random.random() < 0.005):  # 0.5% chance (reduced from 1%)
                    surface_block = self.get_block(x, height, z)
                    if surface_block.type == BlockType.GRASS:
                        self._generate_tree(x, height + 1, z)
        
        # print(f"Generated {blocks_generated} blocks in chunk ({self.x}, {self.z})")
        self.generated = True
        # Mark cache as dirty after terrain generation
        self._cache_dirty = True
    
    def _get_height_at(self, world_x: int, world_z: int) -> int:
        """Get terrain height at world coordinates"""
        if noise:
            # Create more interesting terrain with hills and valleys (reduced complexity)
            # Base terrain layer for variety (reduced)
            base_height = noise.pnoise2(world_x * 0.02, world_z * 0.02,  # Less detail
                                      octaves=2, persistence=0.4, lacunarity=1.8)  # Simpler
            
            # Create hills in visible area in front of spawn (smaller)
            hill_height = 0
            hill_center_x, hill_center_z = 8, 18  # Closer hill
            hill_distance = math.sqrt((world_x - hill_center_x)**2 + (world_z - hill_center_z)**2)
            
            if hill_distance < 8:  # Smaller hill radius
                hill_factor = (1 - hill_distance / 8) ** 1.2
                hill_height = hill_factor * 8  # Smaller hill (8 blocks high)
            
            # Flat area around spawn point for easy start
            spawn_distance = math.sqrt((world_x - 8)**2 + (world_z - 8)**2)
            if spawn_distance < 6:
                # Very flat area around spawn
                total_height = 28 + base_height * 1 + hill_height * 0.1
            elif world_z < 15:
                # Gentle terrain near spawn
                total_height = 28 + base_height * 3 + hill_height * 0.3
            else:
                # Normal terrain with full hills
                total_height = 30 + base_height * 6 + hill_height
            
            return max(25, int(total_height))  # Minimum height of 25
        else:
            # Fallback without noise - create simple but visible hill pattern
            # Flat area around spawn
            spawn_distance = math.sqrt((world_x - 8)**2 + (world_z - 8)**2)
            if spawn_distance < 6:
                return 28  # Flat ground around spawn
            
            # Create a visible hill in front of spawn
            hill_center_x, hill_center_z = 8, 20
            hill_distance = math.sqrt((world_x - hill_center_x)**2 + (world_z - hill_center_z)**2)
            
            if hill_distance < 10:
                hill_factor = (1 - hill_distance / 10) ** 2
                return int(28 + hill_factor * 15)
            else:
                # Gentle rolling terrain
                return int(29 + 3 * math.sin(world_x * 0.2) * math.cos(world_z * 0.15))
    
    def _generate_tree(self, x: int, y: int, z: int):
        """Generate a simple tree at the given position"""
        tree_height = random.randint(4, 7)
        
        # Tree trunk
        for dy in range(tree_height):
            if y + dy < 256:  # Height limit
                self.set_block(x, y + dy, z, BlockType.WOOD)
        
        # Tree leaves (simple sphere)
        leaf_y = y + tree_height - 1
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in range(-1, 3):
                    if abs(dx) + abs(dz) + abs(dy) <= 3:
                        leaf_x, leaf_z = x + dx, z + dz
                        if (0 <= leaf_x < self.SIZE and 0 <= leaf_z < self.SIZE and
                            leaf_y + dy < 256 and random.random() < 0.7):
                            # Only place leaves if within chunk bounds
                            current_block = self.get_block(leaf_x, leaf_y + dy, leaf_z)
                            if not current_block.is_solid():
                                self.set_block(leaf_x, leaf_y + dy, leaf_z, BlockType.GRASS)  # Using grass as leaves

class World:
    """Game world containing chunks and blocks"""
    
    def __init__(self, seed: int = None, use_multiprocessing: bool = True):
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.seed = seed or random.randint(0, 1000000)
        random.seed(self.seed)
        
        if noise:
            # Set noise seed for consistent terrain
            random.seed(self.seed)
    
    @property
    def chunk_size(self) -> int:
        """Get the chunk size"""
        return Chunk.SIZE
    
    def get_chunk_coords(self, world_x: int, world_z: int) -> Tuple[int, int]:
        """Convert world coordinates to chunk coordinates"""
        chunk_x = math.floor(world_x / Chunk.SIZE)
        chunk_z = math.floor(world_z / Chunk.SIZE)
        return (chunk_x, chunk_z)
    
    def get_local_coords(self, world_x: int, world_y: int, world_z: int) -> Tuple[int, int, int]:
        """Convert world coordinates to local chunk coordinates"""
        local_x = world_x % Chunk.SIZE
        local_z = world_z % Chunk.SIZE
        return (local_x, world_y, local_z)
    
    def get_or_create_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """Get existing chunk or create new one"""
        chunk_coords = (chunk_x, chunk_z)
        
        if chunk_coords not in self.chunks:
            chunk = Chunk(chunk_x, chunk_z)
            chunk.generate_terrain()
            self.chunks[chunk_coords] = chunk
        
        return self.chunks[chunk_coords]
    
    def get_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """Get existing chunk without creating it"""
        chunk_coords = (chunk_x, chunk_z)
        return self.chunks.get(chunk_coords, None)
    
    def get_block(self, world_x: int, world_y: int, world_z: int) -> Block:
        """Get block at world coordinates"""
        if world_y < 0 or world_y >= 256:  # Height limits
            return Block(BlockType.AIR)
        
        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        
        local_x, local_y, local_z = self.get_local_coords(world_x, world_y, world_z)
        return chunk.get_block(local_x, local_y, local_z)
    
    def set_block(self, world_x: int, world_y: int, world_z: int, block_type: BlockType):
        """Set block at world coordinates"""
        if world_y < 0 or world_y >= 256:  # Height limits
            return
        
        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        
        local_x, local_y, local_z = self.get_local_coords(world_x, world_y, world_z)
        chunk.set_block(local_x, local_y, local_z, block_type)

    def get_visible_chunks(self, center_x: int, center_z: int, render_distance: int = 2) -> List[Chunk]:
        """Get list of chunks that should be visible/loaded, sorted by distance from center"""
        visible_chunks = []

        center_chunk_x, center_chunk_z = self.get_chunk_coords(center_x, center_z)

        # Debug: Print chunk loading info occasionally
        if hasattr(self, '_debug_chunk_counter'):
            self._debug_chunk_counter += 1
        else:
            self._debug_chunk_counter = 0

        # Collect chunks with their distances
        chunk_distance_pairs = []

        for dx in range(-render_distance, render_distance + 1):
            for dz in range(-render_distance, render_distance + 1):
                chunk_x = center_chunk_x + dx
                chunk_z = center_chunk_z + dz

                # Only load chunks within circular distance
                distance = math.sqrt(dx*dx + dz*dz)
                if distance <= render_distance:
                    chunk = self.get_or_create_chunk(chunk_x, chunk_z)
                    chunk_distance_pairs.append((chunk, distance))

        # Sort by distance (closest first)
        chunk_distance_pairs.sort(key=lambda x: x[1])

        # Extract sorted chunks
        visible_chunks = [chunk for chunk, _ in chunk_distance_pairs]
        return visible_chunks