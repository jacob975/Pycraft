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

AIR_BLOCK = Block(BlockType.AIR)

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
        self._world_seed = 0

    def _height_variant(self, x: int, y: int, z: int) -> int:
        """Pick one of three grass heights deterministically per position."""
        n = (x * 374761393) ^ (y * 668265263) ^ (z * 2147483647) ^ (self._world_seed * 1274126177)
        return n % 3

    def _grass_texture_variant(self, x: int, y: int, z: int) -> int:
        """Pick one of three tall grass textures deterministically per position."""
        n = (x * 1597334677) ^ (y * 3812015801) ^ (z * 958689641) ^ (self._world_seed * 122949829)
        return n % 3

    def mark_dirty(self):
        """Mark cached face data dirty so it will be rebuilt on next render."""
        self._cache_dirty = True
    
    def get_block(self, x: int, y: int, z: int) -> Block:
        """Get block at local coordinates"""
        pos = (x, y, z)
        return self.blocks.get(pos, AIR_BLOCK)
    
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
        self.mark_dirty()

    def get_visible_faces(self, world: Optional['World'] = None) -> Dict[str, np.ndarray]:
        """Get optimized arrays of visible block data for rendering (cached)"""
        # Return cached result if available and valid
        if not self._cache_dirty and self._visible_faces_cache is not None:
            return self._visible_faces_cache

        # Pre-allocate lists for better performance
        positions = []
        colors = []
        face_ids = []
        texture_layers = []
        
        chunk_world_x = self.x * self.SIZE
        chunk_world_z = self.z * self.SIZE
        
        # Face order must match renderer face lookup in gpu_renderer.py
        directions = (
            (0, 0, -1),  # north
            (0, 0, 1),   # south
            (1, 0, 0),   # east
            (-1, 0, 0),  # west
            (0, 1, 0),   # up
            (0, -1, 0),  # down
        )
        
        # Pre-compute block colors to avoid repeated object creation
        color_cache = Block._COLORS
        
        # Iterate through all solid blocks in this chunk
        for (x, y, z), block in self.blocks.items():
            if not block.is_solid():
                continue
            
            world_pos = (x + chunk_world_x, y, z + chunk_world_z)

            # Add one instance per visible face instead of one per block
            for face_idx, (dx, dy, dz) in enumerate(directions):
                neighbor_x, neighbor_y, neighbor_z = x + dx, y + dy, z + dz
                
                # Check if neighbor position is within chunk bounds
                if (0 <= neighbor_x < self.SIZE and 
                    0 <= neighbor_y < 256 and  # World height limit
                    0 <= neighbor_z < self.SIZE):
                    # Get the neighboring block within this chunk
                    neighbor_block = self.get_block(neighbor_x, neighbor_y, neighbor_z)
                else:
                    if world is not None:
                        neighbor_world_x = chunk_world_x + neighbor_x
                        neighbor_world_z = chunk_world_z + neighbor_z
                        neighbor_block = world.get_block_no_create(neighbor_world_x, neighbor_y, neighbor_world_z)
                    else:
                        # If world context is unavailable, treat outside-chunk neighbors as air.
                        neighbor_block = AIR_BLOCK
                
                # Face is visible if neighboring block is not solid
                if not neighbor_block.is_solid():
                    positions.append(world_pos)
                    colors.append(color_cache.get(block.type, Block._DEFAULT_COLOR))
                    face_ids.append(face_idx)

                    # Texture layer layout is defined in gpu_renderer._load_block_textures.
                    if block.type == BlockType.GRASS:
                        # Side faces use dedicated grass_side texture; top/bottom use grass texture.
                        texture_layers.append(3 if face_idx <= 3 else 4)
                    elif block.type == BlockType.DIRT:
                        texture_layers.append(5)
                    elif block.type == BlockType.STONE:
                        texture_layers.append(6)
                    elif block.type == BlockType.WOOD:
                        texture_layers.append(7)
                    elif block.type in (BlockType.LEAF, BlockType.LEAVES):
                        texture_layers.append(8)
                    else:
                        texture_layers.append(-1)
        
        # Convert to optimized NumPy arrays
        # Render crossed-strip vegetation after solid faces.
        for (x, y, z), block in self.blocks.items():
            if block.type != BlockType.TALL_GRASS:
                continue

            # Keep grass attached to grass blocks only.
            if y <= 0:
                continue
            below = self.get_block(x, y - 1, z)
            if below.type != BlockType.GRASS:
                continue

            world_pos = (x + chunk_world_x, y, z + chunk_world_z)
            grass_color = color_cache.get(block.type, Block._DEFAULT_COLOR)
            height_variant = self._height_variant(world_pos[0], y, world_pos[2])
            texture_variant = self._grass_texture_variant(world_pos[0], y, world_pos[2])

            # TALL_GRASS mesh: 2 crossed vertical planes (X shape).
            # Face id encoding: 6 + (height_variant * 2) + plane_idx
            # height_variant: 0->1.0m, 1->0.66m, 2->0.33m
            face_base = 6 + (height_variant * 2)
            for plane_idx in range(2):
                positions.append(world_pos)
                colors.append(grass_color)
                face_ids.append(face_base + plane_idx)
                texture_layers.append(texture_variant)

        result = {
            'positions': np.array(positions, dtype=np.float32) if positions else np.empty((0, 3), dtype=np.float32),
            'colors': np.array(colors, dtype=np.float32) if colors else np.empty((0, 3), dtype=np.float32),
            'face_ids': np.array(face_ids, dtype=np.uint8) if face_ids else np.empty(0, dtype=np.uint8),
            'texture_layers': np.array(texture_layers, dtype=np.int8) if texture_layers else np.empty(0, dtype=np.int8)
        }
        
        # Cache the result
        self._visible_faces_cache = result
        self._cache_dirty = False
        return result

    def generate_terrain(self, world_seed: int = 0):
        """Generate terrain for this chunk"""
        if self.generated:
            return

        self._world_seed = world_seed
        
        world_x = self.x * self.SIZE
        world_z = self.z * self.SIZE
        chunk_rng = random.Random(
            ((self._world_seed & 0xFFFFFFFF) << 32)
            ^ ((self.x & 0xFFFFFFFF) * 73856093)
            ^ ((self.z & 0xFFFFFFFF) * 19349663)
        )
        
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
                spawn_distance = math.sqrt((world_x + x - 8) ** 2 + (world_z + z - 8) ** 2)
                if (height < 35 and height > 25 and  # Smaller height range
                    spawn_distance > 10 and  # Keep trees away from spawn in all directions
                    chunk_rng.random() < 0.005):  # 0.5% chance (reduced from 1%)
                    surface_block = self.get_block(x, height, z)
                    if surface_block.type == BlockType.GRASS:
                        self._generate_tree(x, height + 1, z)

                # Tall grass: independent random spawn on grass blocks.
                if height + 1 < 256:
                    surface_block = self.get_block(x, height, z)
                    above_block = self.get_block(x, height + 1, z)
                    if surface_block.type == BlockType.GRASS and above_block.type == BlockType.AIR:
                        if chunk_rng.random() < TALL_GRASS_SPAWN_CHANCE:
                            self.set_block(x, height + 1, z, BlockType.TALL_GRASS)
        
        # print(f"Generated {blocks_generated} blocks in chunk ({self.x}, {self.z})")
        self.generated = True
        # Mark cache as dirty after terrain generation
        self.mark_dirty()
    
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
        """Generate a tree with deterministic branches and canopy around wood supports."""
        world_x = self.x * self.SIZE + x
        world_z = self.z * self.SIZE + z
        tree_seed = (
            ((self._world_seed & 0xFFFFFFFF) << 32)
            ^ ((world_x & 0xFFFFFFFF) * 73856093)
            ^ ((y & 0xFFFFFFFF) * 19349663)
            ^ ((world_z & 0xFFFFFFFF) * 83492791)
        )
        tree_rng = random.Random(tree_seed)
        tree_height = tree_rng.randint(4, 7)

        def _in_local_bounds(bx: int, by: int, bz: int) -> bool:
            return 0 <= bx < self.SIZE and 0 <= bz < self.SIZE and 0 <= by < 256

        wood_positions: List[Tuple[int, int, int]] = []
        
        # Tree trunk
        for dy in range(tree_height):
            trunk_y = y + dy
            if _in_local_bounds(x, trunk_y, z):
                self.set_block(x, trunk_y, z, BlockType.WOOD)
                wood_positions.append((x, trunk_y, z))

        # Random side branches from upper trunk.
        branch_min_dy = max(2, tree_height // 2)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dy in range(branch_min_dy, tree_height):
            if tree_rng.random() >= 0.35:
                continue

            tree_rng.shuffle(directions)
            dir_x, dir_z = directions[0]
            branch_length = tree_rng.randint(1, 2)
            branch_x, branch_y, branch_z = x, y + dy, z

            for _ in range(branch_length):
                branch_x += dir_x
                branch_z += dir_z
                if tree_rng.random() < 0.45:
                    branch_y += 1

                if not _in_local_bounds(branch_x, branch_y, branch_z):
                    break

                current_block = self.get_block(branch_x, branch_y, branch_z)
                if current_block.is_solid() and current_block.type not in (BlockType.LEAF, BlockType.LEAVES):
                    break

                self.set_block(branch_x, branch_y, branch_z, BlockType.WOOD)
                wood_positions.append((branch_x, branch_y, branch_z))

        def _leaf_noise(nx: int, ny: int, nz: int) -> float:
            n = (
                (nx * 374761393)
                ^ (ny * 668265263)
                ^ (nz * 2147483647)
                ^ (self._world_seed * 1274126177)
            )
            return (n & 0xFFFF) / 65535.0

        # Build canopy around wood supports (upper trunk + branch tips), not in free space.
        canopy_centers: List[Tuple[int, int, int, int]] = []
        top_y = y + tree_height - 1
        for wx, wy, wz in wood_positions:
            if wy < y + 2:
                continue
            radius = 2 if wy >= top_y - 1 else 1
            canopy_centers.append((wx, wy, wz, radius))

        for center_x, center_y, center_z, radius in canopy_centers:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        leaf_x = center_x + dx
                        leaf_y = center_y + dy
                        leaf_z = center_z + dz

                        if not _in_local_bounds(leaf_x, leaf_y, leaf_z):
                            continue

                        if abs(dx) + abs(dy) + abs(dz) > radius + 1:
                            continue

                        noise = _leaf_noise(leaf_x, leaf_y, leaf_z)
                        chance = 0.86 - 0.16 * (abs(dx) + abs(dz)) - 0.08 * abs(dy)
                        chance = max(0.3, min(0.9, chance + (noise - 0.5) * 0.25))
                        if noise > chance:
                            continue

                        current_block = self.get_block(leaf_x, leaf_y, leaf_z)
                        if not current_block.is_solid():
                            self.set_block(leaf_x, leaf_y, leaf_z, BlockType.LEAF)

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

    def _mark_chunk_dirty(self, chunk_x: int, chunk_z: int):
        """Mark a chunk dirty if it already exists."""
        chunk = self.get_chunk(chunk_x, chunk_z)
        if chunk is not None:
            chunk.mark_dirty()

    def _mark_adjacent_chunks_dirty_for_chunk(self, chunk_x: int, chunk_z: int):
        """Mark orthogonal neighboring chunks dirty when this chunk is created/loaded."""
        self._mark_chunk_dirty(chunk_x + 1, chunk_z)
        self._mark_chunk_dirty(chunk_x - 1, chunk_z)
        self._mark_chunk_dirty(chunk_x, chunk_z + 1)
        self._mark_chunk_dirty(chunk_x, chunk_z - 1)

    def _mark_adjacent_chunks_dirty_for_block(self, chunk_x: int, chunk_z: int, local_x: int, local_z: int):
        """Mark neighboring chunks dirty only when a changed block sits on a chunk border."""
        if local_x == 0:
            self._mark_chunk_dirty(chunk_x - 1, chunk_z)
        elif local_x == Chunk.SIZE - 1:
            self._mark_chunk_dirty(chunk_x + 1, chunk_z)

        if local_z == 0:
            self._mark_chunk_dirty(chunk_x, chunk_z - 1)
        elif local_z == Chunk.SIZE - 1:
            self._mark_chunk_dirty(chunk_x, chunk_z + 1)
    
    def get_chunk_coords(self, world_x: int, world_z: int) -> Tuple[int, int]:
        """Convert world coordinates to chunk coordinates"""
        # Use floor division for stable chunk boundaries and negative coordinates.
        chunk_x = world_x // Chunk.SIZE
        chunk_z = world_z // Chunk.SIZE
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
            # Create and generate new chunk
            chunk = Chunk(chunk_x, chunk_z)
            chunk.generate_terrain(world_seed=self.seed)
            self.chunks[chunk_coords] = chunk
            # A new neighboring chunk changes border visibility for existing chunks.
            self._mark_adjacent_chunks_dirty_for_chunk(chunk_x, chunk_z)

        return self.chunks[chunk_coords]
    
    def get_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """Get existing chunk without creating it"""
        chunk_coords = (chunk_x, chunk_z)
        return self.chunks.get(chunk_coords)
    
    def get_block(self, world_x: int, world_y: int, world_z: int) -> Block:
        """Get block at world coordinates"""
        if world_y < 0 or world_y >= 256:  # Height limits
            return AIR_BLOCK
        
        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        
        local_x, local_y, local_z = self.get_local_coords(world_x, world_y, world_z)
        return chunk.get_block(local_x, local_y, local_z)

    def get_block_no_create(self, world_x: int, world_y: int, world_z: int) -> Block:
        """Get block at world coordinates without creating missing chunks."""
        if world_y < 0 or world_y >= 256:
            return AIR_BLOCK

        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk = self.get_chunk(chunk_x, chunk_z)
        if chunk is None:
            return AIR_BLOCK

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
        self._mark_adjacent_chunks_dirty_for_block(chunk_x, chunk_z, local_x, local_z)

    def get_visible_chunks(self, center_x: int, center_z: int, render_distance: int = 2, to_create: bool = True) -> List[Chunk]:
        """Get list of chunks that should be visible/loaded, sorted by distance from center"""
        center_chunk_x, center_chunk_z = self.get_chunk_coords(center_x, center_z)

        # Debug: Print chunk loading info occasionally
        if hasattr(self, '_debug_chunk_counter'):
            self._debug_chunk_counter += 1
        else:
            self._debug_chunk_counter = 0

        # Collect chunks with their squared distances (avoid sqrt in hot path)
        chunk_distance_pairs = []
        render_distance_sq = render_distance * render_distance

        for dx in range(-render_distance, render_distance + 1):
            for dz in range(-render_distance, render_distance + 1):
                chunk_x = center_chunk_x + dx
                chunk_z = center_chunk_z + dz

                # Only load chunks within circular distance
                distance_sq = dx * dx + dz * dz
                if distance_sq <= render_distance_sq:
                    if to_create:
                        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
                    else:
                        chunk = self.get_chunk(chunk_x, chunk_z)
                        if chunk is None:
                            continue
                    chunk_distance_pairs.append((chunk, distance_sq))

        # Sort by distance (closest first)
        chunk_distance_pairs.sort(key=lambda x: x[1])

        # Extract sorted chunks
        visible_chunks = [chunk for chunk, _ in chunk_distance_pairs]
        return visible_chunks