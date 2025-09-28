"""Utilities for saving and loading Pycraft game sessions."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from config import VERSION
from .blocks import Block, BlockType
from .world import Chunk, World

if TYPE_CHECKING:  # pragma: no cover - import only for typing
	from .engine import GameEngine
	from .player import Player

# ---------------------------------------------------------------------------
# Save path helpers
# ---------------------------------------------------------------------------

SAVE_DIR = Path(__file__).resolve().parent.parent / "saves"
SAVE_FILE_NAME = "save.json"


def _ensure_save_dir() -> None:
	SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
	slug = re.sub(r"[^a-zA-Z0-9-_]", "-", name.strip())
	slug = re.sub(r"-+", "-", slug).strip("-")
	return slug.lower() or "world"


def _unique_slug(base_slug: str) -> str:
	candidate = base_slug
	index = 1
	while (SAVE_DIR / candidate).exists():
		candidate = f"{base_slug}-{index:02d}"
		index += 1
	return candidate


def _serialize_block_list(blocks: Dict[Tuple[int, int, int], Block]) -> List[List[int]]:
	serialized: List[List[int]] = []
	for (x, y, z), block in blocks.items():
		serialized.append([x, y, z, block.type.value])
	return serialized


def _deserialize_block_list(data: Iterable[Iterable[int]]) -> Dict[Tuple[int, int, int], Block]:
	blocks: Dict[Tuple[int, int, int], Block] = {}
	for item in data:
		try:
			x, y, z, block_value = item
			block_type = BlockType(block_value)
		except (ValueError, TypeError):
			continue
		blocks[(int(x), int(y), int(z))] = Block(block_type)
	return blocks


def _serialize_world(world: World) -> Dict[str, Any]:
	chunks: Dict[str, Any] = {}
	for (chunk_x, chunk_z), chunk in world.chunks.items():
		chunk_key = f"{chunk_x},{chunk_z}"
		chunks[chunk_key] = {
			"generated": chunk.generated,
			"blocks": _serialize_block_list(chunk.blocks),
		}
	return {
		"seed": world.seed,
		"chunks": chunks,
	}


def _deserialize_world(world: World, data: Dict[str, Any]) -> None:
	world.chunks.clear()
	if "seed" in data:
		world.seed = int(data["seed"])

	chunks_data: Dict[str, Any] = data.get("chunks", {}) or {}
	for key, chunk_data in chunks_data.items():
		try:
			chunk_x_str, chunk_z_str = key.split(",", 1)
			chunk_coords = (int(chunk_x_str), int(chunk_z_str))
		except (ValueError, AttributeError):
			continue

		chunk = Chunk(*chunk_coords)
		chunk.generated = bool(chunk_data.get("generated", True))
		chunk.blocks = _deserialize_block_list(chunk_data.get("blocks", []))
		chunk._cache_dirty = True
		world.chunks[chunk_coords] = chunk


def _serialize_player(player: "Player") -> Dict[str, Any]:
	position = player.camera.position.astype(float).tolist()
	return {
		"position": position,
		"yaw": float(player.camera.yaw),
		"pitch": float(player.camera.pitch),
		"flying": bool(player.flying),
		"selected_block": player.selected_block.value,
	}


def _apply_player_state(player: "Player", data: Dict[str, Any]) -> None:
	position = data.get("position")
	if isinstance(position, (list, tuple)) and len(position) == 3:
		player.camera.position = np.array([float(position[0]), float(position[1]), float(position[2])], dtype=float)

	if "yaw" in data:
		player.camera.yaw = float(data["yaw"])
	if "pitch" in data:
		player.camera.pitch = float(data["pitch"])
	if "flying" in data:
		player.flying = bool(data["flying"])

	selected = data.get("selected_block")
	if selected is not None:
		try:
			player.selected_block = BlockType(selected)
		except ValueError:
			pass


@dataclass
class SaveMetadata:
	"""Metadata describing a saved world."""

	identifier: str
	display_name: str
	created_at: float
	updated_at: float

	@property
	def created_at_iso(self) -> str:
		return datetime.fromtimestamp(self.created_at).isoformat()

	@property
	def updated_at_iso(self) -> str:
		return datetime.fromtimestamp(self.updated_at).isoformat()


def save_game(engine: "GameEngine", save_name: Optional[str] = None, overwrite: bool = False) -> SaveMetadata:
	"""Persist the current game state to disk and return save metadata."""

	_ensure_save_dir()

	timestamp = time.time()
	display_name = save_name.strip() if save_name else datetime.fromtimestamp(timestamp).strftime("World %Y-%m-%d %H:%M:%S")
	base_slug = _slugify(display_name)
	slug = base_slug if overwrite else _unique_slug(base_slug)

	save_path = SAVE_DIR / slug
	save_path.mkdir(parents=True, exist_ok=True)

	created_at = timestamp
	if overwrite:
		existing_file = save_path / SAVE_FILE_NAME
		if existing_file.exists():
			try:
				with open(existing_file, "r", encoding="utf-8") as fp:
					existing_state = json.load(fp)
				existing_meta = existing_state.get("metadata") or {}
				created_at = float(existing_meta.get("created_at", timestamp))
				if not save_name:
					display_name = existing_meta.get("name", display_name)
			except (OSError, json.JSONDecodeError, ValueError):
				created_at = timestamp

	metadata = SaveMetadata(
		identifier=slug,
		display_name=display_name,
		created_at=created_at,
		updated_at=timestamp,
	)

	state = {
		"metadata": {
			"id": metadata.identifier,
			"name": metadata.display_name,
			"created_at": metadata.created_at,
			"updated_at": metadata.updated_at,
			"version": VERSION,
			"renderer": engine.renderer_preference,
		},
		"world": _serialize_world(engine.world),
		"player": _serialize_player(engine.player),
		"engine": {
			"debug_mode": bool(engine.debug_mode),
			"performance_mode": bool(engine.performance_mode),
			"fps_target": getattr(engine, "fps_target", 60),
		},
	}

	with open(save_path / SAVE_FILE_NAME, "w", encoding="utf-8") as fp:
		json.dump(state, fp, ensure_ascii=False, indent=2)

	return metadata


def load_game(save_identifier: str) -> Optional[Dict[str, Any]]:
	"""Load a saved game state into memory."""

	if not save_identifier:
		return None

	save_file = SAVE_DIR / save_identifier / SAVE_FILE_NAME
	if not save_file.exists():
		return None

	try:
		with open(save_file, "r", encoding="utf-8") as fp:
			return json.load(fp)
	except (OSError, json.JSONDecodeError):
		return None


def apply_loaded_state(engine: "GameEngine", state: Dict[str, Any]) -> None:
	"""Apply a loaded state to an existing game engine."""

	if not state:
		return

	world_state = state.get("world") or {}
	_deserialize_world(engine.world, world_state)

	player_state = state.get("player") or {}
	_apply_player_state(engine.player, player_state)

	engine_state = state.get("engine") or {}
	if "debug_mode" in engine_state:
		engine.debug_mode = bool(engine_state["debug_mode"])
	if "performance_mode" in engine_state:
		engine.performance_mode = bool(engine_state["performance_mode"])
	if "fps_target" in engine_state:
		engine.fps_target = int(engine_state["fps_target"])

	metadata = state.get("metadata") or {}
	engine.loaded_metadata = metadata


def apply_world_state(world: World, state: Dict[str, Any]) -> None:
	"""Populate a world instance from serialized state."""

	_deserialize_world(world, state or {})


def apply_player_state(player: "Player", state: Dict[str, Any]) -> None:
	"""Populate a player instance from serialized state."""

	_apply_player_state(player, state or {})


def list_saves() -> List[SaveMetadata]:
	"""Return available saves ordered by newest first."""

	_ensure_save_dir()
	saves: List[SaveMetadata] = []

	for entry in SAVE_DIR.iterdir():
		if not entry.is_dir():
			continue
		save_file = entry / SAVE_FILE_NAME
		if not save_file.exists():
			continue
		try:
			with open(save_file, "r", encoding="utf-8") as fp:
				data = json.load(fp)
		except (OSError, json.JSONDecodeError):
			continue

		meta = data.get("metadata") or {}
		created = float(meta.get("created_at", meta.get("updated_at", time.time())))
		updated = float(meta.get("updated_at", created))
		display = meta.get("name") or entry.name

		saves.append(SaveMetadata(
			identifier=meta.get("id", entry.name),
			display_name=display,
			created_at=created,
			updated_at=updated,
		))

	saves.sort(key=lambda m: m.updated_at, reverse=True)
	return saves


def delete_save(save_identifier: str) -> bool:
	"""Delete a save slot. Returns True on success."""

	if not save_identifier:
		return False

	target_dir = SAVE_DIR / save_identifier
	if not target_dir.exists():
		return False

	try:
		for item in target_dir.iterdir():
			if item.is_file():
				item.unlink()
			elif item.is_dir():
				delete_save_folder(item)
		target_dir.rmdir()
		return True
	except OSError:
		return False


def delete_save_folder(path: Path) -> None:
	"""Recursively delete a folder."""

	for item in path.iterdir():
		if item.is_file():
			item.unlink()
		elif item.is_dir():
			delete_save_folder(item)
	path.rmdir()