# Project Guidelines

## Code Style
- Use Python with type hints, following existing patterns in `game/engine.py`, `game/world.py`, and `game/saves.py`.
- Preserve current naming and structure style (dataclasses for structured data, enums for fixed sets, clear module boundaries in `game/`).
- Keep user-facing text consistent with existing language usage in this project (English and Traditional Chinese both appear in current UI/log messages).
- Prefer small, focused edits over broad refactors unless explicitly requested.

## Architecture
- Entry point is `main.py`.
- Core runtime orchestration is `game/engine.py` (`GameEngine`).
- World/chunk generation and storage live in `game/world.py`.
- Player movement/input logic is in `game/player.py`.
- GPU rendering path is in `game/gpu_renderer.py` (ModernGL).
- Save/load logic is in `game/saves.py`.
- Shared constants are defined in `config.py`.

## Build and Test
- Install dependencies: `pip install -r requirements.txt`.
- Run game: `python main.py`.
- There is currently no automated test suite in this repository; validate changes with targeted reasoning and lightweight runtime checks where possible.

## Conventions
- This project is GPU-first: CPU renderer fallback is not implemented. Do not assume non-OpenGL fallback behavior.
- Keep save compatibility in mind when touching serialization/state fields in `game/saves.py`, `game/engine.py`, and `game/world.py`.
- Avoid changing block/chunk coordinate conventions without tracing call sites across world, player, and renderer modules.
- Physics-related constants exist but some behavior is intentionally incomplete; avoid silently introducing partially wired physics changes.

## Known Pitfalls
- Headless or non-OpenGL environments can fail at runtime due to ModernGL requirements.
- Font availability can affect Chinese text rendering.

## References
- See `README.md` for installation, controls, and high-level structure.
