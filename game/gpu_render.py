"""Alias module to maintain naming flexibility.

Engine currently imports `gpu_renderer`, but user requested `gpu_render.py`.
We re-export GPURenderer so either name works.
"""

from .gpu_renderer import GPURenderer  # noqa: F401

__all__ = ["GPURenderer"]
