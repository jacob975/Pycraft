"""Simple loading screen with progress bar for Pycraft."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame


@dataclass
class LoadingColors:
    background: Tuple[int, int, int] = (22, 24, 36)
    title: Tuple[int, int, int] = (245, 245, 255)
    message: Tuple[int, int, int] = (210, 215, 240)
    bar_border: Tuple[int, int, int] = (120, 130, 170)
    bar_fill: Tuple[int, int, int] = (90, 140, 255)
    bar_bg: Tuple[int, int, int] = (40, 45, 65)


class LoadingScreen:
    """Lightweight progress indicator rendered with pygame surfaces."""

    def __init__(
        self,
        size: Tuple[int, int],
        title: str = "Loading...",
        total_steps: int = 6,
        colors: LoadingColors | None = None,
    ) -> None:
        self.width, self.height = size
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        self.message: str = ""
        self.title = title
        self.colors = colors or LoadingColors()
        self.active = True

        # Switch display to a simple pygame surface for loading feedback
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)

        # Fonts (pygame.font is expected to be initialised by caller)
        self.title_font = pygame.font.Font(None, 52)
        self.message_font = pygame.font.Font(None, 28)

        self._render(initial=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_status(self, message: str) -> None:
        """Update the status text without advancing the progress bar."""
        if not self.active:
            return
        self.message = message
        self._render()

    def advance(self, message: str) -> None:
        """Advance the progress bar by a single step and update the message."""
        if not self.active:
            return
        self.current_step = min(self.total_steps, self.current_step + 1)
        self.message = message
        self._render()

    def complete(self, message: str | None = None) -> None:
        """Mark the loading as complete and optionally update the message."""
        if not self.active:
            return
        self.current_step = self.total_steps
        if message is not None:
            self.message = message
        self._render()

    def finish(self) -> None:
        """Stop rendering the loading screen and release its surface."""
        self.active = False
        self.surface = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _render(self, initial: bool = False) -> None:
        """Render the loading screen to the current pygame surface."""
        if not self.active or not self.surface:
            return

        current_surface = pygame.display.get_surface()
        if current_surface is not self.surface:
            # Display surface has changed (e.g., OpenGL context created). Stop rendering.
            self.finish()
            return

        # Pump events so the window remains responsive during loading
        pygame.event.pump()

        self.surface.fill(self.colors.background)

        title_surface = self.title_font.render(self.title, True, self.colors.title)
        title_rect = title_surface.get_rect(center=(self.width // 2, self.height // 2 - 80))
        self.surface.blit(title_surface, title_rect)

        if self.message:
            message_surface = self.message_font.render(self.message, True, self.colors.message)
            message_rect = message_surface.get_rect(center=(self.width // 2, self.height // 2 + 20))
            self.surface.blit(message_surface, message_rect)

        bar_width = int(self.width * 0.55)
        bar_height = 30
        bar_x = (self.width - bar_width) // 2
        bar_y = self.height // 2 - bar_height // 2 + 60

        pygame.draw.rect(
            self.surface,
            self.colors.bar_bg,
            pygame.Rect(bar_x, bar_y, bar_width, bar_height),
            border_radius=8,
        )
        pygame.draw.rect(
            self.surface,
            self.colors.bar_border,
            pygame.Rect(bar_x, bar_y, bar_width, bar_height),
            width=2,
            border_radius=8,
        )

        progress_ratio = self.current_step / self.total_steps
        fill_width = int(bar_width * progress_ratio)
        if fill_width > 0:
            pygame.draw.rect(
                self.surface,
                self.colors.bar_fill,
                pygame.Rect(bar_x + 3, bar_y + 3, fill_width - 6 if fill_width > 6 else fill_width, bar_height - 6),
                border_radius=6,
            )

        pygame.display.flip()

        # On the very first draw, show a tiny delay so the window appears
        if initial:
            pygame.time.delay(50)