"""The window and the main loop.

Frames run at ``FPS``. Each frame the current screen handles the events,
updates with the frame time and draws onto the logical canvas, which is
then scaled ``SCALE`` times into the window with no smoothing.
"""

import random

import pygame

from ui.config import CANVAS_SIZE, FPS, GAME_NAME, SEED_COUNT, WINDOW_SIZE
from ui.screens import StartScreen
from ui.sprites import Sprites


class App:
    """Holds the current screen, the sprites and the UI's own seed RNG.

    ``rng`` only picks seeds for new games; it is never the global
    ``random`` state.
    """

    def __init__(self, window: pygame.Surface, rng: random.Random | None = None) -> None:
        self.window = window
        self.canvas = pygame.Surface(CANVAS_SIZE)
        self.rng = rng if rng is not None else random.Random()
        self.sprites = Sprites()
        self.running = True
        self.screen = StartScreen(self)

    def new_seed(self) -> int:
        return self.rng.randrange(SEED_COUNT)

    def switch(self, screen) -> None:
        self.screen = screen

    def quit(self) -> None:
        self.running = False

    def frame(self, events, dt_ms: float) -> None:
        """One frame: events, update, draw, scale into the window."""
        for event in events:
            if event.type == pygame.QUIT:
                self.quit()
            else:
                self.screen.handle(event)
        if not self.running:
            return
        self.screen.update(dt_ms)
        self.screen.draw(self.canvas)
        pygame.transform.scale(self.canvas, self.window.get_size(), self.window)

    def run(self) -> None:
        clock = pygame.time.Clock()
        while self.running:
            dt_ms = clock.tick(FPS)
            self.frame(pygame.event.get(), dt_ms)
            pygame.display.flip()


def main() -> None:
    # Display only: no mixer, so no audio device is needed.
    pygame.display.init()
    try:
        window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(GAME_NAME)
        App(window).run()
    finally:
        pygame.quit()
