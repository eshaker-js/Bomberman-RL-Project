"""The screens of the app. Each one handles events, updates and draws itself.

A screen talks to the app only through ``app.switch(screen)``,
``app.quit()``, ``app.new_seed()`` and ``app.sprites``.
"""

import pygame

from ui import palette as pal
from ui.clock import TickClock
from ui.config import (
    ARENA_H,
    ARENA_W,
    ARENA_X,
    ARENA_Y,
    CANVAS_H,
    CANVAS_W,
    GAME_NAME,
    GO_MS,
    READY_MS,
)
from ui.font import draw_text_centered
from ui.input import InputBuffer
from ui.menu import Menu
from ui.render import draw_box, draw_menu, draw_panel, draw_scene
from ui.scene import initial_facings, scene, update_facings
from ui.session import MODE_BOTS, MODE_FREE, GameSession

CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER)

START_FREE = "FREE WORLD"
START_BOTS = "PLAYER VS BOTS"
START_SPECTATOR = "AGENT SPECTATOR"


def menu_key(menu: Menu, key: int) -> bool:
    """Apply up/down to ``menu``. Return ``True`` if the key confirms the choice."""
    if key == pygame.K_UP:
        menu.move(-1)
    elif key == pygame.K_DOWN:
        menu.move(1)
    return key in CONFIRM_KEYS


class Screen:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, event) -> None:
        pass

    def update(self, dt_ms: float) -> None:
        pass

    def draw(self, canvas) -> None:
        pass


class StartScreen(Screen):
    """Game name and the three modes. Esc quits."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.menu = Menu([START_FREE, START_BOTS, START_SPECTATOR])

    def handle(self, event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.app.quit()
        elif menu_key(self.menu, event.key):
            self.choose(self.menu.selected)

    def choose(self, option: str) -> None:
        if option == START_FREE:
            self.app.switch(GameScreen(self.app, GameSession(MODE_FREE, self.app.new_seed())))
        elif option == START_BOTS:
            # Rule bots until the opponent screen exists.
            session = GameSession(MODE_BOTS, self.app.new_seed(), "rule")
            self.app.switch(GameScreen(self.app, session))
        else:
            self.app.switch(NotReadyScreen(self.app))

    def draw(self, canvas) -> None:
        canvas.fill(pal.rgb(pal.BLACK))
        cx = CANVAS_W // 2
        draw_text_centered(canvas, GAME_NAME, cx, 28, pal.YELLOW, scale=3)
        sprites = self.app.sprites
        for i in range(4):
            x = cx - 46 + i * 24
            canvas.blit(sprites.get(("char", i, "down", 0)), (x, 60))
        draw_menu(canvas, self.menu, cx, 104, spacing=14)
        draw_text_centered(canvas, "ARROWS + ENTER    ESC QUIT", cx, CANVAS_H - 16, pal.DARK_GREY)


class NotReadyScreen(Screen):
    """Placeholder for the agent spectator mode: no trained agent exists yet."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.menu = Menu(["BACK"])

    def handle(self, event) -> None:
        if event.type == pygame.KEYDOWN and (
            event.key == pygame.K_ESCAPE or menu_key(self.menu, event.key)
        ):
            self.app.switch(StartScreen(self.app))

    def draw(self, canvas) -> None:
        canvas.fill(pal.rgb(pal.BLACK))
        cx = CANVAS_W // 2
        draw_text_centered(canvas, START_SPECTATOR, cx, 50, pal.LIGHT_GREY, scale=2)
        draw_text_centered(canvas, "NOT READY", cx, 80, pal.RED, scale=3)
        draw_menu(canvas, self.menu, cx, 130)


class GameScreen(Screen):
    """One running game: READY/GO, then one tick every ``TICK_MS``.

    The human's action for a tick is whatever ``InputBuffer`` holds when the
    tick is due. The screen never changes the game except through
    ``session.advance``. Between ticks it draws ``scene(prev, cur, t)`` with
    ``t`` the clock's progress, so each tick's animation ends exactly on the
    new state. After the final tick it lets that animation finish, then
    stops (``finished``).
    """

    def __init__(self, app, session: GameSession) -> None:
        super().__init__(app)
        self.session = session
        self.input = InputBuffer()
        self.clock = TickClock()
        self.start()

    def start(self) -> None:
        """(Re)start the countdown and the clock for the session's current state."""
        self.ready_ms = READY_MS
        self.clock.reset()
        self.input.clear()
        self.facings = initial_facings(self.session.cur)
        self.finished = False

    def handle(self, event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.switch(StartScreen(self.app))
            else:
                self.input.key_down(event.key)
        elif event.type == pygame.KEYUP:
            self.input.key_up(event.key)
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.input.release_all()

    def update(self, dt_ms: float) -> None:
        if self.ready_ms > 0:
            self.ready_ms -= dt_ms
            return
        if self.finished:
            return
        if self.session.over:
            # The final tick's animation is still playing; stop when it ends.
            if self.clock.update(dt_ms):
                self.clock.finish()
                self.finished = True
            return
        if self.clock.update(dt_ms):
            prev = self.session.cur
            cur = self.session.advance(self.input.consume())
            self.facings = update_facings(self.facings, prev, cur)

    def draw(self, canvas) -> None:
        canvas.fill(pal.rgb(pal.BLACK))
        state = self.session.cur
        items = scene(self.session.prev, state, self.clock.progress, self.facings)
        draw_scene(canvas, self.app.sprites, items)
        draw_panel(canvas, self.app.sprites, self.session, state)
        if self.ready_ms > 0:
            text = "GO!" if self.ready_ms <= GO_MS else "READY..."
            color = pal.GREEN if self.ready_ms <= GO_MS else pal.YELLOW
            draw_box(canvas, [(text, color, 2)], ARENA_X + ARENA_W // 2, ARENA_Y + ARENA_H // 2)
