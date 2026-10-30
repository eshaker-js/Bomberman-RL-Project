"""Pygame window for the Bomberman environment.

Controls (one action per key press; holding a key does not repeat):
  Arrow keys or WASD  move
  Space               place a bomb
  N or . (period)     wait
  R                   reset
  Esc                 quit

The clock only keeps the window responsive. It never calls ``step``.
"""

import pygame

from game.environment import MAP_HEIGHT, MAP_WIDTH, Action, BombermanEnv, Cell

CELL_SIZE = 48
HUD_HEIGHT = 136
WINDOW_WIDTH = MAP_WIDTH * CELL_SIZE
WINDOW_HEIGHT = MAP_HEIGHT * CELL_SIZE + HUD_HEIGHT

COLOR_BACKGROUND = (24, 24, 28)
COLOR_FLOOR = (186, 196, 170)
COLOR_WALL = (52, 56, 64)
COLOR_BLOCK = (176, 122, 62)
COLOR_PLAYER = (47, 108, 214)
COLOR_PLAYER_DEAD = (210, 64, 64)
COLOR_BOMB = (28, 28, 32)
COLOR_BOMB_TEXT = (255, 220, 80)
COLOR_EXPLOSION = (242, 132, 34)
COLOR_HUD = (32, 34, 40)
COLOR_TEXT = (236, 236, 236)
COLOR_ALIVE = (140, 210, 140)
COLOR_DEATH = (255, 214, 214)
COLOR_DEATH_BG = (128, 32, 32)

KEY_TO_ACTION = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
    pygame.K_n: Action.WAIT,
    pygame.K_PERIOD: Action.WAIT,
    pygame.K_SPACE: Action.BOMB,
}


def cell_rect(x: int, y: int) -> pygame.Rect:
    return pygame.Rect(
        x * CELL_SIZE + 1,
        y * CELL_SIZE + 1,
        CELL_SIZE - 2,
        CELL_SIZE - 2,
    )


def handle_keydown(env: BombermanEnv, key: int) -> bool:
    """Apply one key. Return False when the window should close.

    After death, play keys are ignored. Reset and quit still work, and the
    caller keeps pumping events so the window does not freeze.
    """
    if key == pygame.K_ESCAPE:
        return False
    if key == pygame.K_r:
        env.reset()
        return True
    if env.dead:
        return True
    action = KEY_TO_ACTION.get(key)
    if action is not None:
        env.step(action)
    return True


def draw(screen: pygame.Surface, font: pygame.font.Font, env: BombermanEnv) -> None:
    screen.fill(COLOR_BACKGROUND)
    _draw_board(screen, font, env)
    _draw_hud(screen, font, env)
    if env.dead:
        _draw_death_banner(screen, font)


def _draw_board(screen: pygame.Surface, font: pygame.font.Font, env: BombermanEnv) -> None:
    for y, row in enumerate(env.grid):
        for x, cell in enumerate(row):
            rect = cell_rect(x, y)
            if cell is Cell.BLOCK:
                pygame.draw.rect(screen, COLOR_FLOOR, rect)
                pygame.draw.rect(screen, COLOR_BLOCK, rect.inflate(-12, -12))
            elif cell is Cell.WALL:
                pygame.draw.rect(screen, COLOR_WALL, rect)
            else:
                pygame.draw.rect(screen, COLOR_FLOOR, rect)

    # Drawn from env.explosion only. This loop does not damage or tick time.
    for x, y in env.explosion:
        pygame.draw.rect(screen, COLOR_EXPLOSION, cell_rect(x, y))

    bomb_rect = None
    if env.bomb is not None:
        bomb_rect = cell_rect(env.bomb.x, env.bomb.y)
        pygame.draw.circle(screen, COLOR_BOMB, bomb_rect.center, CELL_SIZE // 4)

    player_rect = cell_rect(env.player_x, env.player_y)
    player_color = COLOR_PLAYER_DEAD if env.dead else COLOR_PLAYER
    pygame.draw.circle(screen, player_color, player_rect.center, CELL_SIZE // 3)

    # Drawn after the player so the fuse stays visible while they stand on it.
    if env.bomb is not None and bomb_rect is not None:
        label = font.render(str(env.bomb.countdown), True, COLOR_BOMB_TEXT)
        screen.blit(label, label.get_rect(center=bomb_rect.center))


def _draw_hud(screen: pygame.Surface, font: pygame.font.Font, env: BombermanEnv) -> None:
    top = MAP_HEIGHT * CELL_SIZE
    pygame.draw.rect(screen, COLOR_HUD, (0, top, WINDOW_WIDTH, HUD_HEIGHT))
    if env.bomb is None:
        bomb_line = "Bomb countdown: none"
    else:
        bomb_line = f"Bomb countdown: {env.bomb.countdown}"
    if env.dead:
        status = "You died. Press R to reset."
        status_color = COLOR_DEATH
    else:
        status = "Alive"
        status_color = COLOR_ALIVE

    lines = [
        (f"Steps: {env.steps}", COLOR_TEXT),
        (bomb_line, COLOR_TEXT),
        (status, status_color),
        ("Move: Arrows or WASD   Bomb: Space", COLOR_TEXT),
        ("Wait: N or .    Reset: R    Quit: Esc", COLOR_TEXT),
    ]
    y = top + 8
    for text, color in lines:
        image = font.render(text, True, color)
        screen.blit(image, (12, y))
        y += font.get_linesize()


def _draw_death_banner(screen: pygame.Surface, font: pygame.font.Font) -> None:
    text = font.render("You died. Press R to reset.", True, COLOR_DEATH, COLOR_DEATH_BG)
    center = (WINDOW_WIDTH // 2, MAP_HEIGHT * CELL_SIZE // 2)
    screen.blit(text, text.get_rect(center=center))


def main() -> None:
    pygame.init()
    # Disabled key repeat: one KEYDOWN per press, so holding a key is one action.
    pygame.key.set_repeat(0)
    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bomberman RL")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        env = BombermanEnv()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and not getattr(event, "repeat", 0):
                    running = handle_keydown(env, event.key)

            draw(screen, font, env)
            pygame.display.flip()
            # Caps the frame rate only. The environment does not advance here.
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
