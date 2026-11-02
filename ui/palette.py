"""The 16-colour PICO-8 palette, used for everything the UI draws. Pure: no pygame.

Palette values from PICO-8 by Lexaloffle Games
(https://www.lexaloffle.com/pico-8.php). Sprite grids in ``ui.art`` write
colour ``i`` as the hex digit of ``i`` (``0``-``9``, ``a``-``f``).
"""

PALETTE = (
    (0, 0, 0),        # 0 black
    (29, 43, 83),     # 1 dark blue
    (126, 37, 83),    # 2 dark purple
    (0, 135, 81),     # 3 dark green
    (171, 82, 54),    # 4 brown
    (95, 87, 79),     # 5 dark grey
    (194, 195, 199),  # 6 light grey
    (255, 241, 232),  # 7 white
    (255, 0, 77),     # 8 red
    (255, 163, 0),    # 9 orange
    (255, 236, 39),   # 10 yellow
    (0, 228, 54),     # 11 green
    (41, 173, 255),   # 12 blue
    (131, 118, 156),  # 13 lavender
    (255, 119, 168),  # 14 pink
    (255, 204, 170),  # 15 peach
)

BLACK = 0
DARK_BLUE = 1
DARK_PURPLE = 2
DARK_GREEN = 3
BROWN = 4
DARK_GREY = 5
LIGHT_GREY = 6
WHITE = 7
RED = 8
ORANGE = 9
YELLOW = 10
GREEN = 11
BLUE = 12
LAVENDER = 13
PINK = 14
PEACH = 15


def rgb(index: int) -> tuple[int, int, int]:
    """RGB of palette colour ``index``."""
    return PALETTE[index]
