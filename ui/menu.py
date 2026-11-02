"""A vertical list of options with a cursor. Pure: no pygame."""


class Menu:
    """Options chosen with up/down (wrapping around) and confirmed with Enter."""

    def __init__(self, options: list[str]) -> None:
        if not options:
            raise ValueError("a menu needs at least one option")
        self.options = list(options)
        self.index = 0

    @property
    def selected(self) -> str:
        return self.options[self.index]

    def move(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.options)
