"""A rule-based baseline bot that plays sensibly.

Every turn the bot re-plans from the current state, in this priority order:

1. Survive. It predicts when every bomb on the board explodes, including
   chain reactions, and only takes actions after which it can still reach a
   cell no blast will ever reach. If it already stands in a future blast, it
   takes the shortest way out.
2. Win. As the agent (ID 0), once the gate is open it walks onto it.
3. Bomb. If a bomb here would hit a crate or an opponent and it can still
   escape its own blast, it places one.
4. Move. Otherwise it walks toward the nearest cell where a bomb would hit
   a crate or an opponent, avoiding cells a known blast will reach.
5. Wait, if nothing else applies.

Caution about opponents: an opponent without an active bomb could drop one
on its own cell this very action. The bot prefers actions that stay safe even
if every such opponent does, and only bombs when its escape survives that
too. If no action is safe under that worst case, it falls back to the actions
that are safe against the bombs actually on the board.

Known limits (on purpose, it is a baseline): it cannot foresee bombs placed
later than the next action, and it treats crates as permanent while planning
an escape, which only ever makes it more careful. It does not look for buffs;
it picks one up only when its path crosses it.

Timing used throughout, matching ``BombermanEnv.step``: a bomb shown with
``countdown`` c explodes during the c-th next action, after that action's
moves. A bomb placed now explodes during action ``1 + bomb_fuse`` and blocks
movement from action 2 on. Fuse, blast range and flame duration come from
``state["rules"]``. A cell burned on action e keeps burning (and killing)
through action ``e + flame_ticks - 1``, and a cell already burning with
``ticks_left`` k burns through action ``k - 1``; the bot treats every such
action as deadly on that cell.

Buffs (when the state has them): every bomb uses its own ``range`` and
``burn`` from ``state["bombs"]``; a cell within ``burn`` tiles of its bomb
burns ``flame_ticks + burn`` actions. The bot's own bombs use its own levels,
it may place another bomb while ``bombs_left`` is above 0 (only if it can
still escape every bomb on the board, chain reactions included), and an
opponent counts as a possible bomber while its ``bombs_left`` is above 0.
Without these keys every value falls back to ``state["rules"]`` and one bomb
per character, exactly as before.
"""

import random
from collections import deque
from dataclasses import dataclass

from game.environment import (
    AGENT_ID,
    BLAST_RANGE,
    GATE_POSITION,
    MAP_HEIGHT,
    MAP_WIDTH,
    Action,
    Cell,
)

MOVES = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}
# Actions that do not place a bomb, in a fixed order for reproducibility.
STEP_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.WAIT)

# How much a bomb spot is worth: crates count 1, opponents count more.
CRATE_VALUE = 1
OPPONENT_VALUE = 3
# Bomb spots checked for an escape route per turn, nearest first.
MAX_BOMB_SPOTS_CHECKED = 12

Cell2 = tuple[int, int]


@dataclass(frozen=True)
class PlannedBomb:
    """A bomb with the action it really explodes on, after chain reactions.

    ``burn_cells`` are the blast cells within ``burn`` tiles of the bomb's
    cell: they burn ``burn`` actions longer than the others.
    """

    x: int
    y: int
    explode_at: int
    blocks_from: int
    blast: frozenset
    burn: int = 0
    burn_cells: frozenset = frozenset()


def _inside(x: int, y: int) -> bool:
    return 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT


def blast_distances(grid, x: int, y: int, bomb_cells, blast_range: int = BLAST_RANGE) -> dict:
    """Cells a bomb at (x, y) burns -> distance from the bomb along its ray.

    Walls stop a ray and do not burn. Crates and other bombs' cells burn and
    stop the ray.
    """
    cells = {(x, y): 0}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, blast_range + 1):
            cx, cy = x + dx * distance, y + dy * distance
            if not _inside(cx, cy) or grid[cy][cx] is Cell.WALL:
                break
            cells[(cx, cy)] = distance
            if grid[cy][cx] is Cell.BLOCK or (cx, cy) in bomb_cells:
                break
    return cells


def blast_cells(grid, x: int, y: int, bomb_cells, blast_range: int = BLAST_RANGE) -> set:
    """Cells a bomb at (x, y) burns, with the engine's rules."""
    return set(blast_distances(grid, x, y, bomb_cells, blast_range))


def plan_bombs(grid, bombs, blast_range: int = BLAST_RANGE) -> list:
    """Predict when each bomb explodes, including chain reactions.

    ``bombs`` holds ``(x, y, explode_at, blocks_from)`` tuples, optionally
    followed by the bomb's own ``range`` and ``burn`` (default
    ``blast_range`` and 0). A bomb whose cell lies in another bomb's blast
    explodes no later than that bomb; each bomb keeps its own blast.
    """
    bombs = list(bombs)
    bomb_cells = {(b[0], b[1]) for b in bombs}
    ranges = [b[4] if len(b) > 4 else blast_range for b in bombs]
    burns = [b[5] if len(b) > 5 else 0 for b in bombs]
    distances = [blast_distances(grid, b[0], b[1], bomb_cells, ranges[i])
                 for i, b in enumerate(bombs)]
    blasts = [frozenset(d) for d in distances]
    times = [b[2] for b in bombs]
    changed = True
    while changed:
        changed = False
        for i in range(len(bombs)):
            for j in range(len(bombs)):
                reaches = (bombs[j][0], bombs[j][1]) in blasts[i]
                if i != j and reaches and times[i] < times[j]:
                    times[j] = times[i]
                    changed = True
    return [
        PlannedBomb(b[0], b[1], times[i], b[3], blasts[i], burns[i],
                    frozenset(c for c, d in distances[i].items() if d <= burns[i]))
        for i, b in enumerate(bombs)
    ]


def danger_times(planned, flame_ticks: int = 1, burning=None) -> dict:
    """Cell -> set of actions (1 = next) on which that cell burns.

    ``burning`` maps cells already burning to their ``ticks_left``. A cell
    in a bomb's ``burn_cells`` burns ``burn`` actions longer. The engine
    keeps the longer timer when blasts overlap, so the union of the
    intervals is exactly when a cell burns.
    """
    danger: dict = {}
    for bomb in planned:
        for cell in bomb.blast:
            ticks = flame_ticks + (bomb.burn if cell in bomb.burn_cells else 0)
            danger.setdefault(cell, set()).update(
                range(bomb.explode_at, bomb.explode_at + ticks))
    for cell, ticks_left in (burning or {}).items():
        if ticks_left > 1:  # a flame on its last tick is out by the next action
            danger.setdefault(cell, set()).update(range(1, ticks_left))
    return danger


def bombs_from_state(state: dict) -> list:
    """The board's bombs as ``(x, y, explode_at, blocks_from, range, burn)`` tuples.

    Without buffs a bomb has no ``range`` or ``burn``: the rules' blast
    range and 0 are used.
    """
    blast_range = state["rules"]["blast_range"]
    return [(b["x"], b["y"], b["countdown"], 1, b.get("range", blast_range), b.get("burn", 0))
            for b in state["bombs"]]


def levels_of(state: dict, character: dict) -> tuple[int, int, int]:
    """``(range, burn, bombs_left)`` of one character entry of ``state``.

    Without buffs: the rules' blast range, burn 0, and one bomb unless the
    character already has a bomb on the board.
    """
    if "bombs_left" in character:
        bombs_left = character["bombs_left"]
    else:
        bombs_left = 0 if any(b["owner"] == character["id"] for b in state["bombs"]) else 1
    return (character.get("range", state["rules"]["blast_range"]),
            character.get("burn", 0), bombs_left)


def steps_to_safety(grid, start: Cell2, planned, first_action: Action,
                    flame_ticks: int = 1, burning=None):
    """Actions needed to reach a cell no known blast will reach, or ``None``.

    ``first_action`` is taken now; ``Action.BOMB`` means staying put (the
    caller adds the new bomb to ``planned``). A move into a wall, crate or
    bomb also leaves the character in place, as in the engine. After that the
    search tries every move and wait each action, dying on any cell that
    burns on that action.
    """
    danger = danger_times(planned, flame_ticks, burning)
    horizon = max((max(times) for times in danger.values() if times), default=0)

    def walkable(cell, t):
        x, y = cell
        if not _inside(x, y) or grid[y][x] is not Cell.EMPTY:
            return False
        return not any(
            (b.x, b.y) == cell and b.blocks_from <= t <= b.explode_at for b in planned
        )

    def safe_forever(cell, t):
        return all(when <= t for when in danger.get(cell, ()))

    first = start
    if first_action in MOVES:
        dx, dy = MOVES[first_action]
        target = (start[0] + dx, start[1] + dy)
        if walkable(target, 1):
            first = target
    if 1 in danger.get(first, ()):
        return None
    if safe_forever(first, 1):
        return 1

    frontier = {first}
    for t in range(2, horizon + 1):
        reached = set()
        for x, y in frontier:
            for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
                cell = (x + dx, y + dy)
                if (dx, dy) != (0, 0) and not walkable(cell, t):
                    continue
                if t in danger.get(cell, ()):
                    continue
                reached.add(cell)
        if not reached:
            return None
        if any(safe_forever(cell, t) for cell in reached):
            return t
        frontier = reached
    return horizon if frontier else None


class RuleBot:
    """Survives, bombs crates and opponents, chases them, and takes the open gate."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers

    def _pick(self, actions):
        """Seeded tie-break that does not depend on set or dict ordering."""
        order = {a: i for i, a in enumerate(tuple(Action))}
        return self._rng.choice(sorted(actions, key=order.__getitem__))

    @staticmethod
    def _opponents(state: dict, me: int) -> list:
        return [
            (c["x"], c["y"])
            for c in state["characters"]
            if c["id"] != me and not c["dead"]
        ]

    @staticmethod
    def _bomb_value(grid, cell: Cell2, bomb_cells, opponents, blast_range: int) -> int:
        burned = blast_cells(grid, cell[0], cell[1], bomb_cells, blast_range)
        crates = sum(1 for x, y in burned if grid[y][x] is Cell.BLOCK)
        hit = sum(1 for o in opponents if o in burned)
        return crates * CRATE_VALUE + hit * OPPONENT_VALUE

    @staticmethod
    def _distances(grid, start: Cell2, bomb_cells, avoid) -> dict:
        """BFS distance and first step from ``start`` over floor cells."""
        result = {start: (0, None)}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            dist, first = result[cell]
            for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                dx, dy = MOVES[action]
                nxt = (cell[0] + dx, cell[1] + dy)
                x, y = nxt
                if nxt in result or not _inside(x, y) or grid[y][x] is not Cell.EMPTY:
                    continue
                if nxt in bomb_cells or nxt in avoid:
                    continue
                result[nxt] = (dist + 1, first or action)
                queue.append(nxt)
        return result

    @staticmethod
    def _possible_bombs(state: dict, me: int) -> list:
        """Bombs opponents could place on this action, as planning tuples.

        An opponent with a bomb left could bomb its cell, with its own range
        and burn. Two opponents on one cell: the longest range and burn.
        """
        occupied = {(b["x"], b["y"]) for b in state["bombs"]}
        worst: dict = {}
        for c in state["characters"]:
            if c["id"] == me or c["dead"]:
                continue
            blast_range, burn, bombs_left = levels_of(state, c)
            cell = (c["x"], c["y"])
            if bombs_left <= 0 or cell in occupied:
                continue
            old = worst.get(cell, (0, 0))
            worst[cell] = (max(old[0], blast_range), max(old[1], burn))
        fuse = state["rules"]["bomb_fuse"]
        return [(x, y, 1 + fuse, 2, *worst[(x, y)]) for x, y in sorted(worst)]

    def _escape_after_bomb(self, grid, cell: Cell2, bombs, rules: dict, burning,
                           my_range: int | None = None, my_burn: int = 0) -> bool:
        """Whether, after bombing ``cell``, some path escapes every bomb in ``bombs``
        and the new one (with chain reactions and burn times)."""
        if my_range is None:
            my_range = rules["blast_range"]
        new_bomb = (cell[0], cell[1], 1 + rules["bomb_fuse"], 2, my_range, my_burn)
        planned = plan_bombs(grid, bombs + [new_bomb], rules["blast_range"])
        return steps_to_safety(grid, cell, planned, Action.BOMB,
                               rules["flame_ticks"], burning) is not None

    # ------------------------------------------------------------------ policy

    def act(self, state: dict, me: int) -> Action:
        character = state["characters"][me]
        if character["dead"] or state["outcome"] is not None:
            return Action.WAIT

        grid = state["grid"]
        here = (character["x"], character["y"])
        rules = state["rules"]
        blast_range, flame_ticks = rules["blast_range"], rules["flame_ticks"]
        burning = {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}
        bombs = bombs_from_state(state)
        bomb_cells = {(b[0], b[1]) for b in bombs}
        planned = plan_bombs(grid, bombs, blast_range)
        danger = danger_times(planned, flame_ticks, burning)

        possible = self._possible_bombs(state, me)
        cautious_plan = plan_bombs(grid, bombs + possible, blast_range)
        steps = {a: steps_to_safety(grid, here, cautious_plan, a, flame_ticks, burning)
                 for a in STEP_ACTIONS}
        safe = [a for a in STEP_ACTIONS if steps[a] is not None]
        if not safe:
            # No action survives the worst case; use the bombs really there.
            steps = {a: steps_to_safety(grid, here, planned, a, flame_ticks, burning)
                     for a in STEP_ACTIONS}
            safe = [a for a in STEP_ACTIONS if steps[a] is not None]
        if not safe:
            return Action.WAIT  # trapped: no action survives

        # 1. Already inside a future blast: take the fastest way out.
        if danger.get(here):
            best = min(steps[a] for a in safe)
            return self._pick([a for a in safe if steps[a] == best])

        def go(first_step):
            if first_step in safe:
                return first_step
            return Action.WAIT if Action.WAIT in safe else self._pick(safe)

        avoid = set(danger)
        reach = self._distances(grid, here, bomb_cells - {here}, avoid)

        # 2. Agent only: walk onto the open gate.
        if me == AGENT_ID and state["gate"]["state"] == "open":
            if GATE_POSITION in reach:
                return go(reach[GATE_POSITION][1])
            route = self._distances(grid, here, bomb_cells - {here}, set())
            if GATE_POSITION in route:
                return go(route[GATE_POSITION][1])

        opponents = self._opponents(state, me)
        my_range, my_burn, bombs_left = levels_of(state, character)
        can_bomb = bombs_left > 0 and here not in bomb_cells

        # 3. Bomb here if it hits something and the bot can still get away
        # from every bomb on the board, its own earlier bombs included.
        if can_bomb and self._bomb_value(grid, here, bomb_cells, opponents, my_range) > 0:
            if self._escape_after_bomb(grid, here, bombs + possible, rules, burning,
                                       my_range, my_burn):
                return Action.BOMB

        # 4. Walk toward the nearest useful bomb spot.
        spots = []
        for cell, (dist, first) in reach.items():
            if cell == here:
                continue
            value = self._bomb_value(grid, cell, bomb_cells, opponents, my_range)
            if value > 0:
                spots.append((dist, -value, cell, first))
        spots.sort()
        for dist, _, cell, first in spots[:MAX_BOMB_SPOTS_CHECKED]:
            if self._escape_after_bomb(grid, cell, [], rules, burning, my_range, my_burn):
                return go(first)

        # 5. Nothing useful in reach.
        return Action.WAIT if Action.WAIT in safe else self._pick(safe)
