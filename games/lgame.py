from __future__ import annotations
from typing import List, ClassVar, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from copy import copy
from functools import reduce

import pyxel

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
BLOCK_SIZE = 8
WIDTH = BLOCK_SIZE * 4
HEIGHT = BLOCK_SIZE * 5
SCALE = 6
MISTAKE_TIMER = 7

INITIAL_BOARD = [
    3, 1, 1, 0,
    0, 2, 1, 0,
    0, 2, 1, 0,
    0, 2, 2, 3,
]

LEGAL_SHAPES = (
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(1, 0), (1, 1), (1, 2), (0, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(1, 0), (1, 1), (1, 2), (0, 0)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 1), (2, 1), (2, 0)},
    {(0, 0), (1, 0), (2, 0), (0, 1)},
    {(0, 1), (1, 1), (2, 1), (0, 0)},
)


# -------------------------------------------------------
# Basic datatypes
# -------------------------------------------------------
class Point(NamedTuple):
    x: int
    y: int

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __abs__(self) -> Point:
        return Point(*abs(*self))


LEGAL_SHAPES = tuple(
    {Point(*p) for p in points}
    for points in LEGAL_SHAPES
)


class Shape(NamedTuple):
    points: Set[Tuple[int, int]]
    pos: Tuple[int, int]

    @staticmethod
    def extract(points: List[Point]) -> Shape:
        min_pos = Point(
            min(p.x for p in points),
            min(p.y for p in points),
        )
        return Shape(
            {p - min_pos for p in points},
            min_pos,
        )

    def reproject(self, pos: Point) -> Shape:
        diff = self.pos - pos
        return Shape(
            {p - diff for p in self.points},
            pos,
        )


# -------------------------------------------------------
# Gamedata container
# -------------------------------------------------------
@dataclass
class Data:
    board: List[int] = field(init=False, default_factory=lambda: copy(INITIAL_BOARD))
    player: int = field(init=False, default=1)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            x, y = key
            return self.board[y * 4 + x]
        else:
            return self.board[key]

    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            x, y = key
            self.board[y * 4 + x] = val
        else:
            self.board[key] = val

    def extract_shape(self, *targets: int) -> Tuple[Set[Tuple[int, int]], Tuple[int, int]]:
        return Shape.extract([
            Point(i % 4, i // 4) for i, cell in enumerate(self.board) if cell in targets
        ])

    @property
    def next_player(self) -> int:
        return (self.player) % 2 + 1

    def detect_loss(self) -> bool:
        # TODO: this is still not 100% correct, don't know why though...
        current_shape = self.extract_shape(self.player)
        total_shape   = self.extract_shape(0, self.player)
        for i in range(16):
            local_origin = Point(i % 4, i // 4)
            _total_shape = total_shape.reproject(local_origin)
            for shape in LEGAL_SHAPES:
                if _total_shape.points.issuperset(shape):
                    return (current_shape.pos != local_origin or current_shape.shape != shape)
        return True


# -------------------------------------------------------
# Application
# -------------------------------------------------------
@dataclass
class App:
    data: Data = field(init=False, default_factory=Data)
    _state: int = field(init=False, default=0)
    _states: List[State] = field(init=False)
    _winner: int = field(init=False, default=0)

    def __post_init__(self):
        self._states = (LState(self), NState(self))
        pyxel.init(WIDTH, HEIGHT, scale=SCALE, border_width=0, caption="L GAME")

    def reset(self):
        self.data = Data()
        self._winner = 0
        self._state = 0
        for state in self._states:
            state.init()

    @property
    def state(self):
        return self._states[self._state]

    def next_state(self):
        self._state = (self._state + 1) % len(self._states)
        if self._state == 0:
            self.data.player = self.data.next_player
            if self.data.detect_loss():
                self.forfit()
        self.state.init()

    def forfit(self):
        self._winner = self.data.next_player

    def draw(self):
        pyxel.cls(0)
        if self.game_over:
            x = (WIDTH // 2) - (pyxel.FONT_WIDTH * 3)
            y1 = (HEIGHT // 2) - pyxel.FONT_HEIGHT - 3
            y2 = y1 + pyxel.FONT_HEIGHT + 3
            pyxel.rectb(0, 0, WIDTH, HEIGHT, 6)
            pyxel.text(x+1, y1+1, "WINNER", 5)
            pyxel.text(x,   y1,   "WINNER", 7)
            if pyxel.frame_count % 15 >= 5:
                pyxel.text(x+1, y2+1, f"~ P{self._winner} ~", self._winner)
                pyxel.text(x,   y2,   f"~ P{self._winner} ~", [12, 14][self._winner - 1])
        else:
            self.state.draw()

    def update(self):
        if not self.game_over:
            self.state.update()
        elif pyxel.btnp(pyxel.KEY_SPACE):
            self.reset()

    @property
    def game_over(self):
        return self._winner in [1, 2]

    def run(self):
        self.state.init()
        pyxel.run(self.update, self.draw)


# -------------------------------------------------------
# Game states
# -------------------------------------------------------
@dataclass
class State:
    label: ClassVar[str] = " "

    app: App
    cursor: Tuple[int, int] = field(init=False, default=(0, 0))
    _moved: bool = field(init=False, default=False)
    _selection_active: bool = field(init=False, default=False)
    _selection: List[Point] = field(init=False, default_factory=list)
    _mistake: int = field(init=False, default=0)

    def init(self):
        self.cursor = Point(0, 0)
        self._moved = False
        self._selection_active = False
        self._selection = []

    def draw(self):
        self.draw_ui()
        self.draw_field()
        if self._selection_active:
            self.draw_selection()
        self.draw_cursor()

    def draw_ui(self):
        y_offset = BLOCK_SIZE // 2 - pyxel.FONT_HEIGHT // 2
        pyxel.text(
            0, y_offset,
            f" P{self.app.data.player}",
            self.app.data.player
        )
        pyxel.text(
            WIDTH - pyxel.FONT_WIDTH * 2, y_offset,
            self.label, 7
        )
        x_center = WIDTH // 2 - pyxel.FONT_WIDTH // 2
        if self._mistake > 0:
            pyxel.text(x_center, y_offset, "!", 8)
            self._mistake -= 1

    def draw_field(self):
        for i, cell in enumerate(self.app.data.board):
            x, y = i % 4, i // 4
            pyxel.rect(x * BLOCK_SIZE, (y + 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, cell)

    def draw_selection(self):
        for p in self._selection:
            pyxel.rectb(p.x * BLOCK_SIZE, (p.y + 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 8)

    def draw_cursor(self):
        pyxel.rectb(self.cursor.x * BLOCK_SIZE, (self.cursor.y + 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 9 if self._selection_active else 15)

    def signal_mistake(self):
        self._mistake = MISTAKE_TIMER

    def update(self):
        if pyxel.btnp(pyxel.KEY_Q):
            self.app.forfit()
            return
        movement = (
            (pyxel.btnp(pyxel.KEY_W) << 0) |
            (pyxel.btnp(pyxel.KEY_A) << 1) |
            (pyxel.btnp(pyxel.KEY_S) << 2) |
            (pyxel.btnp(pyxel.KEY_D) << 3)
        )
        if movement == 0b0001:
            self.move_cursor(0, -1)
        elif movement == 0b0010:
            self.move_cursor(-1, 0)
        elif movement == 0b0100:
            self.move_cursor(0, 1)
        elif movement == 0b1000:
            self.move_cursor(1, 0)
        if pyxel.btnp(pyxel.KEY_SPACE):
            if self._selection_active:
                if self.deselect():
                    self._selection_active = False
                else:
                    self.signal_mistake()
            else:
                if self.select():
                    self._selection_active = True
                else:
                    self.signal_mistake()

    def move_cursor(self, dx: int, dy: int):
        new_cursor = self.cursor + Point(dx, dy)
        if self._can_move(*new_cursor):
            self.cursor = new_cursor
            if self._selection_active:
                self.moved()
        else:
            self.signal_mistake()

    def _can_move(self, x: int, y: int) -> bool:
        return (0 <= x <= 3 and 0 <= y <= 3) and (
            not self._selection_active or self.can_move(Point(x, y))
        )

    def can_move(self, point: Point) -> bool:
        return True

    def moved(self):
        ...

    def select(self) -> bool:
        ...

    def deselect(self) -> bool:
        ...

    @property
    def selected(self) -> int:
        return self.app.data[self.cursor]


@dataclass
class LState(State):
    label = "L"

    _current_shape: Shape = field(init=False)

    def init(self):
        super().init()
        self._current_shape = self.app.data.extract_shape(self.app.data.player)

    def can_move(self, point: Point) -> bool:
        return self.app.data[point] in [0, self.app.data.player] and (
            point in self._selection
            or len(self._selection) < 3
            or len(self._selection) == 3
            and (local_shape := Shape.extract(self._selection + [point])).points in LEGAL_SHAPES
            and local_shape != self._current_shape
        )

    def moved(self):
        if self.cursor in self._selection:
            del self._selection[self._selection.index(self.cursor) :]
        self._selection.append(self.cursor)

    def select(self) -> bool:
        init_select = self.app.data[self.cursor] in [0, self.app.data.player]
        if init_select:
            self._selection = [self.cursor]
        return init_select

    def deselect(self) -> bool:
        if len(self._selection) == 4:
            old_shape, (old_x, old_y) = self._current_shape
            for p in self._current_shape.points:
                self.app.data[p + self._current_shape.pos] = 0
            for p in self._selection:
                self.app.data[p] = self.app.data.player
            self.app.next_state()
        return True


class NState(State):
    label = "N"

    def select(self) -> bool:
        init_select = self.selected == 3
        if init_select:
            self._selection.append(self.cursor)
        return init_select

    def deselect(self) -> bool:
        done = self.selected == 0 or self.cursor in self._selection
        if done:
            self.app.data[self._selection[0]] = 0
            self.app.data[self.cursor] = 3
            self.app.next_state()
        return done


if __name__ == "__main__":
    App().run()
