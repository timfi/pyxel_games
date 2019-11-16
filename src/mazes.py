from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Iterator, Iterable, Optional
from random import choice

import pyxel

# -------------------------------------------------------
# Types
# -------------------------------------------------------
Maze = Tuple[int, ...]


# -------------------------------------------------------
# Constants
# -------------------------------------------------------
SCALE = 3
BOARD_WIDTH = 32
BOARD_HEIGHT = 32
CELL_SIZE = 6
CELL_COLOR = 15
WALL_SIZE = 1
WALL_COLOR = 5

# Flags
UP     = 1 << 0
LEFT   = 1 << 1
DOWN   = 1 << 2
RIGHT  = 1 << 3
VISTED = 1 << 4

# Calculated
N_CELLS = BOARD_WIDTH * BOARD_HEIGHT
BLOCK_SIZE = CELL_SIZE + WALL_SIZE * 2
WINDOW_WIDTH = BOARD_WIDTH * BLOCK_SIZE
WINDOW_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE

NEIGHBORS = ((0, -1), (-1, 0), (0, 1), (1, 0))


# -------------------------------------------------------
# Maze
# -------------------------------------------------------
@dataclass
class Generator:
    width: int
    height: int
    start_pos: InitVar[Optional[Tuple[int, int]]] = None

    _visited_cells: int = field(init=False, default=0)
    _stack: List[Tuple[int, int]] = field(init=False, default_factory=list)
    _maze: List[int] = field(init=False)

    def __post_init__(self, start_pos: Optional[Tuple[int, int]]):
        x, y = start_pos = start_pos or (0, 0)
        self._stack.append(start_pos)
        self._visited_cells = 1
        self._maze = [0 for _ in range(self.width * self.height)]
        self._maze[y * self.width + x] |= VISTED

    def _get_neighbors(self, x: int, y: int) -> List[int]:
        return [
            (i, dx, dy)
            for i, (dx, dy) in enumerate(NEIGHBORS)
            if (
                0 <= x + dx < self.width and
                0 <= y + dy < self.height and
                self._maze[(y + dy) * self.width + (x + dx)] & VISTED == 0
            )
        ]    

    def step(self) -> Tuple[Maze, Tuple[int, int], bool]:
        if self._visited_cells < self.width * self.height:
            x, y = self._stack[-1]
            neighbors = self._get_neighbors(x, y)
            if neighbors:
                d, dx, dy = choice(neighbors)
                self._maze[y * self.width + x] |= 1 << d
                x_, y_ = x + dx, y + dy
                self._maze[y_ * self.width + x_] |= 1 << ((d + 2) % 4) | VISTED
                self._stack.append((x_, y_))
                self._visited_cells += 1
            else:
                del self._stack[-1]
            return tuple(self._maze), self._stack[-1], False
        else:
            return tuple(self._maze), (0, 0), True


# -------------------------------------------------------
# Application
# -------------------------------------------------------
@dataclass
class App:
    maze: Maze = field(init=False, default=tuple(0 for _ in range(N_CELLS)))
    generator: Optional[Generator] = field(init=False, default=None)
    running: bool = field(init=False, default=False)
    pos: Tuple[int, int] = field(init=False, default=(0, 0))

    def run(self):
        pyxel.init(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            scale=SCALE, caption="Mazes",
            border_width=SCALE, border_color=pyxel.DEFAULT_PALETTE[5],
            fps=100
        )
        pyxel.mouse(True)
        pyxel.run(self.update, self.draw)

    def draw(self):
        pyxel.cls(0)
        for i, cell in enumerate(self.maze):
            x, y = i % BOARD_WIDTH, i // BOARD_WIDTH
            scr_x, scr_y = x * BLOCK_SIZE, y * BLOCK_SIZE
            pyxel.rect(
                scr_x, scr_y,
                BLOCK_SIZE, BLOCK_SIZE,
                WALL_COLOR
            )
            if cell & VISTED:
                pyxel.rect(
                    scr_x + WALL_SIZE, scr_y + WALL_SIZE,
                    CELL_SIZE, CELL_SIZE,
                    CELL_COLOR
                )
                if cell & UP:
                    pyxel.rect(
                        scr_x + WALL_SIZE, scr_y,
                        CELL_SIZE, WALL_SIZE,
                        CELL_COLOR
                    )
                if cell & LEFT:
                    pyxel.rect(
                        scr_x, scr_y + WALL_SIZE,
                        WALL_SIZE, CELL_SIZE,
                        CELL_COLOR
                    )
                if cell & DOWN:
                    pyxel.rect(
                        scr_x + WALL_SIZE, scr_y + WALL_SIZE + CELL_SIZE,
                        CELL_SIZE, WALL_SIZE,
                        CELL_COLOR
                    )
                if cell & RIGHT:
                    pyxel.rect(
                        scr_x + WALL_SIZE + CELL_SIZE, scr_y + WALL_SIZE,
                        WALL_SIZE, CELL_SIZE,
                        CELL_COLOR
                    )

        x, y = self.pos
        pyxel.rectb(
            x * BLOCK_SIZE + WALL_SIZE, y * BLOCK_SIZE + WALL_SIZE,
            CELL_SIZE, CELL_SIZE,
            2 if self.running else 1
        )

    def update(self):
        if pyxel.btnp(pyxel.KEY_SPACE) or pyxel.btnp(pyxel.MOUSE_LEFT_BUTTON):
            self.running = not self.running
            if self.running and self.generator is None:
                self.generator = Generator(BOARD_WIDTH, BOARD_HEIGHT, self.pos)
        if self.running:
            next_maze, pos, done = self.generator.step()
            if done:
                self.running = False
                self.generator = None
            self.maze = next_maze
            self.pos = pos
        else:
            scr_x, scr_y = pyxel.mouse_x, pyxel.mouse_y
            self.pos = scr_x // BLOCK_SIZE, scr_y // BLOCK_SIZE

if __name__ == '__main__':
    App().run()