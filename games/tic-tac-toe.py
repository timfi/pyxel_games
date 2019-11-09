from __future__ import annotations
from typing import List, ClassVar, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from copy import copy
from itertools import chain

import pyxel

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
BLOCK_SIZE = 10
WIDTH = BLOCK_SIZE * 3
HEIGHT = BLOCK_SIZE * 3
SCALE = 5
MISTAKE_TIMER = 7


# -------------------------------------------------------
# Application
# -------------------------------------------------------
@dataclass
class App:
    board: List[int] = field(init=False)
    player: int = field(init=False)
    winner: int = field(init=False)
    cursor: int = field(init=False)
    mistake: int = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.board = [0 for _ in range(9)]
        self.player = 1
        self.winner = 0
        self.cursor = 4
        self.mistake = 0

    def draw(self):
        pyxel.cls(0)
        if self.game_over:
            self.draw_game_over()
        else:
            self.draw_field()
            self.draw_cursor()

    def draw_game_over(self):
            x = (WIDTH // 2) - (pyxel.FONT_WIDTH * 3)
            y1 = (HEIGHT // 2) - pyxel.FONT_HEIGHT - 3
            y2 = y1 + pyxel.FONT_HEIGHT + 3
            pyxel.rectb(0, 0, WIDTH, HEIGHT, 6)
            pyxel.text(x+1, y1+1, "WINNER", 5)
            pyxel.text(x,   y1,   "WINNER", 7)
            if pyxel.frame_count % 15 >= 5:
                pyxel.text(x+1, y2+1, f"~ P{self.winner} ~", self.winner)
                pyxel.text(x,   y2,   f"~ P{self.winner} ~", [12, 14][self.winner - 1])

    def draw_field(self):
        for i, cell in enumerate(self.board):
            x, y = i % 3, i // 3
            if cell == 1:
                pyxel.rect(
                    x * BLOCK_SIZE , y * BLOCK_SIZE, 
                    BLOCK_SIZE, BLOCK_SIZE,
                    1
                )
                pyxel.line(
                    (x + 0) * BLOCK_SIZE,     (y + 0) * BLOCK_SIZE,
                    (x + 1) * BLOCK_SIZE - 1, (y + 1) * BLOCK_SIZE - 1,
                    12
                )
                pyxel.line(
                    (x + 0) * BLOCK_SIZE,     (y + 1) * BLOCK_SIZE - 1,
                    (x + 1) * BLOCK_SIZE - 1, (y + 0) * BLOCK_SIZE,
                    12
                )
            elif cell == 2:
                pyxel.rect(
                    x * BLOCK_SIZE , y * BLOCK_SIZE, 
                    BLOCK_SIZE, BLOCK_SIZE,
                    2
                )
                pyxel.rectb(
                    x * BLOCK_SIZE , y * BLOCK_SIZE, 
                    BLOCK_SIZE, BLOCK_SIZE,
                    14
                )

    def draw_cursor(self):
        x, y = self.cursor % 3, self.cursor // 3
        if self.mistake > 0:
            col = 8
            self.mistake -= 1
        else:
            col = 15
        pyxel.rectb(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, col)

    def update(self):
        if not self.game_over:
            if pyxel.btnp(pyxel.KEY_Q):
                self.winner = (self.player) % 2 + 1
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
                if self.board[self.cursor] == 0:
                    self.board[self.cursor] = self.player
                    if self.detect_win():
                        self.winner = self.player
                    else:
                        self.player = (self.player) % 2 + 1
                else:
                    self.signal_mistake()
        elif pyxel.btnp(pyxel.KEY_SPACE):
            self.reset()

    def move_cursor(self, dx: int, dy: int):
        x, y = self.cursor % 3, self.cursor // 3
        x_, y_ = x + dx, y + dy
        if (0 <= x_ < 3 and 0 <= y_ < 3):
            self.cursor = y_ * 3 + x_
        else:
            self.signal_mistake()

    def signal_mistake(self):
        self.mistake = MISTAKE_TIMER

    def detect_win(self) -> bool:
        cols = (
            self.board[::3],
            self.board[1::3],
            self.board[2::3]
        )
        rows = (
            self.board[:3],
            self.board[3:6],
            self.board[6:]
        )
        dias = [
            (self.board[0], self.board[4], self.board[8]),
            (self.board[2], self.board[4], self.board[6])
        ]
        return any(
            all(cell == self.player for cell in shape)
            for shape in chain(cols, rows, dias)
        )

    @property
    def game_over(self):
        return self.winner in [1, 2]

    def run(self):
        pyxel.init(WIDTH, HEIGHT, scale=SCALE, border_width=0, caption="T.T.T.")
        pyxel.run(self.update, self.draw)


if __name__ == "__main__":
    App().run()
