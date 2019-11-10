from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, cast, Set, Optional
from itertools import chain
from collections import defaultdict
from random import random, shuffle

import pyxel

# -------------------------------------------------------
# Types
# -------------------------------------------------------
BoardData = Tuple[(int,)*81]
BoardLock = Tuple[(bool,)*81]
BoardObj = Tuple[(int,)*9]
BoardIdxSet = Tuple[(BoardObj,)*9]


# -------------------------------------------------------
# Constants
# -------------------------------------------------------
BLOCK_SIZE = 8
BOARD_WIDTH = BLOCK_SIZE * 9 + 8
BOARD_HEIGHT = BLOCK_SIZE * 9 + 8
WINDOW_WIDTH = BOARD_WIDTH + 64
WINDOW_HEIGHT = BOARD_HEIGHT
SCALE = 5
MISTAKE_TIMER = 7

GEN_EMPTY_BOARD = lambda: cast(BoardData, tuple([0     for _ in range(81)]))
GEN_EMPTY_LOCKS = lambda: cast(BoardLock, tuple([False for _ in range(81)]))

# raw data
_ROW_IDX = (range(i*9, (i+1)*9) for i in range(9))
_COL_IDX = (range(i, 81, 9) for i in range(9))
_SQR_IDX = (
    chain(
        range(
            18 * (i//3) + 0 * 9 + i    * 3,
            18 * (i//3) + 0 * 9 + (i+1)* 3
        ),
        range(
            18 * (i//3) + 1 * 9 + i    * 3,
            18 * (i//3) + 1 * 9 + (i+1)* 3
        ),
        range(
            18 * (i//3) + 2 * 9 + i    * 3,
            18 * (i//3) + 2 * 9 + (i+1)* 3
        )
    )
    for i in range(9)
)

# typed
ROW_IDX = cast(BoardIdxSet, tuple(cast(BoardObj, tuple(idxs)) for idxs in _ROW_IDX))
COL_IDX = cast(BoardIdxSet, tuple(cast(BoardObj, tuple(idxs)) for idxs in _COL_IDX))
SQR_IDX = cast(BoardIdxSet, tuple(cast(BoardObj, tuple(idxs)) for idxs in _SQR_IDX))


# -------------------------------------------------------
# Board data
# -------------------------------------------------------
@dataclass(frozen=True)
class Board:
    _data: BoardData = field(default_factory=GEN_EMPTY_BOARD)
    _locked: BoardLock = field(default_factory=GEN_EMPTY_LOCKS)

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __getitem__(self, idx: int) -> int:
        if 0 <= idx < 81:
            return self._data[idx]
        else:
            raise KeyError(f"Index out of scope. {idx}")

    def set(self, idx: int, val: int) -> Board:
        data = list(self._data)
        data[idx] = val
        return Board(tuple(data), self._locked)

    def is_locked(self, idx: int):
        return self._locked[idx]

    def lock(self, idx: int) -> Board:
        locked = list(self._locked)
        locked[idx] = True
        return Board(self._data, tuple(locked))

    def unlock(self, idx: int) -> Board:
        locked = list(self._locked)
        locked[idx] = False
        return Board(self._data, tuple(locked))

    def freeze(self) -> Board:
        return Board(self._data, tuple([
            cell != 0
            for cell in self._data
        ]))

    def _iter(self, idx_set: BoardIdxSet) -> Iterator[BoardObj]:
        yield from (cast(BoardObj, tuple(self._data[idx] for idx in idxs)) for idxs in idx_set)

    @property
    def rows(self) -> Iterator[BoardObj]:
        yield from self._iter(ROW_IDX)

    @property
    def cols(self) -> Iterator[BoardObj]:
        yield from self._iter(COL_IDX)

    @property
    def sqrs(self) -> Iterator[BoardObj]:
        yield from self._iter(SQR_IDX)

    @property
    def mistakes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return (
            tuple([i for i, row in enumerate(self.rows) if sum(set(row)) != sum(row)]),
            tuple([i for i, col in enumerate(self.cols) if sum(set(col)) != sum(col)]),
            tuple([i for i, sqr in enumerate(self.sqrs) if sum(set(sqr)) != sum(sqr)]),
        )

    @property
    def valid(self) -> bool:
        return all(len(group) == 0 for group in self.mistakes)

    @property
    def done(self) -> bool:
        return sum(self._data) == 405 and self.valid

    @property
    def to_fill(self):
        return sum(
            not locked and cell == 0
            for cell, locked in zip(self._data, self._locked)
        )

    def diff(self, other: Board) -> int:
        return sum(
            abs(a - b)
            for a, b in zip(self, other)
        )

    @property
    def candidates(self) -> Tuple[(Set[int],)*81]:
        M = set(range(1, 10))
        row_idx = lambda idx: idx // 9
        col_idx = lambda idx: idx % 9
        sqr_idx = lambda idx: (idx % 9) // 3 + 3 * (idx // 27)

        cols = list(map(set, self.cols))
        rows = list(map(set, self.rows))
        sqrs = list(map(set, self.sqrs))

        return tuple(
            M - (cols[col_idx(i)] | rows[row_idx(i)] | sqrs[sqr_idx(i)])
            if not self.is_locked(i) else
            set()
            for i in range(81)            
        )


# -------------------------------------------------------
# Solver
# -------------------------------------------------------
@dataclass
class Solver:
    _running: bool = field(init=False, default=False)
    _stack: List = field(init=False, default_factory=list)
    _checked_boards: List = field(init=False, default_factory=list)

    def start(self, board: Board, reset_after: int = -1):
        self._stack = [self._solve(board)]
        self._checked_boards = []
        self._running = True

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running
    
    def update(self) -> Board:
        if not self._stack:
            return None

        try:
            follow, board = next(self._stack[-1])
        except StopIteration:
            self._stack.pop(-1)
            return self.update()

        if board.done:
            self._running = False
        elif follow:
            self._stack.append(self._solve(board))
        else:
            self._stack.pop(-1)
        return board

    def _solve(self, board: Board) -> Iterator[Tuple[Board, int]]:
        idx = float('inf')
        candidates = set(range(1, 10))
        importance = float('-inf')

        for _idx, _candidates in enumerate(board.candidates):
            if board[_idx] == 0:
                if len(_candidates) == 0:
                    yield False, board
                elif (
                    len(_candidates) < len(candidates) and
                    importance < (_importance := -sum(c == i for c in _candidates for i in board))
                ):
                    candidates = _candidates
                    idx = _idx
                    importance = _importance

        for candidate in sorted(candidates, key=lambda c: -sum(c == i for i in board)):
            _board = board.set(idx, candidate)
            if _board not in self._checked_boards:
                yield True, _board
                self._checked_boards.append(_board)
        yield False, board


# -------------------------------------------------------
# Utils
# -------------------------------------------------------
def generate_riddle(fill: float = 0.5) -> Board:
    def _fill(board: Board, idx: int = 0) -> Optional[Board]:
        if idx >= 80:
            return board
        if board.is_locked(idx) or board[idx] != 0:
            return _fill(board, idx + 1)
        else:
            candidates = list(range(1, 10))
            shuffle(candidates)
            for i in candidates:
                _board = board.set(idx, i)
                if _board.valid:
                    _new_board = _fill(_board, idx+1)
                    if _new_board is not None:
                        return _new_board
            return None

    while True:
        board = _fill(Board())
        if board is not None:
            for i in range(81):
                if random() >= fill:
                    board = board.set(i, 0)
            return board.freeze()


# -------------------------------------------------------
# Application
# -------------------------------------------------------
@dataclass
class App:
    board: Board = field(init=False, default_factory=Board)
    cursor: int = field(init=False, default=0)
    mistake: int = field(init=False, default=0)
    mistake_location: int = field(init=False, default=0)
    solver: Solver = field(init=False, default_factory=Solver)

    _fill: float = field(init=False, default=0.25)
    _iterations: int = field(init=False, default=0)

    def run(self):
        pyxel.init(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            scale=SCALE, caption="Sudoko Solver",
            border_width=0
        )
        pyxel.run(self.update, self.draw)

    def draw(self):
        pyxel.cls(0)
        self.draw_board()
        self.draw_grid()
        self.draw_mistakes()
        if not self.solver.running:
            self.draw_cursor()
        self.draw_ui()

    def draw_mistakes(self):
        rows, cols, sqrs = self.board.mistakes
        for row in rows:
            pyxel.rectb(
                0, row * BLOCK_SIZE + row,
                BOARD_WIDTH, BLOCK_SIZE,
                8
            )
        for col in cols:
            pyxel.rectb(
                col * BLOCK_SIZE + col, 0,
                BLOCK_SIZE, BOARD_HEIGHT,
                8
            )
        for sqr in sqrs:
            i = 18 * (sqr // 3) + sqr * 3
            x, y = i % 9, i // 9
            pyxel.rectb(
                x * BLOCK_SIZE + x, y * BLOCK_SIZE + y,
                2 + 3 * BLOCK_SIZE, 2 + 3 * BLOCK_SIZE,
                8
            )

    def draw_board(self):
        for i, cell in enumerate(self.board):
            if cell > 0:
                x, y = i % 9, i // 9
                pyxel.text(
                    x * BLOCK_SIZE + x + BLOCK_SIZE // 2 - pyxel.FONT_WIDTH // 2,
                    y * BLOCK_SIZE + y + BLOCK_SIZE // 2 - pyxel.FONT_HEIGHT // 2,
                    str(cell), 6 if not self.board.is_locked(i) else 5
                )

    def draw_grid(self):
        for col in range(8):
            col_ = col + 1
            pyxel.line(
                col_ * BLOCK_SIZE + col, 0,
                col_ * BLOCK_SIZE + col, BOARD_HEIGHT,
                5
            )

        for row in range(8):
            row_ = row + 1
            pyxel.line(
                0, row_ * BLOCK_SIZE + row,
                BOARD_WIDTH, row_ * BLOCK_SIZE + row,
                15 if row_ % 3 == 0 else 5
            )

        pyxel.line(3 * BLOCK_SIZE + 2, 0, 3 * BLOCK_SIZE + 2, BOARD_HEIGHT, 15)
        pyxel.line(6 * BLOCK_SIZE + 5, 0, 6 * BLOCK_SIZE + 5, BOARD_HEIGHT, 15)
        pyxel.line(BOARD_WIDTH, 0, BOARD_WIDTH, BOARD_HEIGHT, 15)

    def draw_cursor(self):
        x, y = self.cursor % 9, self.cursor // 9
        pyxel.rectb(
            x * BLOCK_SIZE + x, y * BLOCK_SIZE + y,
            BLOCK_SIZE, BLOCK_SIZE, 15
        )

    def draw_ui(self):
        pyxel.text(
            BOARD_WIDTH + 5, 2,
            f"step:    {self._iterations}",
            5
        )
        pyxel.text(
            BOARD_WIDTH + 5, 1 * pyxel.FONT_HEIGHT + 6,
            f"infill:  {self._fill:.0%}",
            5 + 2 * (not self.solver.running)
        )

    def move_cursor(self, dx: int, dy: int):
        x, y = self.cursor % 9, self.cursor // 9
        x_, y_ = x + dx, y + dy
        if (0 <= x_ < 9 and 0 <= y_ < 9):
            self.cursor = y_ * 9 + x_

    def update(self):
        if self.solver.running:
            board = self.solver.update()
            if pyxel.btnp(pyxel.KEY_SPACE) or board is None:
                if board is None:
                    print("Failed to find solution...")
                self.solver.stop()
            else:
                self._iterations += 1
                self.board = board
        else:
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

            if not self.board.is_locked(self.cursor):
                inputs = [
                    pyxel.btnp(getattr(pyxel, f"KEY_{i}"))
                    for i in range(10)
                ]
                if sum(inputs) == 1:
                    for val, pressed in enumerate(inputs):
                        if pressed:
                            self.board = self.board.set(self.cursor, val)
                            break

            if pyxel.btnp(pyxel.KEY_ENTER):
                self.board = generate_riddle(self._fill)

            if pyxel.btnp(pyxel.KEY_SPACE):
                self.solver.start(self.board)
                self._iterations = 0

            if pyxel.btnp(pyxel.KEY_BACKSPACE):
                for i in range(81):
                    if not self.board.is_locked(i):
                        self.board = self.board.set(i, 0)

            fill_input = pyxel.btnp(pyxel.KEY_UP) << 1 | pyxel.btnp(pyxel.KEY_DOWN)
            if fill_input == 0b01:
                self._fill = max(0, self._fill - 0.05)
            elif fill_input == 0b10:
                self._fill = min(1, self._fill + 0.05)

if __name__ == '__main__':
    App().run()