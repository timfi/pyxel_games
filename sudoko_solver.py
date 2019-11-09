from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, cast, Set
from itertools import chain
from collections import defaultdict
from random import shuffle, random

import pyxel

from _tools import movement

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
# Solvers
# -------------------------------------------------------
@dataclass
class SolverBf:
    _running: bool = field(init=False, default=False)
    _stack: List = field(init=False, default_factory=list)

    def start(self, board: Board):
        self._stack = [self._solve(board)]
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
            board, idx = next(self._stack[-1])
        except StopIteration:
            self._stack.pop(-1)
            return self.update()
        if board.done:
            self._running = False
        elif 80 >= idx > 0:
            self._stack.append(self._solve(board, idx))
        else:
            self._stack.pop(-1)
        return board

    @staticmethod
    def _solve(board: Board, idx: int = 0) -> Iterator[Tuple[bool, Board]]:
        if board.is_locked(idx) or board[idx] != 0:
            yield board, idx + 1
        else:
            candidates = list(range(1, 10))
            shuffle(candidates)
            for i in candidates:
                _board = board.set(idx, i)
                if _board.valid:
                    yield _board, idx+1
            yield board, -1


@dataclass
class SolverDyn:
    _initial_board: Board = field(init=False, default_factory=Board)
    _running: bool = field(init=False, default=False)
    _stack: List = field(init=False, default_factory=list)
    _memo: List = field(init=False, default_factory=list)
    _iterations: int = field(init=False, default=0)

    def start(self, board: Board, reset_after: int = -1):
        self._initial_board = board
        self._stack = [self._solve(board)]
        self._reset_after = reset_after
        self._reset_counter = 0
        self._memo = []
        self._running = True
        self._iterations = 0

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running

    @property
    def iterations(self):
        return self._iterations
    
    def update(self) -> Board:
        if not self._stack:
            return None

        try:
            follow, board = next(self._stack[-1])
        except StopIteration:
            self._stack.pop(-1)
            return self.update()

        if not board.valid:
            self._running = False
            return board

        if board.done:
            self._running = False
        elif follow:
            self._stack.append(self._solve(board))
        else:
            self._stack.pop(-1)

        self._iterations += 1
        return board

    def _solve(self, board: Board) -> Iterator[Tuple[Board, int]]:
        groups = defaultdict(list)
        for idx, candidates in enumerate(board.candidates):
            if board[idx] == 0:
                if len(candidates) == 0:
                    yield False, board
                else:
                    groups[len(candidates)].append((idx, candidates))

        for l, group in sorted(groups.items(), key=lambda t: t[0]):
            shuffle(group)
            for idx, candidates in group:
                for candidate in candidates:
                    _board = board.set(idx, candidate)
                    if _board not in self._memo:
                        yield True, _board
                        self._memo.append(_board)
        yield False, board


# -------------------------------------------------------
# Utils
# -------------------------------------------------------
def generate_riddle(fill: float = 0.5) -> Board:
    solver = SolverBf()
    while True:
        solver.start(Board())
        while solver.running:
            board = solver.update()
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
    solver: SolverDyn = field(init=False, default_factory=SolverDyn)

    _fill: float = field(init=False, default=0.25)

    # TODO: change from naive reset to detecting oscillations
    _reset_after_count: int = field(init=False, default=100)
    _reset_cell_fraction: float= field(init=False, default=0.2)

    _reset_counter: int = field(init=False, default=0)
    _reset_cells: float = field(init=False, default=0)

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
            f"step:    {self.solver.iterations}",
            5
        )
        pyxel.text(
            BOARD_WIDTH + 5, pyxel.FONT_HEIGHT + 4,
            f"R count: {self._reset_counter}",
            5
        )
        col = 5 + 2 * (not self.solver.running)
        pyxel.text(
            BOARD_WIDTH + 5, 2 * pyxel.FONT_HEIGHT + 8,
            f"infill:  {self._fill:.0%}",
            col
        )
        pyxel.text(
            BOARD_WIDTH + 5, 3 * pyxel.FONT_HEIGHT + 10,
            f"reset:   {self._reset_after_count}",
            col
        )
        pyxel.text(
            BOARD_WIDTH + 5, 4 * pyxel.FONT_HEIGHT + 12,
            f"R cells: {self._reset_cell_fraction:.0%}",
            col
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
                self.board = board
                if board.to_fill <= self._reset_cells:
                    self._reset_counter += 1
                    if self._reset_counter >= self._reset_after_count:
                        self.solver.start(self.solver._initial_board)
                else:
                    self._reset_counter = 0
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
                self._reset_cells = self.board.to_fill * self._reset_cell_fraction

            if pyxel.btnp(pyxel.KEY_BACKSPACE):
                for i in range(81):
                    if not self.board.is_locked(i):
                        self.board = self.board.set(i, 0)

            fill_input = pyxel.btnp(pyxel.KEY_KP_4) << 1 | pyxel.btnp(pyxel.KEY_KP_1)
            if fill_input == 0b01:
                self._fill = max(0, self._fill - 0.05)
            elif fill_input == 0b10:
                self._fill = min(1, self._fill + 0.05)

            reset_after_input = pyxel.btnp(pyxel.KEY_KP_5) << 1 | pyxel.btnp(pyxel.KEY_KP_2)
            if reset_after_input == 0b01:
                self._reset_after_count = max(0, self._reset_after_count - 1)
            elif reset_after_input == 0b10:
                self._reset_after_count = self._reset_after_count + 1

            reset_after_input = pyxel.btnp(pyxel.KEY_KP_6) << 1 | pyxel.btnp(pyxel.KEY_KP_3)
            if reset_after_input == 0b01:
                self._reset_cell_fraction = max(0, self._reset_cell_fraction - 0.05)
            elif reset_after_input == 0b10:
                self._reset_cell_fraction = min(1, self._reset_cell_fraction + 0.05)

if __name__ == '__main__':
    App().run()