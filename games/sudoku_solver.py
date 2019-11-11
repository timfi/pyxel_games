from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, cast, Set, Optional, NewType
from itertools import chain
from collections import defaultdict
from random import random, shuffle

import pyxel

# -------------------------------------------------------
# Types
# -------------------------------------------------------
BoardData = NewType("BoardData", Tuple[int, ...])
BoardLock = NewType("BoardLock", Tuple[bool, ...])
BoardObj = NewType("BoardObj", Tuple[int, ...])
BoardIdxSet = NewType("BoardIdxSet", Tuple[BoardObj, ...])
CellCandidates = NewType("CellCandidates", Tuple[Set[int], ...])


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
    """Representation of a soduko riddle and its state during a solve."""
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
        """Get a new board with a given cell set to a given value.

        This will return the current board if the given arguments
        don't comply with the current riddle. 
        """
        if self.is_locked(idx) or 0 > val or val > 9:
            return self
        data = list(self._data)
        data[idx] = val
        return Board(cast(BoardData, tuple(data)), self._locked)

    def is_locked(self, idx: int):
        """Check if a cell is part of the riddle."""
        return self._locked[idx]

    def freeze(self) -> Board:
        """Make riddle out of all currently filled cells."""
        return Board(self._data, cast(BoardLock, tuple(
            cell != 0
            for cell in self._data
        )))

    def clear(self) -> Board:
        """Get a unfilled copy of the current riddle."""
        return Board(cast(BoardData, tuple(
            cell if locked else 0
            for cell, locked in zip(self._data, self._locked)
        )), self._locked)

    def _iter(self, idx_set: BoardIdxSet) -> Iterator[BoardObj]:
        """Get all board objects definied by a given board index set."""
        yield from (cast(BoardObj, tuple(self._data[idx] for idx in idxs)) for idxs in idx_set)

    @property
    def rows(self) -> Iterator[BoardObj]:
        """Get all rows."""
        yield from self._iter(ROW_IDX)

    @property
    def cols(self) -> Iterator[BoardObj]:
        """Get all columns."""
        yield from self._iter(COL_IDX)

    @property
    def sqrs(self) -> Iterator[BoardObj]:
        """Get all squares."""
        yield from self._iter(SQR_IDX)

    @property
    def mistakes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get rows, columns, and squares that violate the puzzle constraints."""
        return (
            tuple([i for i, row in enumerate(self.rows) if sum(set(row)) != sum(row)]),
            tuple([i for i, col in enumerate(self.cols) if sum(set(col)) != sum(col)]),
            tuple([i for i, sqr in enumerate(self.sqrs) if sum(set(sqr)) != sum(sqr)]),
        )

    @property
    def valid(self) -> bool:
        """Check if the current state of the riddle is valid."""
        return all(len(group) == 0 for group in self.mistakes)

    @property
    def done(self) -> bool:
        """Check if the riddle is completed."""
        return sum(self._data) == 405 and self.valid

    @property
    def to_fill(self):
        """Get the amount of fillable cells that aren't filled."""
        return sself.fillable - self.filled

    @property
    def filled(self):
        """Get the amount of fillable cells that are filled."""
        return sum(
            not locked and cell != 0
            for cell, locked in zip(self._data, self._locked)
        )

    @property
    def fillable(self):
        """Get the amount of fillable cells."""
        return 81  - sum(self._locked)

    def occurences(self, val: int) -> float:
        return sum(val == i for i in self._data) / 9

    @property
    def candidates(self) -> CellCandidates:
        """
        Get the entry candidates for each cell.
        """
        M = set(range(1, 10))
        row_idx = lambda idx: idx // 9
        col_idx = lambda idx: idx % 9
        sqr_idx = lambda idx: (idx % 9) // 3 + 3 * (idx // 27)

        cols = [set(obj) for obj in self.cols]
        rows = [set(obj) for obj in self.rows]
        sqrs = [set(obj) for obj in self.sqrs]

        return cast(CellCandidates, tuple(
            (
                M - (cols[col_idx(i)] | rows[row_idx(i)] | sqrs[sqr_idx(i)])
                if not self.is_locked(i) else
                set()
            )
            for i in range(81)            
        ))


# -------------------------------------------------------
# Solver
# -------------------------------------------------------
@dataclass
class Solver:
    """Sudoko solver

    This is a sudoko solver that makes use of a stack
    of generators to store intermediate execution states.
    This allows the solver to be executed step by step
    and paused after any iteration.
    """
    _running: bool = field(init=False, default=False)
    _stack: List = field(init=False, default_factory=list)
    _checked_boards: List = field(init=False, default_factory=list)

    def start(self, board: Board):
        """Initialize solver with given board."""
        self._stack = [self._next(board)]
        self._checked_boards = []
        self._running = True

    def stop(self):
        """Halt solver."""
        self._running = False

    @property
    def running(self):
        """Check if the solver is currently running"""
        return self._running
    
    def next(self) -> Optional[Board]:
        """Execute the next step of the solver."""
        if not self._stack:
            return None

        try:
            follow, board = next(self._stack[-1])
        except StopIteration:
            self._stack.pop(-1)
            return self.next()

        if board.done:
            self._running = False
        elif follow:
            self._stack.append(self._next(board))
        else:
            self._stack.pop(-1)
        return cast(Board, board)

    def _next(self, board: Board) -> Iterator[Tuple[bool, Board]]:
        """Get prefered child-states for given board."""
        idx, candidates = self._get_preffered_cell(board)

        if not candidates:
            yield False, board

        for candidate in sorted(candidates, key=board.occurences):
            _board = board.set(idx, candidate)
            if _board not in self._checked_boards:
                yield True, _board
                self._checked_boards.append(_board)
        yield False, board

    def _get_preffered_cell(self, board: Board) -> Tuple[int, Set[int]]:
        """Get prefered cell and its candidates for a given board."""
        all_candidates = board.candidates
        idx = min(
            (idx for idx, cell in enumerate(board) if cell == 0),
            key=lambda idx: (
                len(candidates := all_candidates[idx]),
                sum(board.occurences(c) for c in candidates)
            )
        )
        return idx, all_candidates[idx]


# -------------------------------------------------------
# Utils
# -------------------------------------------------------
def generate_riddle(fill: float = 0.5) -> Board:
    """Generate a sudoko riddle.
    
    This generates a new riddle by filling an empty board using
    simple bruteforce-backtracking and then randomly removing
    some cells based on the given fill percentage.
    """
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
            border_width=SCALE, border_color=pyxel.DEFAULT_PALETTE[15]
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
            f"iter: {self._iterations: >8d}",
            5
        )
        pyxel.text(
            BOARD_WIDTH + 5, pyxel.FONT_HEIGHT + 4,
            f"fill:    {self.board.filled: >2d}/{self.board.fillable: >2d}",
            5
        )
        pyxel.text(
            BOARD_WIDTH + 5, 2 * pyxel.FONT_HEIGHT + 6,
            f"          {self.board.filled/self.board.fillable: >4.0%}",
            5
        )
        pyxel.text(
            BOARD_WIDTH + 5, 3 * pyxel.FONT_HEIGHT + 10,
            f"infill:  {self._fill: >5.0%}",
            5 + 2 * (not self.solver.running)
        )

    def move_cursor(self, dx: int, dy: int):
        x, y = self.cursor % 9, self.cursor // 9
        x_, y_ = x + dx, y + dy
        if (0 <= x_ < 9 and 0 <= y_ < 9):
            self.cursor = y_ * 9 + x_

    def update(self):
        if self.solver.running:
            board = self.solver.next()
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
                self.board = self.board.clear()

            fill_input = pyxel.btnp(pyxel.KEY_UP) << 1 | pyxel.btnp(pyxel.KEY_DOWN)
            if fill_input == 0b01:
                self._fill = max(0, self._fill - 0.05)
            elif fill_input == 0b10:
                self._fill = min(1, self._fill + 0.05)

if __name__ == '__main__':
    App().run()