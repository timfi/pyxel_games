# Pyxel Games

This is small collection of games written in python using [pyxel](https://github.com/kitao/pyxel).

## Tic-Tac-Toe
![ttt](.images/tic-tac-toe.png)
### Controls
* `WASD` to move the cursor
* `space` to mark selected cell
* `Q` to forfit the match

## Sudoku Solver
![ss](.images/sudoku-solver.png)
### Controls
* `WASD` to move the cursor
* `0, ..., 9` to mark selected cell
* `space` to start/stop the solve
* `UP, DOWN` to set the riddle fill percentage
* `enter` to generate riddle

### Approach
The general apporach used in this solver is a brute-force solve using back-tracking enhanced with a bit of domain knowledge:
1. each cell on the board has a set of entry candidates
2. cells with less entry candidates are preferred
3. entry candidates that are less common are preferred
4. cells with the least common entry candidates are preffered

## Lgame
![lgame](.images/lgame.png)
### Controls
* `WASD` to move the cursor
* `space` to start, end, and terminate the selection
* `Q` to forfit the match

### TODO
* win/loss detection still has some bugs that should be fixed/removed


