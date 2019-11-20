import pyxel

from boids import App as boids_app
from mazes import App as mazes_app
from sudoku_solver import App as sudoku_solver_app


WIDTH = 128
HEIGHT = 128
SCALE = 3


APPS = {
    "boids": boids_app,
    "mazes": mazes_app,
    "sudoku_solver_app": sudoko_solver_app,
}


class App:
    def run(self):
        pyxel.init(
            WIDTH, HEIGHT,
            scale=SCALE, caption="Start",
            border_width=0
        )
        pyxel.run(self.update, self.draw)

    def update(self):
        ...

    def draw(self):
        pyxel.cls(0)


if __name__ == "__main__":
    App().run()
