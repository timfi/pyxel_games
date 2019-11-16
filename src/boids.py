from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt, acos, radians, cos, sin
from random import randrange, random
from typing import List, Union, NamedTuple, Tuple

import pyxel

Number = Union[int, float]


WIDTH = 256
HEIGHT = 256
SCALE = 3

N_BOIDS = 100
BOID_SPEED = 2
VIEW_RADIUS = 20

GENERAL_INFLUENCE = 0.75
SEPARATION_WEIGHT = 1
ALIGNMENT_WEIGHT = 1
COHESION_WEIGHT = 1


@dataclass
class Vector2D:
    x: Number
    y: Number

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(
            self.x + other.x,
            self.y + other.y
        )

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, other: Union[float, int]) -> Vector2D:
        return Vector2D(
            self.x * other,
            self.y * other
        )

    def __truediv__(self, other: Union[float, int]) -> Vector2D:
        return Vector2D(
            self.x / other,
            self.y / other
        )

    def __matmul__(self, other: Vector2D) -> Vector2D:
        return self.x * other.x + self.y * other.y

    def __round__(self, ndigits: int = 1) -> Vecotr2D:
        return Vector2D(
            round(self.x, ndigits=ndigits),
            round(self.y, ndigits=ndigits)
        )

    def __mod__(self, limits: Tuple[int, int]) -> Vector2D:
        return Vector2D(
            self.x % limits[0],
            self.y % limits[1],
        )

    @property
    def orth(self) -> Vector2D:
        return Vector2D(self.y, -self.x)

    @property
    def mag(self) -> float:
        return sqrt(self.x ** 2 + self.y ** 2)

    @property
    def norm(self) -> Vector2D:
        return self / self.mag

    @property
    def as_ituple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def with_mag(self, mag: Number) -> Vector2D:
        return self.norm * mag


@dataclass
class Boid:
    pos: Vector2D
    vel: Vector2D

    @classmethod
    def random(cls, width: int, height: int) -> Boid:
        return Boid(
            Vector2D(randrange(0, width), randrange(0, height)),
            Vector2D(random() - 0.5, random() - 0.5).norm
        )

    def draw(self):
        vec_o1 = round(self.pos + self.vel.orth.norm).as_ituple
        vec_o2 = round(self.pos - self.vel.orth.norm).as_ituple

        pyxel.line(*vec_o1, *vec_o2, 5)

        vec_l1 = round(self.pos + self.vel.norm * 2).as_ituple
        vec_l2 = round(self.pos - self.vel.norm * 2).as_ituple

        pyxel.line(*vec_l1, *vec_l2, 6)
        pyxel.pix(*vec_l1, 9)


    def can_see(self, other: Boid) -> bool:
        return (
            abs(((self.pos - other.pos)).mag) <= VIEW_RADIUS
        )

    def update(self, others: List[Boid]):
        neighbors = [
            neighbor
            for j, neighbor in enumerate(others)
            if self.can_see(neighbor)
        ]

        acc = Vector2D(0, 0)
        if neighbors:
            # 1) Seperation
            acc -= sum((neighbor.pos - self.pos for neighbor in neighbors if (neighbor.pos - self.pos).mag < VIEW_RADIUS/2), Vector2D(0, 0)) * SEPARATION_WEIGHT

            # 2) Alignment
            acc += (average([neighbor.pos for neighbor in neighbors]) - self.pos) * ALIGNMENT_WEIGHT / 10

            # 3) Cohesion
            acc += (average([neighbor.vel for neighbor in neighbors]) - self.vel) * COHESION_WEIGHT / 8

        if acc.mag > 0:
            self.vel += acc.norm * GENERAL_INFLUENCE
            if self.vel.mag > BOID_SPEED:
                self.vel = self.vel.norm * BOID_SPEED
            elif self.vel.mag < BOID_SPEED * 0.5:
                self.vel = self.vel.norm * 0.5 * BOID_SPEED
        self.pos = (self.pos + self.vel) % (WIDTH, HEIGHT)


def average(l: List[Vector2D]) -> Vector2D:
    return sum(l, Vector2D(0, 0)) / len(l) if len(l) > 0 else Vector2D(0, 0)


@dataclass
class App:
    boids: List[Boid] = field(init=False, default_factory=list)
    _executor: ProcessPoolExecutor = field(init=False)
    _futures: Dict[Future, int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.boids = [Boid.random(WIDTH, HEIGHT) for _ in range(N_BOIDS)]

    def run(self):
        pyxel.init(
            WIDTH, HEIGHT,
            caption="Boids", scale=SCALE,
            border_width=0
        )
        pyxel.mouse(True)
        pyxel.run(self.update, self.draw)

    def update(self):
        for boid in self.boids:
            boid.update(self.boids)

    def draw(self):
        pyxel.cls(0)
        for boid in self.boids:
            boid.draw()


if __name__ == "__main__":
    App().run()
