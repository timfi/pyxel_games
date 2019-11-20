from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt, acos, radians, cos, sin
from random import randrange, random
from typing import List, Union, NamedTuple, Tuple

import pyxel

from base_app import BaseApp

Number = Union[int, float]


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

    def __mul__(self, other: Number) -> Vector2D:
        return Vector2D(
            self.x * other,
            self.y * other
        )

    def __truediv__(self, other: Number) -> Vector2D:
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


@dataclass
class Boid:
    pos: Vector2D
    vel: Vector2D


def average(l: List[Vector2D]) -> Vector2D:
    return sum(l, Vector2D(0, 0)) / len(l) if len(l) > 0 else Vector2D(0, 0)


@dataclass
class App(BaseApp):
    class Config:
        caption: ClassVar[str]       = "Boids"
        scale: ClassVar[int]         = 3
        border_width: ClassVar[int]  = 0
        border_color: ClassVar[int]  = 0

        n_boids: ClassVar[int]     = 100
        boid_speed: ClassVar[int]  = 2
        view_radius: ClassVar[int] = 20
        acceleration: ClassVar[float]      = 0.75
        separation_weight: ClassVar[float] = 1.0
        alignment_weight: ClassVar[float]  = 0.1
        cohesion_weight: ClassVar[float]   = 0.125
        target_weight: ClassVar[float]     = 0.01


    boids: List[Boid] = field(init=False, default_factory=list)

    def init(self, n_boids: int = 100):
        pyxel.mouse(True)
        self.boids = [
            Boid(
                Vector2D(randrange(0, self.Config.width), randrange(0, self.Config.height)),
                Vector2D(random() - 0.5, random() - 0.5).norm
            )
            for _ in range(self.Config.n_boids)
        ]

    def update(self):
        for boid in self.boids:
            neighbors = [
                neighbor
                for j, neighbor in enumerate(self.boids)
                if abs(((boid.pos - neighbor.pos)).mag) <= self.Config.view_radius
            ]

            acc = Vector2D(0, 0)
            if neighbors:
                # 1) Seperation
                acc -= sum(
                    (
                        neighbor.pos - boid.pos
                        for neighbor in neighbors
                        if (neighbor.pos - boid.pos).mag < self.Config.view_radius/2
                    ), Vector2D(0, 0)
                ) * self.Config.separation_weight

                # 2) Alignment
                acc += (
                    (average([neighbor.pos for neighbor in neighbors]) - boid.pos)
                    * self.Config.alignment_weight
                )

                # 3) Cohesion
                acc += (
                    (average([neighbor.vel for neighbor in neighbors]) - boid.vel)
                    * self.Config.cohesion_weight
                )

            # 4) Target
            acc += (
                (Vector2D(pyxel.mouse_x, pyxel.mouse_y) - boid.pos)
                * self.Config.target_weight
                * (pyxel.btn(pyxel.MOUSE_LEFT_BUTTON) - pyxel.btn(pyxel.MOUSE_RIGHT_BUTTON))
            )

            if acc.mag > 0:
                boid.vel += acc.norm * self.Config.acceleration
                if boid.vel.mag > self.Config.boid_speed:
                    boid.vel = boid.vel.norm * self.Config.boid_speed
                elif boid.vel.mag < self.Config.boid_speed * 0.5:
                    boid.vel = boid.vel.norm * 0.5 * self.Config.boid_speed
            boid.pos += boid.vel
            boid.pos %= self.Config.width, self.Config.height

    def draw(self):
        pyxel.cls(1)
        for boid in self.boids:
            vec_o1 = round(boid.pos + boid.vel.orth.norm).as_ituple
            vec_o2 = round(boid.pos - boid.vel.orth.norm).as_ituple

            pyxel.line(*vec_o1, *vec_o2, [5, 6, 7, 6][pyxel.frame_count // 3 % 4])

            vec_l1 = round(boid.pos + boid.vel.norm * 2).as_ituple
            vec_l2 = round(boid.pos - boid.vel.norm * 2).as_ituple

            pyxel.line(*vec_l1, *vec_l2, 6)
            pyxel.pix(*vec_l1, 9)


if __name__ == "__main__":
    app = App()
    app.run()
