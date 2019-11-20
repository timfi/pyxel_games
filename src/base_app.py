from typing import List, ClassVar

import pyxel


__all__ = ("BaseApp",)


class BaseApp:
    class Config:
        width: ClassVar[int]         = 256
        height: ClassVar[int]        = 256
        caption: ClassVar[str]       = pyxel.DEFAULT_CAPTION
        scale: ClassVar[int]         = pyxel.DEFAULT_SCALE
        fps: ClassVar[int]           = pyxel.DEFAULT_FPS
        border_width: ClassVar[int]  = pyxel.DEFAULT_BORDER_WIDTH
        border_color: ClassVar[int]  = pyxel.DEFAULT_BORDER_COLOR
        palette: ClassVar[List[int]] = pyxel.DEFAULT_PALETTE

    def __init_subclass__(cls):
        if hasattr(cls, "Config"):
            new_config = type("Config", (getattr(cls, "Config"), BaseApp.Config), {})
        else:
            new_config = type("Config", (BaseApp.Config,), {})
        setattr(cls, "Config", new_config)

    def run(self, *args, **kwargs):
        pyxel.init(
            self.Config.width, self.Config.height,
            caption=self.Config.caption, scale=self.Config.scale,
            fps=self.Config.fps, palette=self.Config.palette,
            border_color=self.Config.border_color, border_width=self.Config.border_width
        )
        self.init(*args, **kwargs)
        pyxel.run(self.update, self.draw)

    def init(self, *args, **kwargs):
        ...

    def update(self):
        ...

    def draw(self):
        ...
