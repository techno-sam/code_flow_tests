import pygame_svg_shim.surface as _surface
import pygame_svg_shim.draw as _draw

import pygame.font as _font
import pygame.rect as _rect
_font.init()

class SysFont:
    def __init__(self, name: str, size: int):
        self.name = name
        self._size = size
        self._internal = _font.SysFont(name, size)

    def render(self, text: str, antialias: bool, color: tuple[int, int, int]) -> '_FontSurface':
        size = self._internal.size(text)
        surf = _FontSurface(size)
        surf.text(text, antialias, color, self.name, self._size, size, _draw.svg_class)
        return surf

    def size(self, *args, **kwargs):
        return self._internal.size(*args, **kwargs)

class _FontSurface(_surface.Surface):
    def __init__(self, size: tuple[int, int]):
        super().__init__(size)

    def get_rect(self, **kwargs) -> _rect.Rect:
        rect = _rect.Rect(0, 0, self.size[0], self.size[1])
        for key, value in kwargs.items():
            setattr(rect, key, value)
        return rect