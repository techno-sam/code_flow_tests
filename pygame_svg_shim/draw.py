import pygame_svg_shim.surface as _surface
from pygame_svg_shim._util import color_to_hex as _color_to_hex

dash_params: tuple[int, int]|None = None

svg_class: str = ""

def line(surface: _surface.Surface, color: tuple[int, int, int], start_pos: tuple[int, int], end_pos: tuple[int, int], width: int = 1):
    surface.line(start_pos, end_pos, _color_to_hex(color), width, dash_params).set_class(svg_class)

def rect(surface: _surface.Surface, color: tuple[int, int, int], rect: tuple[int, int, int, int], width: int = 0):
    if surface.do_rounded_rect:
        surface.rect((rect[0], rect[1]), (rect[2], rect[3]), _color_to_hex(color), width=width, rx=100).set_class(svg_class)
    else:
        surface.rect((rect[0], rect[1]), (rect[2], rect[3]), _color_to_hex(color), width=width).set_class(svg_class)

def polygon(surface: _surface.Surface, color: tuple[int, int, int], points: list[tuple[int, int]], width: int = 0):
    surface.polygon(points, _color_to_hex(color), width=width).set_class(svg_class)

def arc(surface: _surface.Surface, color: tuple[int, int, int], rect: tuple[int, int, int, int], start_angle: float, end_angle: float, width: int = 1):
    surface.arc(color, rect, start_angle, end_angle, width).set_class(svg_class)