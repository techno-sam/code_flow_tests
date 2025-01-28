import html
import math
from abc import abstractmethod
from pygame_svg_shim._util import color_to_hex as _color_to_hex

class DrawCall:
    @abstractmethod
    def to_svg_tag(self) -> str: pass

    @abstractmethod
    def bounds(self) -> tuple[int, int, int, int]:
        """Return the boundaries of this draw call

        :return: (x0, y0, x1, y1)
        """
        pass

    @abstractmethod
    def move(self, offset: tuple[int, int]): pass

    @abstractmethod
    def set_class(self, class_name: str): pass

    @abstractmethod
    def __copy__(self): pass

    def copy(self):
        return self.__copy__()

    def __str__(self):
        return self.to_svg_tag()

    def __repr__(self):
        return f"{self.__class__}({str(self)})"

    def expand_percentages(self, width: float, height: float):
        pass


class DrawLine(DrawCall):
    def __init__(self, start: tuple[int, int], end: tuple[int, int], color: str, width: int, dash_params: tuple[int, int]|None):
        self.start = start
        self.end = end
        self.color = color
        self.width = width
        self.dash_params = dash_params
        self.cls = ""

    def set_class(self, class_name: str):
        self.cls = class_name
        return self

    def expand_percentages(self, width: int, height: int):
        start_x, start_y = self.start
        end_x, end_y = self.end

        start_x: str|int|float
        start_y: str|int|float
        end_x: str|int|float
        end_y: str|int|float

        if type(start_x) == str and start_x[-1] == "%":
            start_x = float(start_x[:-1]) / 100 * width
        if type(start_y) == str and start_y[-1] == "%":
            start_y = float(start_y[:-1]) / 100 * height
        if type(end_x) == str and end_x[-1] == "%":
            end_x = float(end_x[:-1]) / 100 * width
        if type(end_y) == str and end_y[-1] == "%":
            end_y = float(end_y[:-1]) / 100 * height

        self.start = (start_x, start_y)
        self.end = (end_x, end_y)

    def to_svg_tag(self) -> str:
        suffix = ""
        if self.dash_params is not None:
            suffix = f' stroke-dasharray="{self.dash_params[0]},{self.dash_params[1]}"'
        l1 = f'<line x1="{self.start[0]}" y1="{self.start[1]}" x2="{self.end[0]}" y2="{self.end[1]}" stroke="{self.color}" stroke-width="{self.width}"{suffix} class="{self.cls} real" />'
        if self.dash_params is not None:
            return l1
        l2 = f'<line x1="{self.start[0]}" y1="{self.start[1]}" x2="{self.end[0]}" y2="{self.end[1]}" stroke="#000" fill="none" stroke-opacity="0" stroke-width="{self.width*5}"{suffix} class="{self.cls}" />'
        return l1+"\n\t"+l2

    def bounds(self) -> tuple[int, int, int, int]:
        if type(self.start[0]) == str or type(self.end[0]) == str:
            return 0, 0, 0, 0
        x0 = min(self.start[0], self.end[0])
        x1 = max(self.start[0], self.end[0]) + self.width
        y0 = min(self.start[1], self.end[1])
        y1 = max(self.start[1], self.end[1]) + self.width
        return x0, y0, x1, y1

    def move(self, offset: tuple[int, int]):
        if type(self.start[0]) == str or type(self.end[0]) == str:
            self.start = (self.start[0], self.start[1]+offset[1])
            self.end = (self.end[0], self.end[1]+offset[1])
        else:
            self.start = (self.start[0]+offset[0], self.start[1]+offset[1])
            self.end = (self.end[0]+offset[0], self.end[1]+offset[1])

    def __copy__(self):
        return DrawLine(self.start, self.end, self.color, self.width, self.dash_params).set_class(self.cls)


class DrawRect(DrawCall):
    def __init__(self, start: tuple[int, int], size: tuple[int, int], color: str, width: int = 0, rx: int = 0, ry: int = 0):
        self.start = start
        self.size = size
        self.color = color
        self.width = width
        self.rx = rx
        self.ry = ry
        self.cls = ""

    def expand_percentages(self, width: float, height: float):
        # noinspection PyUnresolvedReferences
        if type(self.start[0]) == str and self.start[0][-1] == "%":
            self.start = (float(self.start[0][:-1]) / 100 * width, self.start[1])
        # noinspection PyUnresolvedReferences
        if type(self.start[1]) == str and self.start[1][-1] == "%":
            self.start = (self.start[0], float(self.start[1][:-1]) / 100 * height)

        # noinspection PyUnresolvedReferences
        if type(self.size[0]) == str and self.size[0][-1] == "%":
            self.size = (float(self.size[0][:-1]) / 100 * width, self.size[1])
        # noinspection PyUnresolvedReferences
        if type(self.size[1]) == str and self.size[1][-1] == "%":
            self.size = (self.size[0], float(self.size[1][:-1]) / 100 * height)

    def set_class(self, class_name: str):
        self.cls = class_name
        return self

    def to_svg_tag(self) -> str:
        rounding_suffix = ""
        if self.rx != 0:
            rounding_suffix += f' rx="{self.rx}"'
        if self.ry != 0:
            rounding_suffix += f' ry="{self.ry}"'
        if self.width == 0:
            return f'<rect x="{self.start[0]}" y="{self.start[1]}" width="{self.size[0]}" height="{self.size[1]}" fill="{self.color}"{rounding_suffix} class="{self.cls}" />'
        else:
            return f'<rect x="{self.start[0]}" y="{self.start[1]}" width="{self.size[0]}" height="{self.size[1]}" stroke="{self.color}" stroke-width="{self.width}"{rounding_suffix} class="{self.cls}" />'

    def bounds(self) -> tuple[int, int, int, int]:
        return self.start[0], self.start[1], self.start[0]+self.size[0], self.start[1]+self.size[1]

    def move(self, offset: tuple[int, int]):
        self.start = (self.start[0]+offset[0], self.start[1]+offset[1])

    def __copy__(self):
        return DrawRect(self.start, self.size, self.color, width=self.width, rx=self.rx, ry=self.ry).set_class(self.cls)


class DrawPolygon(DrawCall):
    def __init__(self, points: list[tuple[int, int]], color: str, width: int = 0):
        self.points = points
        self.color = color
        self.width = width
        self.cls = ""

    def set_class(self, class_name: str):
        self.cls = class_name
        return self

    def to_svg_tag(self) -> str:
        points = " ".join([f"{x},{y}" for x, y in self.points])
        if self.width == 0:
            return f'<polygon points="{points}" fill="{self.color}" class="{self.cls} real" />'
        else:
            return f'<polygon points="{points}" stroke="{self.color}" stroke-width="{self.width}" class="{self.cls} real" />'

    def bounds(self) -> tuple[int, int, int, int]:
        if len(self.points) == 0:
            return 0, 0, 0, 0
        x0 = 10000000000000000000000000
        x1 = -10000000000000000000000000
        y0 = 10000000000000000000000000
        y1 = -10000000000000000000000000
        for x, y in self.points:
            x0 = min(x, x0)
            x1 = max(x, x1)
            y0 = min(y, y0)
            y1 = max(y, y1)
        return x0, y0, x1, y1

    def move(self, offset: tuple[int, int]):
        for i in range(len(self.points)):
            self.points[i] = (self.points[i][0]+offset[0], self.points[i][1]+offset[1])

    def __copy__(self):
        return DrawPolygon(self.points.copy(), self.color, self.width).set_class(self.cls)


# find directed angle between (not necessarily the shortest path)
def _directed_angle_between(start: float, end: float) -> float:
    while start < 0 or end < 0:
        start += 2*math.pi
        end += 2*math.pi

    delta = start - end
    if delta < 0:
        delta += 2*math.pi
    return delta


class DrawArc(DrawCall):
    def __init__(self, color: tuple[int, int, int], rect: tuple[int, int, int, int], start_angle: float,
            end_angle: float, width: int = 1):
        self.color = color
        self.rect = rect
        self.start_angle = start_angle * -1
        self.end_angle = end_angle * -1
        while self.start_angle < 0 or self.end_angle < 0:
            self.start_angle += 2*math.pi
            self.end_angle += 2*math.pi
        self.width = width
        self.cls = ""

    def set_class(self, class_name: str):
        self.cls = class_name
        return self

    def to_svg_tag(self) -> str:
        cx = self.rect[0] + self.rect[2] / 2  # X-coordinate of the center
        cy = self.rect[1] + self.rect[3] / 2  # Y-coordinate of the center
        rx = self.rect[2] / 2  # Radius in the x-direction
        ry = self.rect[3] / 2  # Radius in the y-direction

        start_x = cx + rx * math.cos(self.start_angle)  # Starting point x-coordinate
        start_y = cy + ry * math.sin(self.start_angle)  # Starting point y-coordinate

        end_x = cx + rx * math.cos(self.end_angle)  # Ending point x-coordinate
        end_y = cy + ry * math.sin(self.end_angle)  # Ending point y-coordinate

        delta_angle = _directed_angle_between(self.start_angle, self.end_angle)

        large_arc_flag = 1 if delta_angle > math.pi else 0
        # sweep_flag: 0 means travel counter-clockwise from start to end, 1 means travel clockwise from start to end
        # in pygame (and thus self.start_angle, self.end_angle) positive angles are clockwise
        # pygame draws counter-clockwise from start to end
        # in SVG positive angles are counter-clockwise
        sweep_flag = 0#1 if (self.start_angle < self.end_angle) ^ (large_arc_flag==0) else 0

        path_data = f"M {start_x},{start_y} A {rx},{ry} 0 {large_arc_flag},{sweep_flag} {end_x},{end_y}"
        return f'<path d="{path_data}" fill="none" stroke="{_color_to_hex(self.color)}" stroke-width="{self.width}" class="{self.cls} real" />'

    def bounds(self) -> tuple[int, int, int, int]:
        return self.rect[0], self.rect[1], self.rect[0]+self.rect[2], self.rect[1]+self.rect[3]

    def move(self, offset: tuple[int, int]):
        self.rect = (self.rect[0]+offset[0], self.rect[1]+offset[1], self.rect[2], self.rect[3])

    def __copy__(self):
        return DrawArc(self.color, self.rect, self.start_angle, self.end_angle, self.width).set_class(self.cls)


class DrawText(DrawCall):
    def __init__(self, text: str, antialias: bool, color: str, font: str, size: int, metrics: tuple[int, int]):
        self.text = text
        self.antialias = antialias
        self.color = color
        self.font = font
        self.size = size
        self.pos = (0, 0)
        self.metrics = metrics
        self.cls = ""

    @property
    def escaped_text(self) -> str:
        return html.escape(self.text).encode('ascii', 'xmlcharrefreplace').decode()

    def set_class(self, class_name: str):
        self.cls = class_name
        return self

    def to_svg_tag(self) -> str:
        y = self.pos[1] + self.metrics[1]*0.75
        return f'<text x="{self.pos[0]}" y="{y}" fill="{self.color}" font-family="{self.font},Ubuntu Mono,monospace,Arial" font-size="{self.size}" class="{self.cls}">{self.escaped_text}</text>'

    def bounds(self) -> tuple[int, int, int, int]:
        return self.pos[0], self.pos[1], self.pos[0]+self.metrics[0], self.pos[1]+self.metrics[1]

    def move(self, offset: tuple[int, int]):
        self.pos = (self.pos[0]+offset[0], self.pos[1]+offset[1])

    def __copy__(self):
        new = DrawText(self.text, self.antialias, self.color, self.font, self.size, self.metrics)
        new.set_class(self.cls)
        new.pos = self.pos
        return new


class Surface:
    def __init__(self, size: tuple[int, int], flags: int = ...):
        self.background_color = "#000000"
        self.size = size
        self.draw_calls: list[DrawCall] = []
        self.do_rounded_rect = False

    def get_width(self) -> int:
        return self.size[0]

    def line(self, start: tuple[int, int], end: tuple[int, int], color: str, width: int, dash_params: tuple[int, int]|None) -> DrawLine:
        line = DrawLine(start, end, color, width, dash_params)
        self.draw_calls.append(line)
        return line

    def rect(self, start: tuple[int, int], size: tuple[int, int], color: str, width: int = 0, rx: int = 0, ry: int = 0) -> DrawRect:
        rect = DrawRect(start, size, color, width=width, rx=rx, ry=ry)
        self.draw_calls.append(rect)
        return rect

    def polygon(self, points: list[tuple[int, int]], color: str, width: int = 0) -> DrawPolygon:
        polygon = DrawPolygon(points, color, width)
        self.draw_calls.append(polygon)
        return polygon

    def arc(self, color: tuple[int, int, int], rect: tuple[int, int, int, int], start_angle: float,
            end_angle: float, width: int = 1) -> DrawArc:
        arc = DrawArc(color, rect, start_angle, end_angle, width)
        self.draw_calls.append(arc)
        return arc

    def text(self, text: str, antialias: bool, color: tuple[int, int, int], font: str, size: int, metrics: tuple[int, int], svg_class: str):
        draw_text = DrawText(text, antialias, _color_to_hex(color), font, size, metrics)
        draw_text.set_class(svg_class)
        self.draw_calls.append(draw_text)
        return draw_text

    def fill(self, color: tuple[int, int, int]):
        self.draw_calls.clear()
        self.background_color = _color_to_hex(color)

    def blit(self, source: 'Surface', dest: tuple[int, int]):
        new_calls = [dc.copy() for dc in source.draw_calls]
        for dc in new_calls:
            dc.move(dest)
        self.draw_calls.extend(new_calls)