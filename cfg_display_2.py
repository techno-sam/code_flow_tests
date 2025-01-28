# Algorithms based on Yuri Stange's "Visualization of Code Flow" (https://www.diva-portal.org/smash/get/diva2:796984/FULLTEXT01.pdf)
import abc
import base64
import json
import math
import random
import sys
import time
import colorsys
import typing
from abc import abstractmethod

import pygame
import pygame_svg_shim

pygame.init()


FONT = pygame.font.SysFont("FiraCode Nerd Font", 20)

RANDOM_DBG_RENDER = False

RENDER_CODE_LINES = True

HORIZONTAL_MARGIN = 30
VERT_MARGIN = 30*2
INTER_TRACK_MARGIN = 17
TRACK_HEIGHT = 2

COLOR_EDGES_BY_NODE_COLORS = False

DUMMY_LINEAR_SEGMENT_MAX_LENGTH = 3

DRAW_TRACKS = True

INCOMING_PORT_WIDTH = 10
INCOMING_INTER_PORT_SPACING = 5

OUTGOING_PORT_WIDTH = 2
OUTGOING_INTER_PORT_SPACING = 4

MOUSE_POS: tuple[int, int] = (0, 0)
SCALE = 1#0.5
SVG_OUTPUT = True
DUMMY_AS_LINE = True
BACK_EDGE_AS_LINE = True
DUMMY_WIDTH = 1
OUTLINE_LINEAR_SEGMENTS = False

DO_SEGMENT_TRANSPOSE = True

LOG_STORED_MOVEMENT = False

# Used to set a breakpoint for a specific iteration of an algorithm
GLOBAL_BREAKPOINT_INDEX = 0

SELECTED_SEGMENTS: tuple[int, list['LinearSegment'], typing.Callable[[list['LinearSegment']], None]]|None = None
"""(needed_count, storage, callback)"""

DO_SELECT: bool = False


def b64enc_css_class(data: str) -> str:
    return base64.urlsafe_b64encode(data.encode('utf-8')).decode('utf-8')#.replace("=", "\\=")


def is_loop_type(typ: str) -> bool:
    return typ == "loop" or typ == "recursion"


# link types:
# if-true (condition passed) -- green
# if-false (condition failed) -- red
# loop (at end of for/while loop, return to start) -- purple
# direct (end of if-block back to outer flow) -- blue


def render_lines(lines: list[str], color: tuple[int, int, int], actually_render: bool = True) -> tuple[pygame.Surface|None, pygame.Rect]:
    line_spacing = 5
    y_vals = []
    rect = pygame.Rect(0, 0, 0, 0)
    for line in lines:
        y_vals.append(rect.height)
        w, h = FONT.size(line)
        rect.width = max(rect.width, w)
        rect.height += h + line_spacing

    if actually_render:
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        for i, line in enumerate(lines):
            surf.blit(FONT.render(line, True, color), (0, y_vals[i]))
        return surf, rect
    else:
        return None, rect


def table_print(rows: list[list[str]]):
    lengths = [len(row) for row in rows]
    assert all(lengths[0] == length for length in lengths), "All rows must have the same length"
    col_widths = [max(len(row[i]) for row in rows) for i in range(lengths[0])]

    """
    format looks like this:
    
    +-----------------+-----------------+-----------------+
    | This is a cell  | This is a cell  | This is a cell  |
    +-----------------+-----------------+-----------------+
    | This is a cell  | This is a cell  | This is a cell  |
    +-----------------+-----------------+-----------------+
    """

    # print header
    sep = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    print(sep)

    for row in rows:
        print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")
        print(sep)



class StopBalance(Exception): pass


def decode_hex_color(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)


def normalize_color(color: tuple[float, float, float]) -> tuple[int, int, int]:
    # noinspection PyTypeChecker
    return tuple(max(0, min(round(c * 255), 255)) for c in color)


def invert_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return 255 - color[0], 255 - color[1], 255 - color[2]


random.seed("CodeFlowVisualization")
def random_rainbow_color() -> tuple[int, int, int]:
    return normalize_color(colorsys.hls_to_rgb(random.random(), random.random()*0.3+0.3, 1.0))


def signum(x: float|int) -> int:
    return -1 if x < 0 else 1


def avg(*args):
    # handle generator case
    if len(args) == 1 and isinstance(args[0], typing.Generator):
        args = list(args[0])
    if len(args) == 0:
        return 0
    return sum(args) / len(args)


EDGE_COLORS: dict[str, tuple[int, int, int]] = {k: decode_hex_color(v) for k, v in
                                                {
                                                    "if-true": "#00FF00",
                                                    "if-false": "#FF0000",
                                                    "loop": "#800080",
                                                    "direct": "#0000FF",
                                                    "recursion": "#FFE100"
                                                }.items()
                                                }



T = typing.TypeVar("T")


class Range:
    def __init__(self, min_: float, max_: float, inclusive: bool = True):
        self._min = min_
        self._max = max_
        self._inclusive = inclusive

    @property
    def min(self) -> float: return self._min

    @property
    def max(self) -> float: return self._max

    @property
    def inclusive(self) -> bool: return self._inclusive

    def __contains__(self, item: float) -> bool:
        if self.inclusive:
            return self.min <= item <= self.max
        else:
            return self.min < item < self.max

    def __repr__(self) -> str:
        return f"Range({self.min}, {self.max}, inclusive={self.inclusive})"

    def __str__(self):
        open_bracket = "[" if self.inclusive else "("
        close_bracket = "]" if self.inclusive else ")"
        return f"Range{open_bracket}{self.min}, {self.max}{close_bracket}"

    def __add__(self, other: 'Range|RangeSet') -> 'RangeSet':
        if isinstance(other, Range):
            return RangeSet(self, other)
        elif isinstance(other, RangeSet):
            return RangeSet(self, *other.ranges)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

    def __radd__(self, other: 'Range|RangeSet') -> 'RangeSet':
        return self + other

    def overlaps(self, other: 'Range') -> bool:
        return self.min < other.max and self.max > other.min

    def offset(self, offset: float) -> 'Range':
        return Range(self.min + offset, self.max + offset, self.inclusive)


class RangeSet:
    def __init__(self, *ranges: Range):
        self.ranges: list[Range] = list(ranges)
        self.condense()

    def __iadd__(self, other: 'Range|RangeSet') -> 'RangeSet':
        if isinstance(other, Range):
            self.ranges.append(other)
        elif isinstance(other, RangeSet):
            self.ranges.extend(other.ranges)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")
        self.condense()
        return self

    def condense(self):
        self.ranges.sort(key=lambda r: r.min)
        i = 0
        while i < len(self.ranges) - 1:
            if self.ranges[i].overlaps(self.ranges[i+1]):
                self.ranges[i] = Range(self.ranges[i].min, self.ranges[i+1].max)
                del self.ranges[i+1]
            else:
                i += 1

    def __contains__(self, item: float) -> bool:
        return any(item in r for r in self.ranges)

    def offset(self, offset: float) -> 'RangeSet':
        return RangeSet(*(r.offset(offset) for r in self.ranges))

    def __repr__(self) -> str:
        return f"RangeSet({', '.join(repr(r) for r in self.ranges)})"

    def __str__(self) -> str:
        return f"RangeSet( {' U '.join(str(r) for r in self.ranges)} )"


def draw_vertical_line(surface: pygame.Surface, color: tuple[int, int, int], x: float|int, y0: float|int, y1: float|int,
                        width: int, crossing_tracks: list[tuple[int, RangeSet]], arc_buffer: list[tuple[int, int, tuple[int, int, int], int]]|None = None):
    if y0 > y1:
        y0, y1 = y1, y0

    crossings: list[int] = []

    for track_y, ranges in crossing_tracks:
        if track_y <= y0 or track_y >= y1:
            continue
        if x in ranges:
            crossings.append(track_y)

    crossing_height = 14

    if len(crossings) == 0:
        pygame.draw.line(surface, color, (x, y0), (x, y1), width)
    else:
        # draw loops
        for track_y in crossings:
            if arc_buffer is not None:
                arc_buffer.append((x - crossing_height//2, track_y - crossing_height//2, color, width))
            else:
                pygame.draw.arc(surface, color, (x - crossing_height//2, track_y - crossing_height//2, crossing_height, crossing_height), 3*math.pi/2, math.pi/2, width)

        # draw vertical line segments
        crossings.append(y1 + crossing_height//2) # just make the loop easier
        for i in range(len(crossings)):
            if i == 0:
                pygame.draw.line(surface, color, (x, y0), (x, crossings[i] - crossing_height // 2), width)
            else:
                pygame.draw.line(surface, color, (x, crossings[i-1] + crossing_height//2), (x, crossings[i] - crossing_height//2), width)


class Node:
    def __init__(self, name: str, color: tuple[int, int, int], children: list['Edge']|None = None,
                 parents: list['Edge']|None = None, depth: int = 0, is_dummy: bool = False, is_ibed: bool = False,
                 is_obed: bool = False, lines: list[str]|None = None, pin_to_bottom: bool = False):
        self.name: str = name
        self._color: tuple[int, int, int] = color
        self.children: list['Edge'] = children or []
        self.parents: list['Edge'] = parents or []
        self.lines: list[str]|None = lines
        self.pin_to_bottom = pin_to_bottom

        ### flipped edges are put in here instead
        ##self.reversed_children: list['Edge'] = []
        ##self.reversed_parents: list['Edge'] = []

        self.ibeds: list['Node'] = []
        self.ibed_parent: 'Node|None' = None
        """Parent (normal node) of this ibed"""
        self.ibed_idx: int|None = None
        self.ibed_depth_diff: int|None = None
        self.ibed_edge_track_index: int | None = None

        self.back_edge_targets: list['Node'] = []
        """Which nodes this node has back edges to"""
        self.obed_parent: 'Node|None' = None
        """Only set for obeds, the node this is tied to"""
        self.obed_edge_track_index: int | None = None
        """Only set for obeds, the edge track this node will use"""

        self.tmp_normal_idx: int|None = None
        self.tmp_obed_idx: int|None = None
        self.tmp_idx: int|None = None

        self._depth: int = depth
        """Old depth metric for debug vis"""

        self.layer: int = 0
        """New layer algorithm layer"""

        self.dfs_active: bool = False
        self.dfs_visited: bool = False

        self.atl_assigned: bool = False
        """Whether this node has been assigned to a layer"""

        self.promote_count: int = 0
        """The number of promotions to a higher (lower-ordinal) layer this node has undergone"""
        self.promote_count0: int = 0

        self.layer_index: int = 0
        """The index within a layer. Not set until after dummy node creation (in prep for uncrossing)"""
        self.uncross_layer_weight: int = 0
        """The weight of this node within its layer, used by the uncrossing algorithm"""

        self.connect_aux_adjacent: list[Edge] = []
        """Auxiliary adjacency map for the connection count algorithm"""

        self.linear_segment: 'LinearSegment' = None
        """The linear segment this node is part of"""

        assert is_dummy + is_ibed + is_obed <= 1, "Node can be at most one of (dummy, ibed, obed)"
        if is_dummy:
            self._type = "dummy"
        elif is_ibed:
            self._type = "ibed"
        elif is_obed:
            self._type = "obed"
        else:
            self._type = "normal"

        self.width: int = DUMMY_WIDTH if self.is_dummy or (BACK_EDGE_AS_LINE and (self.is_ibed or self.is_obed)) else (FONT.size(name)[0] + 10) # *random.randint(5, 13)
        self.height: int = 50

        if RENDER_CODE_LINES:
            if self.lines is not None:
                #self.lines = ["|"+l for l in self.lines]
                self.lines.insert(0, self.name)
                _, rect = render_lines(self.lines, (0, 0, 0), False)
                rect.inflate_ip(40, 40)
                self.width, self.height = rect.size

        self.incoming_ports: int = 1
        self.outgoing_ports: int = 1

        self.forced_depth: int|None = None

    @property
    def is_dummy(self) -> bool: return self._type == "dummy"

    @property
    def is_ibed(self) -> bool: return self._type == "ibed"

    @property
    def is_obed(self) -> bool: return self._type == "obed"

    @property
    def is_normal(self) -> bool: return self._type == "normal"

    def backup_promote_count(self):
        self.promote_count0 = self.promote_count

    def restore_promote_count(self):
        self.promote_count = self.promote_count0

    def apply_promote_count(self):
        """Warn: this does not handle dummy nodes, this should only be run before dummy node creation"""
        assert not any(child.target.is_dummy for child in self.children), "Node#apply_promote_count must be run before dummy node creation"
        assert not any(parent.source.is_dummy for parent in self.parents), "Node#apply_promote_count must be run before dummy node creation"
        self.layer -= self.promote_count
        self.promote_count = 0
        self.promote_count0 = 0

    @property
    def promoted_layer(self) -> int:
        return self.layer - self.promote_count

    @property
    def depth(self) -> int: return self.layer if self.forced_depth is None else self.forced_depth

    @depth.setter
    def depth(self, value: int): pass

    @property
    def color(self) -> tuple[int, int, int]:
        if self.is_dummy:
            return normalize_color(colorsys.hls_to_rgb(((time.time() * 500) % 2000) / 2000, 0.5, 1.0))
        return self._color

    @property
    def non_self_parents(self) -> list['Edge']:
        return [edge for edge in self.parents if edge.source != self]

    @property
    def base_port_x(self) -> float:
        return self.linear_segment.x + (self.width if OUTLINE_LINEAR_SEGMENTS else self.linear_segment.width)/2

    @property
    def ibed_port_x(self) -> float:
        return self.base_port_x - self.width/2

    def port_x_for(self, e: 'Edge') -> float:
        if e.source is self:
            if self.outgoing_ports == 1:
                return self.base_port_x
            # spread out
            allocated_width = self.outgoing_ports * OUTGOING_PORT_WIDTH + (self.outgoing_ports - 1) * OUTGOING_INTER_PORT_SPACING
            if allocated_width > self.width:
                allocated_width = self.width

            # factor is [-0.5, 0.5]
            factor = (e.source_port_index / (self.outgoing_ports - 1)) - 0.5
            return self.base_port_x + factor * allocated_width  # + PORT_WIDTH/2
        elif e.target is self:
            if self.incoming_ports == 1:
                return self.base_port_x
            # spread out
            allocated_width = self.incoming_ports * INCOMING_PORT_WIDTH + (self.incoming_ports - 1) * INCOMING_INTER_PORT_SPACING
            if allocated_width > self.width:
                allocated_width = self.width

            # factor is [-0.5, 0.5]
            factor = (e.target_port_index / (self.incoming_ports-1)) - 0.5
            return self.base_port_x + factor*allocated_width# + PORT_WIDTH/2
        else:
            raise ValueError("Node must be endpoint of edge")

    def assign_ports(self):
        self.parents.sort(key=lambda e: e.source.base_port_x)
        self.children.sort(key=lambda e: e.target.base_port_x)

        self.incoming_ports = 1
        last_color: tuple[int, int, int]|None = None

        if len(self.ibeds) > 0:
            self.incoming_ports += 1

        for parent in self.parents:
            if parent.is_flipped:
                continue
            color = parent.rendered_color
            if last_color is not None and last_color != color:
                self.incoming_ports += 1
            parent.target_port_index = self.incoming_ports - 1
            last_color = color

        self.outgoing_ports = 1
        last_color = None

        if len(self.back_edge_targets) > 0: # has obeds
            self.outgoing_ports += 1

        for child in self.children:
            if child.is_flipped:
                continue
            color = child.rendered_color
            if last_color is not None and last_color != color:
                self.outgoing_ports += 1
            child.source_port_index = self.outgoing_ports - 1
            last_color = color

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("Node", self.name))

    def __str__(self) -> str:
        return f"Node({self.name}, {self._color}, #children: {len(self.children)}, depth: {self.depth}, layer: {self.layer}{', dummy' if self.is_dummy else ''})"

    def __repr__(self) -> str:
        return str(self)

    def add_child(self, child: 'Node', typ: str) -> 'Edge':
        assert not self.pin_to_bottom, "Bottom-pinned node cannot have children"
        if not is_loop_type(typ):
            child.depth = max(child.depth, self.depth + 1)
        e = Edge(self, child, typ)
        self.children.append(e)
        child.parents.append(e)
        return e

    def deepest(self, seen: set['Node'] | None = None) -> int:
        if seen is None:
            seen = set()

        if self in seen:
            return self.depth

        seen.add(self)
        return max([child.target.deepest(seen) for child in self.children] + [self.depth])

    @staticmethod
    def _is_immutable(obj: T, _memodict: dict) -> bool:
        if id(obj) in _memodict:
            return _memodict[id(obj)]
        if isinstance(obj, tuple):
            #print(obj)
            _memodict[id(obj)] = True
            ret = all(Node._is_immutable(item, _memodict) for item in obj)
        elif type(obj) in {int, float, str, bool, type(None)}:
            ret = True
        else:
            ret = False
        _memodict[id(obj)] = ret
        return ret


    _DEEPCOPY_WIP_MARKER = object()


    @staticmethod
    def _deepcopy(obj: T, _memodict: dict, edges_to_create: list[tuple[str, str, str, bool]], nodes: dict[str, 'Node']) -> T:
        """

        :param obj: object to copy
        :param _memodict: already-copied objects
        :param edges_to_create: list[tuple[source_name, target_name, typ, flipped]]
        :param nodes: created nodes for linkage later
        :return:
        """
        if id(obj) in _memodict:
            return _memodict[id(obj)]
        if Node._is_immutable(obj, {}):
            return obj
        elif isinstance(obj, Node):
            _memodict[id(obj)] = Node._DEEPCOPY_WIP_MARKER

            new_obj = Node(obj.name, obj._color)
            nodes[obj.name] = new_obj
            for k, v in obj.__dict__.items():
                if k in {"children", "parents"}:
                    v: list['Edge']
                    new_obj.__dict__[k] = []
                    if k == "children":
                        for edge in v:
                            edges_to_create.append((edge.source.name, edge.target.name, edge.typ, edge._flipped))
                            new_copy = Node._deepcopy(edge.target, _memodict, edges_to_create, nodes)
                            if new_copy is Node._DEEPCOPY_WIP_MARKER:
                                print("Warning: circular reference detected in deepcopy, skipping")
                            else:
                                nodes[edge.target.name] = new_copy
                elif k == "connect_aux_adjacent": # not important to copy
                    new_obj.__dict__[k] = []
                else:
                    new_obj.__dict__[Node._deepcopy(k, _memodict, edges_to_create, nodes)] = Node._deepcopy(v, _memodict, edges_to_create, nodes)
            ret = new_obj
        elif isinstance(obj, list):
            ret = [Node._deepcopy(item, _memodict, edges_to_create, nodes) for item in obj]
        elif isinstance(obj, dict):
            ret = {Node._deepcopy(k, _memodict, edges_to_create, nodes): Node._deepcopy(v, _memodict, edges_to_create, nodes) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            ret = tuple(Node._deepcopy(item, _memodict, edges_to_create, nodes) for item in obj)
        elif isinstance(obj, set):
            ret = {Node._deepcopy(item, _memodict, edges_to_create, nodes) for item in obj}
        else:
            raise ValueError(f"Unsupported type: {type(obj)}")
        _memodict[id(obj)] = ret
        return ret

    def deepcopy(self) -> 'Node':
        # list[tuple[source_name, target_name, typ, flipped]]
        edges_to_create: list[tuple[str, str, str, bool]] = []
        nodes: dict[str, 'Node'] = {}

        new_node = Node._deepcopy(self, {}, edges_to_create, nodes)

        for source_name, target_name, typ, flipped in edges_to_create:
            source = nodes[source_name]
            target = nodes[target_name]
            e = Edge(source, target, typ, flipped=flipped)
            source.children.append(e)
            target.parents.append(e)

        return new_node


class Edge:
    def __init__(self, source: Node, target: Node, typ: str, flipped: bool = False):
        self.id_: str = f"{source.name} -> {target.name}"
        """basically final"""

        self.source: Node = source
        self.target: Node = target
        self.typ: str = typ
        self._flipped = flipped

        self.track_index: int|None = None
        self.source_port_index: int = 0
        self.target_port_index: int = 0

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.id_ == other.id_

    def __hash__(self):
        return hash(("Edge", self.id_))

    @property
    def edge_start_x(self) -> float:
        return min(self.source.port_x_for(self), self.target.port_x_for(self))

    @property
    def edge_end_x(self) -> float:
        return max(self.source.port_x_for(self), self.target.port_x_for(self))

    @property
    def rendered_color(self) -> tuple[int, int, int]:
        if DUMMY_AS_LINE and COLOR_EDGES_BY_NODE_COLORS:
            if self.is_flipped:
                return self.non_dummy_target.color
            else:
                return self.non_dummy_source.color
        else:
            return EDGE_COLORS[self.typ]

    @property
    def is_loop(self) -> bool: return is_loop_type(self.typ)

    @property
    def is_flipped(self) -> bool: return self._flipped

    @property
    def is_single_parent(self) -> bool: return self.typ == "if-true" or self.typ == "if-false"

    @property
    def crossing_startpoint(self) -> Node:
        """return the left-to-right 'startpoint' for the crossing count algorithm"""
        return self.source if self.source.layer_index < self.target.layer_index else self.target

    @property
    def crossing_endpoint(self) -> Node:
        """return the left-to-right 'endpoint' for the crossing count algorithm"""
        return self.target if self.source.layer_index < self.target.layer_index else self.source

    @property
    def non_dummy_source(self) -> Node:
        current = self.source
        while current.is_dummy:
            current = current.parents[0].source
        return current

    @property
    def non_dummy_target(self) -> Node:
        current = self.target
        while current.is_dummy:
            if len(current.children) == 0:
                print("Warning: no children for dummy node", file=sys.stderr)
                # noinspection PyTypeChecker
                return None
            current = current.children[0].target
        return current

    def flipped(self) -> 'Edge':
        if self._flipped:
            print("Warning, un-flipping an already flipped edge!")
        return Edge(self.target, self.source, self.typ, flipped = not self._flipped)

    def reverse_in_place(self):
        self.source, self.target = self.target, self.source
        self._flipped = not self._flipped

        self.target.children.remove(self)
        self.target.parents.append(self)

        self.source.children.append(self)
        self.source.parents.remove(self)

    def insert_dummy_nodes(self, layers: list[list[Node]]):
        if self.source.layer + 1 == self.target.layer or self.source.layer == self.target.layer:
            return

        dummies = [
            Node(f"dummy_{self.source.name}_{self.target.name}_{i}", (255, 255, 255), depth = self.source.depth + 1, is_dummy=True)
            for i in range(self.target.layer - self.source.layer - 1)
        ]
        layer = self.source.layer
        for dummy in dummies:
            layer += 1
            dummy.layer = layer
            layers[layer].insert(int((self.source.layer_index + self.target.layer_index)/2), dummy)

        self.target.parents.remove(self)

        # link source to first dummy, reusing this edge
        target = self.target
        self.target = dummies[0]
        self.id_ = f"{self.source.name} -> {self.target.name}"
        dummies[0].parents.append(self)

        # link dummies to each other
        for i in range(len(dummies) - 1):
            dummies[i].add_child(dummies[i+1], self.typ)._flipped = self._flipped

        # link last dummy to target
        dummies[-1].add_child(target, self.typ)._flipped = self._flipped

    def __str__(self):
        return f"Edge({self.source.name} -> {self.target.name}, {self.typ}, flipped: {self._flipped})"

    def __repr__(self):
        return str(self)


class LinearSegment:
    def __init__(self, *initial_members: Node):
        self.nodes: list[Node] = [*initial_members]

        # we don't need to store fancy edge data here, that's still encoded in the nodes
        self.parents: list['LinearSegment'] = []
        self.children: list['LinearSegment'] = []

        self._min_layer: int = None
        self._max_layer: int = None
        self._min_index: int = None
        self._max_index: int = None
        self.ranking_index: int = None
        self._width: float = 0.0

        self.x: float = 0.0
        self.assigned_coords: bool = False

        self._color: tuple[int, int, int] = random_rainbow_color()

        self.pendulum_force_from_parent_layer: float = 0.0
        """The force that the parent layer wanted to apply to this segment, but couldn't because of blockage on other layers"""

    def __hash__(self):
        return hash(tuple(self.nodes))

    @property
    def min_layer(self) -> int: return self._min_layer

    @property
    def max_layer(self) -> int: return self._max_layer

    @property
    def min_index(self) -> int: return self._min_index

    @property
    def max_index(self) -> int: return self._max_index

    @property
    def width(self) -> float: return self._width+HORIZONTAL_MARGIN

    @property
    def width_no_pad(self) -> float: return self._width

    @property
    def color(self) -> tuple[int, int, int]: return self._color

    @property
    def center_x(self) -> float:
        return self.x + self.width_no_pad / 2

    def is_next_to(self, other: 'LinearSegment') -> tuple[int, int]|None:
        """
        Check if this segment is adjacent to another segment

        :param other: segment to check adjacency with
        :return: tuple of (min_layer, max_layer) if adjacent, else None
        """
        # find overlapping layer indexes
        min_layer = max(self.min_layer, other.min_layer)
        max_layer = min(self.max_layer, other.max_layer)

        if min_layer > max_layer:
            return None

        # check if top nodes are adjacent
        my_top = self.node_for_layer(min_layer)
        other_top = other.node_for_layer(min_layer)
        if abs(my_top.layer_index - other_top.layer_index) != 1:
            return None

        # check if bottom nodes are adjacent
        my_bottom = self.node_for_layer(max_layer)
        other_bottom = other.node_for_layer(max_layer)
        if abs(my_bottom.layer_index - other_bottom.layer_index) != 1:
            return None

        #if abs(self.ranking_index - other.ranking_index) != 1:
        #    return None
        return min_layer, max_layer

    def transpose_with(self, other: 'LinearSegment', layers: list[list[Node]]):
        if self.ranking_index > other.ranking_index:
            other.transpose_with(self, layers)
            return
        # find overlapping layer indexes
        min_layer = max(self.min_layer, other.min_layer)
        max_layer = min(self.max_layer, other.max_layer)

        if min_layer > max_layer:
            return

        for i in range(min_layer, max_layer+1):
            self_node = self.node_for_layer(i)
            other_node = other.node_for_layer(i)
            layer = layers[i]

            layer[self_node.layer_index], layer[other_node.layer_index] = layer[other_node.layer_index], layer[self_node.layer_index]
            self_node.layer_index, other_node.layer_index = other_node.layer_index, self_node.layer_index

        self.ranking_index, other.ranking_index = other.ranking_index-1, self.ranking_index+1

        self.recalculate()
        other.recalculate()

    def node_for_layer(self, layer: int):
        node = self.nodes[layer - self.min_layer]
        assert node.layer == layer
        return node

    def recalculate(self):
        self._min_layer = min(node.layer for node in self.nodes)
        self._max_layer = max(node.layer for node in self.nodes)
        self._min_index = min(node.layer_index for node in self.nodes)
        self._max_index = max(node.layer_index for node in self.nodes)
        self._width = max(node.width for node in self.nodes)

    def __str__(self):
        return f"LinearSegment({', '.join(node.name for node in self.nodes)} Layers {self.min_layer}-{self.max_layer} Indices {self.min_index}-{self.max_index})"

    def __repr__(self):
        return str(self)

    def force_from_above(self) -> float:
        if len(self.parents) == 0:
            return 0.0
        return avg(parent.center_x-self.center_x for parent in self.parents) / len(self.parents)

    def force_from_below(self) -> float:
        if len(self.children) == 0:
            return 0.0
        return avg(child.center_x-self.center_x for child in self.children) / len(self.children)



def display_graph_with_layout(surf: pygame.Surface, root: Node, layers: list[list[Node]]|None = None,
                              use_linear_segment_coords: bool = False, offset: float = 0.0, vert_offset: int = 0,
                              highlight_layer: int|None = None, interlinear: list[int]|None = None,
                              interlinear_ranges: list[list[RangeSet]]|None = None):
    # interlinear: [number of edge tracks]
    # first we need to lay out each layer:
    # each node has a width of 150 and a height of 50
    # on each layer, there must be a gap of 15 between each node
    # between layers, there must be a gap of 30
    # the horizontal center of each node SHOULD be the average of the centers of its children
    if root.depth != 0:
        raise ValueError("Root node must have depth 0")

    if RANDOM_DBG_RENDER:
        arc_size = 40 # 14
        pygame.draw.rect(surf, (255, 0, 0), (10, 10, arc_size, arc_size))

        # 3*math.pi/2, math.pi/3
        # math.pi/6, math.pi/2 - current metric fails
        # math.pi/4, math.pi/6 - current metric fails
        # 3*math.pi/6, 0 - current metric (less than 90 deg works)
        # 3*math.pi/6, 5*math.pi/4 - current metric works
        pygame.draw.arc(surf, (255, 255, 0), (10, 10, arc_size, arc_size), 3*math.pi/6, 5*math.pi/4, width=2)

        def coord(angle):
            relx = arc_size//2 * math.cos(-angle)
            rely = arc_size//2 * math.sin(-angle)
            return (10+arc_size//2 + relx, 10+arc_size//2 + rely)

        tmp_x, tmp_y = coord(math.pi/4)
        if "pygame_bkp" in globals():
            print("Coords[0]: ", tmp_x, tmp_y)
            other_x, other_y = coord(math.pi / 6)
            print("Coords[1]: ", other_x, other_y)
            print("Radius: ", arc_size // 2)
        pygame.draw.rect(surf, (0, 255, 0), (tmp_x - 1, tmp_y - 1, 2, 2))

    y_tmp = vert_offset
    y_values: list[int] = []

    for d in range(root.deepest() + 1):
        layer_height = 50
        if d == 0:
            layer_height = -VERT_MARGIN // 2
        elif layers is not None:
            layer_height = max(n.height for n in layers[d-1])
        interlinear_height = 0
        if interlinear is not None:
            interlinear_height = max(interlinear[d] - 1, 0)*INTER_TRACK_MARGIN + interlinear[d]*TRACK_HEIGHT
        y_tmp += layer_height + VERT_MARGIN + interlinear_height
        y_values.append(y_tmp)

    blocked_ranges: list[tuple[int, RangeSet]] = []

    # draw edge tracks
    if interlinear is not None:
        interlinear_y_values = [[] for _ in range(len(y_values)+1)]
        for d in range(root.deepest()+1):
            bottom_y = y_values[d]
            interlinear_height = (interlinear[d] - 1) * INTER_TRACK_MARGIN + interlinear[d] * TRACK_HEIGHT
            top_y = bottom_y - VERT_MARGIN/2 - interlinear_height
            for i in range(interlinear[d]):
                y_val = top_y + i * (TRACK_HEIGHT + INTER_TRACK_MARGIN)
                interlinear_y_values[d].append(y_val)
                if DRAW_TRACKS:
                    if "pygame_bkp" in globals():
                        # noinspection PyTypeChecker
                        draw_dashed_line(surf, (0, 0, 0), ("0%", y_val), ("100%", y_val), 1, 3, 3)
                    else:
                        draw_dashed_line(surf, (0, 0, 0), (0, y_val), (surf.get_width(), y_val), 1, 3, 3)

        # last layer, just create a local scope
        if True:
            top_y = y_values[-1] + max(n.height for n in layers[-1]) + VERT_MARGIN/2
            for i in range(interlinear[-1]):
                y_val = top_y + i * (TRACK_HEIGHT + INTER_TRACK_MARGIN)
                interlinear_y_values[-1].append(y_val)
                if DRAW_TRACKS:
                    if "pygame_bkp" in globals():
                        # noinspection PyTypeChecker
                        draw_dashed_line(surf, (0, 0, 0), ("0%", y_val), ("100%", y_val), 1, 3, 3)
                    else:
                        draw_dashed_line(surf, (0, 0, 0), (0, y_val), (surf.get_width(), y_val), 1, 3, 3)

        for i, y_vals in enumerate(interlinear_y_values):
            for j, y in enumerate(y_vals):
                blocked_ranges.append((y, interlinear_ranges[i][j].offset(offset)))

    if highlight_layer is not None and highlight_layer < len(y_values):
        highlight_top = y_values[highlight_layer]
        highlight_bottom = highlight_top + 50
        pygame.draw.rect(surf, (255, 255, 0), (0, highlight_top, surf.get_width(), highlight_bottom - highlight_top), width=0)

    nodes_by_layer: list[list[Node]] = [[] for d in range(root.deepest() + 1)]
    todo: list[Node] = [root]
    all_nodes: list[Node] = []
    while len(todo) > 0:
        current = todo.pop(0)
        all_nodes.append(current)
        nodes_by_layer[current.depth].append(current)
        for edge in current.children:
            if edge.target not in todo and edge.target not in all_nodes:
                todo.append(edge.target)
    if layers is not None:
        nodes_by_layer = layers
        all_nodes.clear()
        all_nodes = sum((layer for layer in layers), [])
    del layers

    x_coordinates: dict[str, float] = {}
    layers: list[tuple[int, list[tuple[Node, int]]]] = []

    if use_linear_segment_coords:
        for node in all_nodes:
            x_coordinates[node.name] = node.linear_segment.x# + (node.linear_segment.width - node.width)/2
    else: # deprecated, not implementing new widths for this
        for layer, nodes in reversed(list(enumerate(nodes_by_layer))):
            needed_width = 150 * len(nodes) + 15 * (len(nodes) - 1)
            desired_center = 800 // 2
            if layer > 0:
                pass # this might need a two-pass algorithm or *shudder* some sort of spring simulation

            center = desired_center
            layer_x_coordinates = {node.name: center - needed_width // 2 + 150 * i + 15 * i for i, node in enumerate(nodes)}
            layers.append((center, [(node, layer_x_coordinates[node.name]) for node in nodes]))
            x_coordinates.update(layer_x_coordinates)

    x_coordinates = {k: v + offset for k, v in x_coordinates.items()}

    display_graph(surf, all_nodes, x_coordinates, y_values, outline_linear_segments=use_linear_segment_coords, edge_tracks=interlinear_y_values if interlinear is not None else None, offset=offset, blocked_ranges=blocked_ranges)

    #print("\n".join(str(v) for v in nodes_by_layer))
    #raise StopIteration


def draw_dashed_line(surface: pygame.Surface, color: tuple[int, int, int],
                    start_pos: tuple[int|float, int|float], end_pos: tuple[int|float, int|float],
                    width: int = 1, dash_length: int = 10, space_length: int = 10):
    if "pygame_bkp" in globals():
        pygame_svg_shim.draw.dash_params = (dash_length, space_length)
        pygame.draw.line(surface, color, start_pos, end_pos, width)
        pygame_svg_shim.draw.dash_params = None
        return
    x1, y1 = start_pos
    x2, y2 = end_pos
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))
    if distance == 0:
        return
    dx = dx / distance
    dy = dy / distance
    x, y = x1, y1
    for i in range(int(distance / (dash_length + space_length))):
        try:
            pygame.draw.line(surface, color, (round(x), round(y)), (round(x + dash_length * dx), round(y + dash_length * dy)), width)
        except TypeError:
            return
        x += dash_length * dx + space_length * dx
        y += dash_length * dy + space_length * dy

    try:
        pygame.draw.line(surface, color, (round(x), round(y)), end_pos, width)
    except TypeError:
        pass


def display_graph(surf: pygame.Surface, nodes: list[Node], x_coordinates: dict[str, float], y_values: list[int],
                  x_coordinates0: dict[str, float]|None = None, reverse: bool = False,
                  outline_linear_segments: bool = False, edge_tracks: list[list[int]]|None = None, offset: float = 0,
                    blocked_ranges: list[tuple[int, RangeSet]]|None = None):
    global SELECTED_SEGMENTS, DO_SELECT
    if reverse:
        nodes = nodes[::-1]

    pygame_svg_shim.draw.svg_class = ""

    if blocked_ranges is None:
        blocked_ranges = []

    tooltip: str|None = None

    arc_buffer: list[tuple[int, int, tuple[int, int, int], int]]|None = []
    if "pygame_bkp" in globals():
        arc_buffer = None

    outline_linear_segments &= OUTLINE_LINEAR_SEGMENTS

    if outline_linear_segments:
        # draw outlines
        for node in nodes:
            if node.linear_segment is None:
                continue
            if BACK_EDGE_AS_LINE:
                if node.is_ibed or node.is_obed:
                    continue
                if node.is_dummy and node.parents[0].non_dummy_source.is_ibed:
                    continue

            top_y = y_values[node.depth]
            left_x = x_coordinates[node.name]

            idx = node.linear_segment.nodes.index(node)
            dash_bottom_y = top_y+node.height+(VERT_MARGIN//2)
            dash_top_y = top_y-(VERT_MARGIN - VERT_MARGIN//2) # ensures that rounding is done properly
            color = node.linear_segment.color
            width = node.linear_segment.width_no_pad
            if idx == 0: # top
                draw_dashed_line(surf, color, (left_x-3-2, top_y-6), (left_x + width + 3, top_y - 6), dash_length=5, space_length=5, width=2)
                dash_top_y = top_y - 6

                # handle tooltip stuff
                mx, my = MOUSE_POS
                complete_bottom = y_values[node.linear_segment.nodes[-1].depth]+node.height+4
                if left_x-3-2 <= mx <= left_x+ width +3 and top_y-6 <= my <= complete_bottom:
                    tooltip = f"Force from above: {node.linear_segment.force_from_above()}\n" \
                              f"Force from below: {node.linear_segment.force_from_below()}\n" \
                              f"Ranking index: {node.linear_segment.ranking_index}"

                    if DO_SELECT:
                        print("Selected segment!")
                        DO_SELECT = False
                        # noinspection PyUnresolvedReferences
                        SELECTED_SEGMENTS[1].append(node.linear_segment)
                        # noinspection PyTypeChecker
                        if len(SELECTED_SEGMENTS[1]) == SELECTED_SEGMENTS[0]:
                            # noinspection PyCallingNonCallable
                            SELECTED_SEGMENTS[2](SELECTED_SEGMENTS[1])
                            globals()["SELECTED_SEGMENTS"] = None

            if idx == len(node.linear_segment.nodes) - 1: # bottom
                draw_dashed_line(surf, color, (left_x-3-2, top_y+node.height+4), (left_x + width + 3, top_y + node.height + 4), dash_length=5, space_length=5, width=2)
                dash_bottom_y = top_y+node.height+4
            # middle lines
            draw_dashed_line(surf, color, (left_x + width + 3, dash_top_y), (left_x + width + 3, dash_bottom_y), dash_length=5, space_length=5, width=2)
            draw_dashed_line(surf, color, (left_x-3-2, dash_top_y), (left_x-3-2, dash_bottom_y), dash_length=5, space_length=5, width=2)

    # draw nodes
    for node in nodes:
        if BACK_EDGE_AS_LINE and False:
            if node.is_ibed or node.is_obed:
                continue
            if node.is_dummy and node.parents[0].non_dummy_source.is_ibed:
                continue
        centering_offset = 0 if outline_linear_segments else (node.linear_segment.width - node.width) / 2
        # ibeds need special handling b/c no parents :(
        if (node.is_dummy and DUMMY_AS_LINE) or ((node.is_ibed or node.is_obed) and BACK_EDGE_AS_LINE):
            top_y = y_values[node.depth]
            left_x = node.base_port_x+offset#x_coordinates[node.name] + centering_offset
            # pygame.draw.line(surf, node.parents[0].non_dummy_source.color, (left_x + node.width//2, top_y), (left_x + node.width//2, top_y + node.height), width=2)

            if node.is_dummy:
                source = node.parents[0].non_dummy_source
                target = node.children[0].non_dummy_target

                if source.is_ibed:
                    source, target = target.obed_parent, source.ibed_parent

                from_name = source.name
                to_name = target.name
                if COLOR_EDGES_BY_NODE_COLORS:
                    col = node.parents[0].non_dummy_source.color
                else:
                    col = EDGE_COLORS[node.parents[0].typ]
            elif node.is_ibed:
                from_name = node.children[0].non_dummy_target.obed_parent.name
                to_name = node.ibed_parent.name
                col = EDGE_COLORS[node.children[0].typ]
            elif node.is_obed:
                from_name = node.obed_parent.name
                to_name = node.parents[0].non_dummy_source.ibed_parent.name
                col = EDGE_COLORS[node.parents[0].typ]
            else:
                raise AssertionError("Unreachable code")

            pygame_svg_shim.draw.svg_class = f"edge from_{b64enc_css_class(from_name)} to_{b64enc_css_class(to_name)}"
            draw_vertical_line(surf, col, left_x + node.width // 2, top_y, top_y + node.height, 2, blocked_ranges, arc_buffer=arc_buffer)
            pygame_svg_shim.draw.svg_class = ""
        else:
            top_y = y_values[node.depth]
            left_x = x_coordinates[node.name]+centering_offset

            if node.name.endswith(".py") and surf.__class__.__module__ == "pygame_svg_shim.surface":
                surf.do_rounded_rect = True
            pygame_svg_shim.draw.svg_class = f"node node_{b64enc_css_class(node.name)}"
            pygame.draw.rect(surf, node.color, (left_x, top_y, node.width, node.height), width=(3 if node.is_dummy else 0))

            if node.name.endswith(".py") and surf.__class__.__module__ == "pygame_svg_shim.surface":
                surf.do_rounded_rect = False

            if RENDER_CODE_LINES and node.lines is not None:
                label, label_rect = render_lines(node.lines, (0, 0, 0) if node.is_dummy else invert_color(node.color))
                label_rect.center = (left_x + node.width//2, top_y + node.height//2)
                surf.blit(label, label_rect)
            else:
                label = FONT.render("d" if node.is_dummy else node.name, True, (0, 0, 0) if node.is_dummy else invert_color(node.color))
                label_rect = label.get_rect(center=(left_x + node.width//2, top_y + node.height//2))
                surf.blit(label, label_rect)
            pygame_svg_shim.draw.svg_class = ""

    # draw edges
    for node in nodes:
        centering_offset = 0 if outline_linear_segments else (node.linear_segment.width - node.width) / 2
        top_y = y_values[node.depth]
        left_x = x_coordinates[node.name] + centering_offset

        #p0 = (left_x + node.width//2, top_y + node.height)
        for edge in node.children:
            if edge.is_flipped and not (edge.non_dummy_source.is_ibed or edge.non_dummy_source.is_obed):
                pygame_svg_shim.draw.dash_params = (5, 5)

            naming_source = edge.non_dummy_source
            naming_target = edge.non_dummy_target

            if naming_source.is_ibed:
                naming_source, naming_target = naming_target.obed_parent, naming_source.ibed_parent

            pygame_svg_shim.draw.svg_class = f"edge from_{b64enc_css_class(naming_source.name)} to_{b64enc_css_class(naming_target.name)}"
            p0 = (node.port_x_for(edge)+offset, top_y + node.height)
            child_centering_offset = 0 if outline_linear_segments else (edge.target.linear_segment.width - edge.target.width) / 2
            #p1 = (x_coordinates[edge.target.name] + edge.target.width//2 + child_centering_offset, y_values[edge.target.depth])
            p1 = (edge.target.port_x_for(edge)+offset, y_values[edge.target.depth])
            edge_color = edge.rendered_color
            if edge_tracks is None or edge.track_index is None: # vertical edges might not have an assigned track
                #if edge.track_index is None:
                #    print(f"Edge {edge} does not have an assigned track!")
                draw_vertical_line(surf, edge_color, p0[0], p0[1], p1[1], 2, blocked_ranges, arc_buffer=arc_buffer) # pygame.draw.line(surf, edge_color, p0, p1, 2)
            else:
                track_y = edge_tracks[edge.source.layer+1][edge.track_index]
                draw_vertical_line(surf, edge_color, p0[0], p0[1], track_y, 2, blocked_ranges, arc_buffer=arc_buffer) # pygame.draw.line(surf, edge_color, p0, (p0[0], track_y), 2)
                pygame.draw.line(surf, edge_color, (p0[0], track_y), (p1[0], track_y), 2)
                draw_vertical_line(surf, edge_color, p1[0], track_y, p1[1], 2, blocked_ranges, arc_buffer=arc_buffer) # pygame.draw.line(surf, edge_color, (p1[0], track_y), p1, 2)
            if not (edge.non_dummy_source.is_ibed or edge.non_dummy_source.is_obed):
                if edge.is_flipped:
                    if not (DUMMY_AS_LINE and edge.source.is_dummy):
                        # draw arrow tip facing up
                        pygame.draw.polygon(surf, edge_color, [
                            (p0[0] - 5, p0[1] + 5),
                            (p0[0] + 5, p0[1] + 5),
                            (p0[0], p0[1])
                        ])
                else:
                    if not (DUMMY_AS_LINE and edge.target.is_dummy):
                        # draw arrow tip facing down
                        pygame.draw.polygon(surf, edge_color, [
                            (p1[0] - 5, p1[1] - 5),
                            (p1[0] + 5, p1[1] - 5),
                            (p1[0], p1[1])
                        ])

            pygame_svg_shim.draw.dash_params = None
        pygame_svg_shim.draw.svg_class = ""

        # draw ibed connections
        for ibed in node.ibeds:
            non_dummy_target = ibed.children[0].non_dummy_target
            if non_dummy_target is None:
                print(f"Warning: ibed {ibed} does not have a non-dummy target", file=sys.stderr)
                continue
            ultimate_parent = non_dummy_target.obed_parent
            if BACK_EDGE_AS_LINE:
                from_name = ultimate_parent.name
            else:
                from_name = ibed.name

            pygame_svg_shim.draw.svg_class = f"edge from_{b64enc_css_class(from_name)} to_{b64enc_css_class(node.name)}"

            # draw lines up
            p0 = (ibed.base_port_x+offset, top_y)
            p1 = (node.ibed_port_x+offset, top_y)
            track_y = edge_tracks[node.layer][ibed.ibed_edge_track_index]

            color = ultimate_parent.color if COLOR_EDGES_BY_NODE_COLORS else EDGE_COLORS[ibed.children[0].typ]

            draw_vertical_line(surf, color, p1[0], p1[1], track_y, 2, blocked_ranges, arc_buffer=arc_buffer)
            pygame.draw.line(surf, color, (p0[0], track_y), (p1[0], track_y), 2)

            if not BACK_EDGE_AS_LINE or True:
                draw_vertical_line(surf, color, p0[0], p0[1], track_y, 2, blocked_ranges, arc_buffer=arc_buffer)

            # draw arrow tip facing down
            pygame.draw.polygon(surf, color, [
                (p1[0] - 5, p1[1] - 5),
                (p1[0] + 5, p1[1] - 5),
                (p1[0], p1[1])
            ])

            pygame_svg_shim.draw.svg_class = ""
        # draw obed connection
        if node.is_obed:
            obed_target = node.obed_parent

            if BACK_EDGE_AS_LINE:
                my_ibed = node.parents[0].non_dummy_source
                to_name = my_ibed.ibed_parent.name
            else:
                my_ibed = None
                to_name = node.name
            pygame_svg_shim.draw.svg_class = f"edge from_{b64enc_css_class(obed_target.name)} to_{b64enc_css_class(to_name)}"

            # draw lines down
            p0 = (node.base_port_x+offset, top_y+node.height)
            p1 = (obed_target.base_port_x+offset, top_y+obed_target.height)
            track_y = edge_tracks[node.layer+1][node.obed_edge_track_index]

            color = obed_target.color if COLOR_EDGES_BY_NODE_COLORS else EDGE_COLORS[node.parents[0].typ]

            draw_vertical_line(surf, color, p1[0], p1[1], track_y, 2, blocked_ranges, arc_buffer=arc_buffer)
            pygame.draw.line(surf, color, (p0[0], track_y), (p1[0], track_y), 2)

            draw_vertical_line(surf, color, p0[0], p0[1], track_y, 2, blocked_ranges, arc_buffer=arc_buffer)
            if BACK_EDGE_AS_LINE:
                top_track_y = edge_tracks[my_ibed.layer][my_ibed.ibed_edge_track_index]
                #draw_vertical_line(surf, color, p0[0], top_track_y, track_y, 2, blocked_ranges, arc_buffer=arc_buffer)
            else:
                #draw_vertical_line(surf, color, p0[0], p0[1], track_y, 2, blocked_ranges, arc_buffer=arc_buffer)

                # draw arrow tip facing up
                pygame.draw.polygon(surf, color, [
                    (p0[0] - 5, p0[1] + 5),
                    (p0[0] + 5, p0[1] + 5),
                    (p0[0], p0[1])
                ])

            pygame_svg_shim.draw.svg_class = ""

    if RANDOM_DBG_RENDER:
        draw_vertical_line(surf, (255, 0, 255), 120+offset, 0, 500, 2, blocked_ranges, arc_buffer=arc_buffer)

    if x_coordinates0 is not None:
        for node in nodes:
            centering_offset = 0 if outline_linear_segments else (node.linear_segment.width - node.width) / 2
            top_y = y_values[node.depth]
            left_x = x_coordinates0[node.name] + centering_offset

            pygame.draw.rect(surf, (0, 0, 0), (left_x-2, top_y-2, node.width+4, node.height+4), width=6)
            pygame.draw.rect(surf, node.color, (left_x, top_y, node.width, node.height), width=2)

    # draw arcs
    if "pygame_bkp" in globals():
        arcs = []
        non_arcs = []
        for draw_call in surf.draw_calls:
            if isinstance(draw_call, pygame_svg_shim.surface.DrawArc):
                arcs.append(draw_call)
            else:
                non_arcs.append(draw_call)
        surf.draw_calls = non_arcs + arcs
    else:
        for x, y, color, width in arc_buffer:
            pygame.draw.arc(surf, color, (x, y, 14, 14), 3*math.pi/2, math.pi/2, width=width)

    # draw tooltip
    if tooltip is not None:
        tooltip_lines = tooltip.split("\n")
        x = MOUSE_POS[0] + 10
        y = MOUSE_POS[1] + 10
        y0 = y

        interlinear_space = 5
        width = max(FONT.size(line)[0] for line in tooltip_lines)
        height = sum(FONT.size(line)[1] for line in tooltip_lines) + interlinear_space * (len(tooltip_lines) - 1)

        if x + width + 2 > surf.get_width():
            x -= (x + width + 2 - surf.get_width())

        pygame.draw.rect(surf, (255, 255, 255), (x, y, width, height))

        for line in tooltip_lines:
            tooltip_surf = FONT.render(line, True, (0, 0, 0))
            tooltip_rect = tooltip_surf.get_rect(topleft=(x, y))
            surf.blit(tooltip_surf, tooltip_rect)
            y += tooltip_rect.h + interlinear_space

        pygame.draw.rect(surf, (0, 0, 0), (x-2, y0-2, width+4, height+4), width=2)


def example_cfg(n: int|str, prefix: str = "example_") -> Node:
    with open(f"{prefix}{n}.json") as f:
        data = json.load(f)

    raw_nodes: list[dict[str, str]] = data["nodes"]
    raw_edges: list[dict[str, str]] = data["edges"]

    nodes: dict[str, Node] = {
        data["name"]: Node(
            data["name"],
            decode_hex_color(data["color"]),
            lines=data.get("lines", None),
            pin_to_bottom=data.get("pin_to_bottom", False)
        )
        for data in raw_nodes
    }
    for edge in raw_edges:
        nodes[edge["from"]].add_child(nodes[edge["to"]], edge["type"])

    for node in nodes.values():
        for edge in node.children:
            if edge.source is edge.target:
                raise AssertionError("Self-loops are not allowed")

    if "root" in data:
        return nodes[data["root"]]
    else:
        return nodes["root"]


def get_all_nodes(root: Node) -> list[Node]:
    all_nodes: list[Node] = []
    todo: list[Node] = [root]
    seen: set[Node] = set()
    while len(todo) > 0:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        all_nodes.append(current)
        for edge in current.children:
            todo.append(edge.target)
    return all_nodes


def dfs_cycle_break(node: Node):
    node.dfs_active = True
    node.dfs_visited = True

    for edge in node.children.copy():
        child = edge.target
        if child.dfs_active:
            print(f"Reversing edge: {edge}")
            edge.reverse_in_place()
        elif not child.dfs_visited:
            dfs_cycle_break(child)
    node.dfs_active = False


def assign_to_layers(nodes: list[Node]) -> list[list[Node]]:
    pinned_nodes = [node for node in nodes if node.pin_to_bottom]
    nodes = [node for node in nodes if not node.pin_to_bottom]

    assigned_count = 0
    layers: list[list[Node]] = [[]]

    for node in nodes:
        if node.atl_assigned:
            continue

        if len(node.parents) > 0:
            continue
        layers[0].append(node)
        node.atl_assigned = True
        node.layer = 0
        assigned_count += 1

    while assigned_count < len(nodes):
        layers.append([])

        selection: list[Node] = [node for node in nodes
                                if (not node.atl_assigned) and all(edge.source.atl_assigned or is_loop_type(edge.typ) for edge in node.non_self_parents)]
        for node in selection:
            #if node.name == "D" and len(layers) <= 3:
            #    continue
            node.atl_assigned = True
            layers[-1].append(node)
            node.layer = len(layers) - 1
            assigned_count += 1

    if len(pinned_nodes) > 0:
        layers.append([])
        for node in pinned_nodes:
            node.atl_assigned = True
            layers[-1].append(node)
            node.layer = len(layers) - 1

    return layers


def rebuild_layers(all_nodes: list[Node]) -> list[list[Node]]:
    layers: list[list[Node]] = [[] for _ in range(max(node.layer for node in all_nodes) + 1)]
    for node in all_nodes:
        layers[node.layer].append(node)
    return layers


def promote_vertex(node: Node) -> int:
    dummy_diff = 0
    for parent_edge in node.parents:
        parent = parent_edge.source
        if parent.promoted_layer == node.promoted_layer - 1: # parent is directly above us
            dummy_diff = dummy_diff + promote_vertex(parent)
    node.promote_count += 1
    dummy_diff = dummy_diff - len(node.parents) + len(node.children)
    return dummy_diff


def promote_vertices_heuristic(all_nodes: list[Node]):
    """Note: layer list must be recalculated after this operation"""

    # simple way to apply to all nodes, list() is needed because map is lazy
    list(map(Node.backup_promote_count, all_nodes))

    promotions = -1
    while promotions != 0:
        promotions = 0
        for node in all_nodes:
            if node.pin_to_bottom:
                continue
            if len(node.parents) > 0:
                if promote_vertex(node) < 0: # decrease in dummy nodes from promoting
                    promotions += 1
                    list(map(Node.apply_promote_count, all_nodes))
                else:
                    list(map(Node.restore_promote_count, all_nodes))


def insert_dummy_nodes(all_nodes: list[Node], layers: list[list[Node]]):
    for node in all_nodes:
        for edge in node.children:
            edge.insert_dummy_nodes(layers)


def index_layer(layer: list[Node]) -> list[Node]:
    """Returns input layers as convenience, it does not create a copy"""
    for i, node in enumerate(layer):
        node.layer_index = i
    return layer


def delete_from(l: list[T], indexes: list[int]):
    """Operates 'in-place' (it does create a new temporary list)"""
    tmp = [v for i, v in enumerate(l) if i not in indexes]
    l.clear()
    l.extend(tmp)


# algorithm from: https://i11www.iti.kit.edu/_media/en/members/tamara_mchedlidze/masterarbeit_jonathan_klawitter.pdf Page 20
# (MA Algorithms for crossing minimization in book drawings, Jonathan Klawitter)
def intersection_count(top: list[Node], bottom: list[Node], top_layer_idx: int, bottom_layer_idx: int) -> int:
    # create a sort of back-and-forth flattened indexing
    #   A   B   C   D
    #  / \ / \ /
    # a   b   c
    # becomes aAbBcC_D
    n = max(len(top)*2 + 1, len(bottom)*2)

    # Clear auxiliary adjacency maps
    list(map(lambda node: node.connect_aux_adjacent.clear(), top + bottom))

    for i in range(n, -1, -1):
        if i % 2 == 1:
            idx = (i - 1) // 2
            if idx >= len(top):
                continue
            w = top[idx]
        else:
            idx = i // 2
            if idx >= len(bottom):
                continue
            w = bottom[idx]
        for edge in w.children + w.parents:
            if edge.source is edge.target:
                continue
            if edge.crossing_endpoint == w:
                edge.crossing_startpoint.connect_aux_adjacent.insert(0, edge)

    count = 0

    # these should be double-linked lists
    ul: list[Node] = []
    ll: list[Node] = []

    last_occurrence: list[int|None] = [None for _ in range(n)]

    for i in range(n):
        if i % 2 == 1:
            idx = (i-1) // 2
            if idx >= len(top):
                continue

            w = top[idx]

            k1 = k2 = k3 = 0
            if last_occurrence[i] is not None:
                if last_occurrence[i] < 0:
                    print("Warning: last_occurrence[i] < 0")
                to_del: list[int] = []
                for vi in range(last_occurrence[i]+1):
                    v = ul[vi]
                    if v == w:
                        k1 += 1
                        k3 += k2
                        to_del.append(vi)
                    else:
                        k2 += 1

                before_len = len(ul)
                delete_from(ul, to_del)
                shift_by = before_len - len(ul)
                for j in range(1, n, 2):
                    if last_occurrence[j] is not None:
                        last_occurrence[j] -= shift_by
                        if last_occurrence[j] < 0:
                            last_occurrence[j] = None

                count += k1*len(ll) + k3

            for e in w.connect_aux_adjacent:#all_edges_e_with_start_point(w).in_order_according_ord_of_end_points():
                e: Edge
                w_prime = e.crossing_endpoint
                # Ord(w) < Ord(w')
                ll.append(w_prime)
                last_occurrence[2*w_prime.layer_index + (1 if w_prime.layer == top_layer_idx else 0)] = len(ll) - 1

        else:
            idx = i // 2
            if idx >= len(bottom):
                continue

            w = bottom[idx]

            k1 = k2 = k3 = 0
            if last_occurrence[i] is not None:
                if last_occurrence[i] < 0:
                    print("Warning: last_occurrence[i] < 0")
                to_del: list[int] = []
                for vi in range(last_occurrence[i]+1):
                    v = ll[vi]
                    if v == w:
                        k1 += 1
                        k3 += k2
                        to_del.append(vi)
                    else:
                        k2 += 1

                before_len = len(ll)
                delete_from(ll, to_del)
                shift_by = before_len - len(ll)
                for j in range(0, n, 2):
                    if last_occurrence[j] is not None:
                        last_occurrence[j] -= shift_by
                        if last_occurrence[j] < 0:
                            last_occurrence[j] = None

                count += k1*len(ul) + k3

            for e in w.connect_aux_adjacent:#all_edges_e_with_start_point(w).in_order_according_ord_of_end_points():
                e: Edge
                w_prime = e.crossing_endpoint
                # Ord(w) < Ord(w')
                ul.append(w_prime)
                last_occurrence[2*w_prime.layer_index + (1 if w_prime.layer == top_layer_idx else 0)] = len(ul) - 1

    return count


def layer_by_layer_sweep(layers: list[list[Node]]):
    iters_since_improvement = 0
    iters = 0
    last_crossings = sum(intersection_count(layers[i], layers[i+1], i, i+1) for i in range(len(layers)-1))
    while True:
        # down sweep
        for i in range(1, len(layers)): # second layer down to last layer
            bottom = layers[i]
            for node in bottom:
                if len(node.parents) == 0:
                    node.uncross_layer_weight = 0
                else:
                    node.uncross_layer_weight = (sum(edge.source.layer_index for edge in node.parents) / len(node.parents))# + (random.random() * 2 - 1) * 0.001
            # sort primarily by uncrossing weight, secondarily by pre-existing layer index
            bottom.sort(key=lambda n: (n.uncross_layer_weight, n.layer_index))

            # remove ibeds in order to fix their indexes
            layers[i] = bottom = [node for node in bottom if not node.is_ibed]
            index_layer(bottom)

            # add ibeds back in
            for j in range(len(bottom)-1, -1, -1):
                j0 = j
                for ibed in reversed(bottom[j].ibeds):
                    bottom.insert(j, ibed)
                    j0 += 1

            # reindex bottom layer
            index_layer(bottom)

        # up sweep
        for i in range(len(layers)-2, -1, -1): # second-to-last up to first layer
            top = layers[i]
            for node in top:
                if len(node.children) == 0:
                    # no children, use down sweep weight instead (based on nodes above)
                    if len(node.parents) == 0:
                        # there are no parent nodes available, we'll just go with 0 then
                        # todo check whether current index works better
                        node.uncross_layer_weight = 0
                    else:
                        node.uncross_layer_weight = sum(edge.source.layer_index for edge in node.parents) / len(node.parents)
                else:
                    node.uncross_layer_weight = sum(edge.target.layer_index for edge in node.children) / len(node.children)
            # sort primarily by uncrossing weight, secondarily by pre-existing layer index
            top.sort(key=lambda n: (n.uncross_layer_weight, n.layer_index))

            # remove ibeds in order to fix their indexes
            layers[i] = top = [node for node in top if not node.is_ibed]
            index_layer(top)

            # add ibeds back in
            for j in range(len(top)-1, -1, -1):
                j0 = j
                for ibed in reversed(top[j].ibeds):
                    top.insert(j, ibed)
                    j0 += 1

            # reindex top layer
            index_layer(top)

        # check for improvement
        crossings = sum(intersection_count(layers[i], layers[i+1], i, i+1) for i in range(len(layers)-1))
        if crossings < last_crossings:
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1
            if iters_since_improvement > 5:
                print("Stopping sweep at iteration", iters)
                break
        last_crossings = crossings
        iters += 1


def crossings_above_and_below(layers: list[list[Node]], i: int) -> int:
    crossings = 0
    if i > 0:
        crossings += intersection_count(layers[i - 1], layers[i], i - 1, i)
    if i < len(layers) - 1:
        crossings += intersection_count(layers[i], layers[i + 1], i, i + 1)
    return crossings


def single_layer_transpose(layers: list[list[Node]], i: int) -> bool:
    did_improve = False
    layer = layers[i]

    # todo proper ibed support
    if any(node.is_ibed for node in layer):
        return False

    for j in range(len(layer) - 1):
        crossings_before = crossings_above_and_below(layers, i)

        # swap
        layer[j], layer[j + 1] = layer[j + 1], layer[j]
        layer[j].layer_index, layer[j + 1].layer_index = layer[j + 1].layer_index, layer[j].layer_index

        crossings_after = crossings_above_and_below(layers, i)

        if crossings_after < crossings_before:
            did_improve = True
        elif crossings_after > crossings_before: # swap back
            layer[j], layer[j + 1] = layer[j + 1], layer[j]
            layer[j].layer_index, layer[j + 1].layer_index = layer[j + 1].layer_index, layer[j].layer_index

    return did_improve

def layer_by_layer_transpose_sweep(layers: list[list[Node]]):
    did_improve = True
    while did_improve:
        did_improve = False

        # down sweep
        for i in range(0, len(layers)):
            did_improve |= single_layer_transpose(layers, i)

        # up sweep
        for i in range(len(layers)-1, -1, -1):
            did_improve |= single_layer_transpose(layers, i)


def assign_to_linear_segments(layers: list[list[Node]]) -> list[LinearSegment]:
    rooter: Node|None = None
    if len(layers[0]) > 1:
        # must insert rooter node
        rooter = Node(
            "rooter", (255, 255, 0), depth=-1
        )
        rooter.forced_depth = -1
        rooter.layer = -1
        for node in layers[0]:
            rooter.children.append(Edge(
                rooter,
                node,
                "direct",
                False
            ))
        layers.insert(0, [rooter])
    assert len(layers[0]) == 1, "Root node must be the only node in the first layer"

    all_segments: list[LinearSegment] = []

    current = layers[0][0]
    current.linear_segment = LinearSegment(current)
    all_segments.append(current.linear_segment)

    todo: list[Node] = []

    for i, layer in enumerate(layers):
        if i == 0 or (i == 1 and rooter is not None):
            continue
        for node in layer:
            if len(node.parents) == 0:
                ls = LinearSegment(node)
                all_segments.append(ls)
                node.linear_segment = ls
                todo.append(node)

    def new_segment(ch: 'Edge'):
        if ch.target.linear_segment is not None:
            ch.target.linear_segment.parents.append(current.linear_segment)
            current.linear_segment.children.append(ch.target.linear_segment)
        else:
            ls = LinearSegment(ch.target)
            all_segments.append(ls)
            current.linear_segment.children.append(ls)
            ls.parents.append(current.linear_segment)
            ch.target.linear_segment = ls
            todo.append(ch.target)

    while (current is not None and len(current.children) > 0) or len(todo) > 0:
        if current is None:
            current = todo.pop()
        if len(current.children) == 1:
            if len(current.children[0].target.parents) == 1:
                if not ((current.is_dummy == current.children[0].target.is_dummy) or current.is_ibed or current.children[0].target.is_obed):
                    new_segment(current.children[0])
                    current = None
                elif current.is_dummy and DUMMY_LINEAR_SEGMENT_MAX_LENGTH is not None and len(current.linear_segment.nodes) >= DUMMY_LINEAR_SEGMENT_MAX_LENGTH:
                    new_segment(current.children[0])
                    current = None
                else:
                    current.linear_segment.nodes.append(current.children[0].target)
                    current.children[0].target.linear_segment = current.linear_segment
                    current = current.children[0].target
            else:
                child = current.children[0]
                new_segment(child)
                current = None
        elif len(current.children) > 1:
            for child in current.children:
                new_segment(child)

            current = None
        else:
            current = None

    if rooter is not None:
        rooter_ls = all_segments.pop(0)
        assert len(rooter_ls.nodes) == 1 and rooter_ls.nodes[0] is rooter
        for child in rooter_ls.children:
            child.parents.remove(rooter_ls)
        layers.pop(0)

    list(map(LinearSegment.recalculate, all_segments))
    return all_segments


def stratize_segments(all_segments: list[LinearSegment], custom_key = None) -> tuple[list[list[LinearSegment]], list[list[LinearSegment]]]:
    # each layer is a list of segments present on each layer, sorted by max_index or custom value, depending on `custom_key`
    by_layer: list[list[LinearSegment]] = [[] for _ in range(max(segment.max_layer for segment in all_segments) + 1)]

    # each index is a list of segments, in arbitrary order
    by_max_index: list[list[LinearSegment]] = [[] for _ in range(max(segment.ranking_index for segment in all_segments) + 1)]

    for segment in all_segments:
        for layer in range(segment.min_layer, segment.max_layer + 1):
            by_layer[layer].append(segment)
        by_max_index[segment.ranking_index].append(segment)

    # sort by max_index or custom value, depending on `custom_key`
    for i, layer in enumerate(by_layer):
        layer.sort(key=lambda segment: custom_key(segment) if custom_key is not None else segment.max_index)

    # sort by min_layer
    for column in by_max_index:
        column.sort(key=lambda segment: segment.min_layer)

    return by_layer, by_max_index


def assign_coordinates(all_segments: list[LinearSegment]):

    for segment in all_segments:
        segment.x = 0.0
        segment.assigned_coords = False

    by_layer, by_max_index = stratize_segments(all_segments, custom_key=lambda s: s.ranking_index)

    # start assigning coordinates from the lowest max_index to the highest
    for column in by_max_index:
        for segment in column:
            # assign the left-most coordinate possible
            # basically, go through each layer covered by this segment and for all lefter-indexed segments in that layer, max their right edge with x
            x = 0.0
            for layer in range(segment.min_layer, segment.max_layer + 1):
                for other_segment in by_layer[layer]:
                    if other_segment.ranking_index <= segment.ranking_index:
                        if other_segment.assigned_coords:
                            x = max(x, other_segment.x + other_segment.width)
                    #else:
                    #    break
            segment.x = x
            segment.assigned_coords = True


class Region:
    def __init__(self, *initial_members: LinearSegment):
        self.segments: list[LinearSegment] = [*initial_members]
        self._x: float = min(segment.x for segment in self.segments)
        self._width: float = max(segment.x+segment.width for segment in self.segments) - self._x
        self._removed: bool = False

        self.actual_applied_force: float|None = None

    def _recalculate(self):
        self._x = min(segment.x for segment in self.segments)
        self._width = max(segment.x+segment.width for segment in self.segments) - self._x

    def merge_in(self, right: 'Region'):
        self.segments.extend(right.segments)
        right._removed = True
        self.actual_applied_force = None
        self._recalculate()

    def apply_movement(self, movement: float):
        for segment in self.segments:
            segment.x += movement
        self._recalculate()

    def get_movement_limit(self, movement: float, by_layer: list[list[LinearSegment]], current_layer: int,
                           ignore: list[LinearSegment], leftward: bool) -> float:
        limit = float("-inf") if leftward else float("inf")

        for segment in self.segments:
            for layer in range(segment.min_layer, segment.max_layer + 1):
                if layer == current_layer:
                    continue

                for other_segment in by_layer[layer]:
                    if other_segment in self.segments:
                        continue
                    if other_segment.min_layer <= current_layer <= other_segment.max_layer:
                        continue
                    if other_segment in ignore:
                        continue

                    if movement < 0:
                        if other_segment.x < segment.x:
                            """
                            #---#
                            |o_s|
                            |   |  #-------#
                            |   |  |segment|
                            |   |  #-------#
                            #---#
                            """
                            limit_for_segment = other_segment.x + other_segment.width
                            limit_for_region = limit_for_segment - segment.x + self.x
                            if segment.x + movement < limit_for_segment:
                                if leftward:
                                    limit = max(limit, limit_for_region)
                                else:
                                    limit = min(limit, limit_for_region)
                    else:
                        if other_segment.x > segment.x:
                            limit_for_segment = other_segment.x - segment.width
                            limit_for_region = limit_for_segment - segment.x + self.x
                            if segment.x + movement + segment.width > other_segment.x:
                                if leftward:
                                    limit = max(limit, limit_for_region)
                                else:
                                    limit = min(limit, limit_for_region)

        return limit

    def _get_first_blocking_segment_deprecated(self, movement: float, by_layer: list[list[LinearSegment]], current_layer: int, ignore: list[LinearSegment]) -> LinearSegment | None:
        for segment in self.segments:
            # go through all layers other than the current layer
            for layer in range(segment.min_layer, segment.max_layer + 1):
                if layer == current_layer:
                    continue

                for other_segment in by_layer[layer]:
                    if other_segment in self.segments:
                        continue
                    if other_segment.min_layer <= current_layer <= other_segment.max_layer:
                        continue
                    if other_segment in ignore:
                        continue

                    if other_segment.x < segment.x:
                        if segment.x < other_segment.x + other_segment.width and abs(segment.x - (other_segment.x+other_segment.width)) > 1e-10:
                            print("OH NO 1: ", movement, segment, other_segment, "diff: ", segment.x - (other_segment.x+other_segment.width))
                        if segment.x + movement < other_segment.x + other_segment.width:
                            return other_segment
                    else:
                        if segment.x + segment.width > other_segment.x and abs(segment.x + segment.width - other_segment.x) > 1e-10:
                            print("OH NO 2: ", movement, segment, other_segment, "diff: ", segment.x + segment.width - other_segment.x)
                        if segment.x + movement + segment.width > other_segment.x:
                            return other_segment

        return None

    def apply_movement_or_store(self, movement: float, by_layer: list[list[LinearSegment]], downsweep: bool,
                                current_layer: int, skip_segments_crossing_layer: bool = False) -> bool:
        """Checks lower layers for conflicts, and stores the movement for later if there are any

        :return: True if the movement was applied, False if it was stored
        """
        ok = True

        for segment in self.segments:
            # go through all layers other than the current layer
            for layer in range(segment.min_layer, segment.max_layer + 1):
                if downsweep:
                    if layer == current_layer: # was segment.min_layer
                        continue
                else:
                    if layer == current_layer: # was segment.max_layer
                        continue

                for other_segment in by_layer[layer]:
                    if other_segment in self.segments:
                        continue

                    # such segments should already be caught by region checks
                    if other_segment.min_layer <= current_layer <= other_segment.max_layer and skip_segments_crossing_layer:
                        continue

                    if other_segment.x < segment.x:
                        log = False
                        if segment.x < other_segment.x + other_segment.width and abs(segment.x - (other_segment.x+other_segment.width)) > 1e-10:
                            log = True
                            print("OH NO 1: ", movement, segment, other_segment, "diff: ", segment.x - (other_segment.x+other_segment.width))
                        if (other_segment.x + other_segment.width) - (segment.x + movement) > 1e-10:#segment.x + movement < other_segment.x + other_segment.width:
                            ##if other_segment.min_layer <= current_layer <= other_segment.max_layer and skip_segments_crossing_layer and GLOBAL_BREAKPOINT_INDEX == 4:
                            ##    _ = 0
                            if log: print("Broke")
                            ok = False

                            if GLOBAL_BREAKPOINT_INDEX == 4:
                                print(f"\tA> {segment} hit {other_segment} on layer {layer} with overlap of {(other_segment.x + other_segment.width) - (segment.x + movement)}")

                            break
                        else:
                            if log: print("didn't break")
                    else:
                        log = False
                        if segment.x + segment.width > other_segment.x and abs(segment.x + segment.width - other_segment.x) > 1e-10:
                            log = True
                            print("OH NO 2: ", movement, segment, other_segment, "diff: ", segment.x + segment.width - other_segment.x)
                        if (segment.x + movement + segment.width) - other_segment.x > 1e-10:#segment.x + movement + segment.width > other_segment.x:
                            ##if other_segment.min_layer <= current_layer <= other_segment.max_layer and skip_segments_crossing_layer and GLOBAL_BREAKPOINT_INDEX == 4:
                            ##    _ = 0
                            if log: print("Broke")
                            ok = False

                            if GLOBAL_BREAKPOINT_INDEX == 4:
                                print(f"\tB> {segment} hit {other_segment} on layer {layer} with overlap of {(segment.x + movement + segment.width) - other_segment.x}")

                            break
                        else:
                            if log: print("didn't break")
                if not ok:
                    break
            if not ok:
                break

        if ok:
            self.apply_movement(movement)
            return True
        else:
            for segment in self.segments:
                if LOG_STORED_MOVEMENT:
                    print(f"Storing movement {movement} for segment {segment}")
                    print("Before:", segment.pendulum_force_from_parent_layer)
                segment.pendulum_force_from_parent_layer += movement
            return False

    @property
    def x(self) -> float: return self._x

    @property
    def width(self) -> float: return self._width

    @property
    def removed(self) -> bool: return self._removed

    def __str__(self):
        return f"Region({', '.join(segment.nodes[0].name for segment in self.segments)})"

    def __repr__(self):
        return str(self)

    def applied_force_from_above(self, layer: int) -> float:
        return avg(segment.force_from_above() if segment.min_layer == layer else segment.pendulum_force_from_parent_layer/10 for segment in self.segments)

    def applied_force_from_below(self, layer: int) -> float:
        return avg(segment.force_from_below() if segment.max_layer == layer else segment.pendulum_force_from_parent_layer/10 for segment in self.segments)


def check_overlap(all_segments: list[LinearSegment]) -> bool:
    by_layer, _ = stratize_segments(all_segments, custom_key=lambda s: s.x)
    for layer in by_layer:
        for i, segment in enumerate(layer):
            for other_segment in layer[i+1:]:
                if segment.x < other_segment.x + other_segment.width and segment.x + segment.width > other_segment.x:
                    overlap_amount = min(segment.x + segment.width, other_segment.x + other_segment.width)\
                                        - max(segment.x, other_segment.x)
                    if abs(overlap_amount) < 1e-10:
                        continue
                    print(f"Overlap detected: {segment} and {other_segment}")
                    print(f"\t Overlap amount: {overlap_amount}")
                    return True

    return False


def downsweep_pendulum(all_segments: list[LinearSegment], only_layer: int|None = None):
    sweep_pendulum_internal(all_segments, True, only_layer)

def upsweep_pendulum(all_segments: list[LinearSegment], only_layer: int|None = None):
    sweep_pendulum_internal(all_segments, False, only_layer)

def sweep_pendulum_internal(all_segments: list[LinearSegment], downsweep: bool, only_layer: int | None = None):
    global GLOBAL_BREAKPOINT_INDEX
    for segment in all_segments:
        segment.pendulum_force_from_parent_layer = 0.0

    by_layer, _ = stratize_segments(all_segments, custom_key=lambda s: s.x)

    for i in (range(1, len(by_layer)) if downsweep else range(len(by_layer)-2, -1, -1)):
        if only_layer is not None and i != only_layer:
            continue
        print(f"\nStart of swinging layer {i} from {'above' if downsweep else 'below'}...")

        # check if any segments start on this layer
        if downsweep:
            if not any(segment.min_layer == i for segment in all_segments):
                print("Skipping layer (no segments start here)")
                continue
        else:
            if not any(segment.max_layer == i for segment in all_segments):
                print("Skipping layer (no segments end here)")
                continue

        GLOBAL_BREAKPOINT_INDEX += 1
        if GLOBAL_BREAKPOINT_INDEX == 15:
            print("Breakpoint reached")
            do_raise_ = False
            if do_raise_:
                raise StopBalance("Stopping balance to avoid bug")

        regions: list[Region] = [Region(v) for v in by_layer[i]]

        for region in regions:
            region.actual_applied_force = None

        # combine regions as needed
        did_combine = True
        while did_combine:
            did_combine = False
            new_regions: list[Region] = []
            if len(regions) <= 1:
                break
            for j in range(len(regions)-1):
                my_region = regions[j]
                next_region = regions[j+1]

                my_target_force = my_region.applied_force_from_above(i) if downsweep else my_region.applied_force_from_below(i)
                next_target_force = next_region.applied_force_from_above(i) if downsweep else next_region.applied_force_from_below(i)

                # there's a few possible cases here
                # 1: The regions are already touching
                # 2: The regions are not already touching, but they will be if we move them AND it is actually possible for us to move
                # 3 The regions will not touch even if we move them by their desired force

                # case 1
                '''if my_region.x + my_region.width >= next_region.x and False: # not actually a useful metric, it just prevents movement
                    my_region.merge_in(next_region)
                    new_regions.append(my_region)
                    did_combine = True
                    # add remaining regions to new_regions
                    new_regions.extend(regions[j+2:])
                    break # just the inner loop'''
                # case 2
                ###my_blocker = my_region.get_first_blocking_segment(my_target_force, by_layer, i, next_region.segments)
                ###next_blocker = next_region.get_first_blocking_segment(next_target_force, by_layer, i, my_region.segments)

                my_hypothetical_x = my_region.x + my_target_force
                my_rightward_limit = my_region.get_movement_limit(my_target_force, by_layer, i, next_region.segments, False)#float("inf")
                my_leftward_limit = my_region.get_movement_limit(my_target_force, by_layer, i, next_region.segments, True)#float("-inf")
                #if my_blocker is not None:
                #    if my_target_force > 0:
                #        my_rightward_limit = my_blocker.x - my_region.width # this is completely wrong, since the blocker doesn't necessarily block the entire region
                #        my_hypothetical_x = min(my_hypothetical_x, my_rightward_limit)
                #    else:
                #        my_hypothetical_x = max(my_hypothetical_x, my_blocker.x + my_blocker.width)
                if my_target_force > 0 and my_rightward_limit != float("inf"):
                    my_hypothetical_x = min(my_hypothetical_x, my_rightward_limit)
                    my_region.actual_applied_force = my_rightward_limit - my_region.x
                elif my_target_force < 0 and my_leftward_limit != float("-inf"):
                    my_hypothetical_x = max(my_hypothetical_x, my_leftward_limit)
                    my_region.actual_applied_force = my_leftward_limit - my_region.x

                next_hypothetical_x = next_region.x + next_target_force
                next_leftward_limit = next_region.get_movement_limit(next_target_force, by_layer, i, my_region.segments, True)#float("-inf")
                next_rightward_limit = next_region.get_movement_limit(next_target_force, by_layer, i, my_region.segments, False)#float("inf")
                #if next_blocker is not None:
                #    if next_target_force > 0:
                #        next_hypothetical_x = min(next_hypothetical_x, next_blocker.x - next_region.width)
                #    else:
                #        next_leftward_limit = next_blocker.x + next_blocker.width
                #        next_hypothetical_x = max(next_hypothetical_x, next_leftward_limit)
                if next_target_force > 0 and next_rightward_limit != float("inf"):
                    next_hypothetical_x = min(next_hypothetical_x, next_rightward_limit)
                    next_region.actual_applied_force = next_rightward_limit - next_region.x
                elif next_target_force < 0 and next_leftward_limit != float("-inf"):
                    next_hypothetical_x = max(next_hypothetical_x, next_leftward_limit)
                    next_region.actual_applied_force = next_leftward_limit - next_region.x

                if my_hypothetical_x + my_region.width > next_hypothetical_x:
                    # we need to move the regions towards each other
                    # 3 possible situations here
                    # ================================
                    # 1.
                    # me   :    |______>
                    # next : <______|
                    # detection: my_target_force * next_target_force < 0 (opposite signs)
                    # movement: move both regions to the weighted center (based on the forces)
                    # ================================
                    # 2.
                    # me   : |______>
                    # next :    |_>
                    # detection: my_target_force > 0
                    # movement: move me to the right
                    # ================================
                    # 3.
                    # me   :   <_|
                    # next : <______|
                    # detection: my_target_force < 0
                    # movement: move next to the left
                    # ================================
                    # Constraints:
                    # > me is only allowed to move right (+ force)
                    # > next is only allowed to move left (- force)

                    # case 1
                    if my_target_force * next_target_force < 0:
                        if my_target_force < 0:
                            raise AssertionError("For some reason the forces are moving apart, and yet the regions are colliding")
                        total_force = abs(my_target_force) + abs(next_target_force)

                        me_percent = abs(my_target_force) / total_force
                        next_percent = 1 - me_percent

                        total_movement = next_region.x - (my_region.x+my_region.width)

                        if abs(total_movement) < 1e-10:
                            total_movement = 0.0

                        if total_movement != 0.0:
                            if my_region.x + (total_movement * me_percent) > my_rightward_limit:
                                me_percent = (my_rightward_limit - my_region.x) / total_movement

                            if next_region.x - (total_movement * next_percent) < next_leftward_limit:
                                next_percent = (next_region.x - next_leftward_limit) / total_movement

                            if abs(me_percent) < 1e-10:
                                me_percent = 0.0
                            if abs(next_percent) < 1e-10:
                                next_percent = 0.0

                        print("Movement case 1")
                        if total_movement < -1e-10: # floating point precision my beloathed
                            print("Warning: negative movement")
                        total_movement = max(0.0, total_movement)
                        assert total_movement >= 0
                        if not (0 <= me_percent <= 1) or not (0 <= next_percent <= 1):
                            print("Oops")
                        me_percent = max(0.0, min(1.0, me_percent))
                        next_percent = max(0.0, min(1.0, next_percent))
                        assert 0 <= me_percent <= 1
                        print("> Moving me")
                        my_region.apply_movement_or_store(total_movement * me_percent, by_layer, downsweep, i)
                        print("> Moving next")
                        next_region.apply_movement_or_store(-total_movement * next_percent, by_layer, downsweep, i)
                    # case 2
                    elif my_target_force > 0:
                        movement = next_region.x - (my_region.x+my_region.width)

                        if my_region.x + movement > my_rightward_limit:
                            movement = my_rightward_limit - my_region.x

                        print("Movement case 2")
                        if movement < -1e-10:
                            print("Warning: negative movement")
                        movement = max(0.0, movement)
                        assert movement >= 0
                        my_region.apply_movement_or_store(movement, by_layer, downsweep, i)
                    # case 3
                    else:
                        movement = next_region.x - (my_region.x+my_region.width)

                        if next_region.x - movement < next_leftward_limit:
                            movement = next_region.x - next_leftward_limit

                        print("Movement case 3")
                        if movement < -1e-10:
                            print("Warning: negative movement")
                        movement = max(0.0, movement)
                        assert movement >= 0
                        next_region.apply_movement_or_store(-movement, by_layer, downsweep, i)

                    my_region.merge_in(next_region)
                    new_regions.append(my_region)
                    did_combine = True
                    # add remaining regions to new_regions
                    new_regions.extend(regions[j+2:])
                    break
                # case 3
                else:
                    new_regions.append(my_region)
                    if j == len(regions) - 2:
                        new_regions.append(next_region)

            regions = new_regions

        print("Regions:")
        oks = []
        forces = []
        for region in regions:
            force = region.applied_force_from_above(i) if downsweep else region.applied_force_from_below(i)
            print(f"\t{region} | Combined force: {force} | Actual applied force: {region.actual_applied_force}")
            if region.actual_applied_force is not None:
                force = region.actual_applied_force
            # apply force
            #print("Applying force")
            oks.append(region.apply_movement_or_store(force, by_layer, downsweep, i, skip_segments_crossing_layer=True))
            forces.append(force)

        info_tbl = [
            [str(region) for region in regions],
            [str(force) for force in forces],
            [str(ok) for ok in oks]
        ]
        info_tbl[0].insert(0, "Region")
        info_tbl[1].insert(0, "Force")
        info_tbl[2].insert(0, "OK")

        if check_overlap(all_segments):
            print(f"{GLOBAL_BREAKPOINT_INDEX=}")
            table_print(info_tbl)
            print("Overlap detected after pendulum sweep")
            raise StopBalance("Overlap detected after pendulum sweep")

        if all(oks) or all(not ok for ok in oks):
            print("Coherent application of forces")
        else:
            do_raise = False # just to allow debugger to skip this
            a = 0
            if do_raise:
                print(f"{GLOBAL_BREAKPOINT_INDEX=}")
                table_print(info_tbl)
                raise StopBalance("Incoherent application of forces")


def normalize_segment_positions(all_segments: list[LinearSegment]):
    min_x = min(segment.x for segment in all_segments)
    print("Min x:", min_x)
    for segment in all_segments:
        segment.x -= min_x


def pendulum_balance(all_segments: list[LinearSegment]):
    sum_of_forces = abs(sum(segment.force_from_above()+segment.force_from_below() for segment in all_segments))
    iters_since_improvement = 0
    iters = 0
    while True:
        iters += 1
        downsweep_pendulum(all_segments)
        #normalize_segment_positions(all_segments)

        for _ in range(2):
            upsweep_pendulum(all_segments)
        #normalize_segment_positions(all_segments)

        new_sum_of_forces = abs(sum(segment.force_from_above()+segment.force_from_below() for segment in all_segments))
        if new_sum_of_forces >= sum_of_forces:
            iters_since_improvement += 1
        else:
            iters_since_improvement = 0
        if iters_since_improvement > 1 or iters > 200:
            print(f"Stopping pendulum balance after {iters} iters")
            break
        sum_of_forces = new_sum_of_forces

    normalize_segment_positions(all_segments)


def rank_segments(all_segments: list[LinearSegment]):
    # initialize ranking indexes to max_index
    for segment in all_segments:
        segment.ranking_index = segment.max_index

    by_layer, _ = stratize_segments(all_segments)

    changed = True
    while changed:
        changed = False

        for layer in by_layer:
            for i in range(1, len(layer)):
                a = layer[i-1]
                b = layer[i]

                if a.ranking_index >= b.ranking_index:
                    b.ranking_index = a.ranking_index + 1
                    changed = True


def generate_adjacent_segments(all_segments: list[LinearSegment]) -> list[tuple[LinearSegment, LinearSegment, tuple[int, int]]]:
    out: list[tuple[LinearSegment, LinearSegment, tuple[int, int]]] = []

    for i in range(len(all_segments)):
        for j in range(i+1, len(all_segments)):
            a = all_segments[i]
            b = all_segments[j]

            if overlap := a.is_next_to(b):
                if a.ranking_index > b.ranking_index:
                    a, b = b, a
                out.append((a, b, overlap))

    def key(dat: tuple[LinearSegment, LinearSegment, tuple[int, int]]):
        a_, b_, _ = dat
        return a_.ranking_index, b_.ranking_index

    out.sort(key=key)

    return out


def single_segment_transpose(a: LinearSegment, b: LinearSegment, overlap: tuple[int, int], layers: list[list[Node]]) -> bool:
    if not a.is_next_to(b):
        print("Segments are not adjacent")
        return False
    o1, o2 = overlap
    crossings_before = crossings_above_and_below(layers, a.min_layer) + crossings_above_and_below(layers, a.max_layer)\
                    + crossings_above_and_below(layers, b.min_layer) + crossings_above_and_below(layers, b.max_layer)

    if crossings_before == 0:
        return False

    # swap
    bkp_a = a.ranking_index
    bkp_b = b.ranking_index
    a.transpose_with(b, layers)

    crossings_after = crossings_above_and_below(layers, a.min_layer) + crossings_above_and_below(layers, a.max_layer) \
                    + crossings_above_and_below(layers, b.min_layer) + crossings_above_and_below(layers, b.max_layer)

    if crossings_after < crossings_before:
        print(f"Successfully swapped {a} and {b} with {crossings_after} crossings (before: {crossings_before})")
        return True
    else:#if crossings_after > crossings_before:  # swap back
        a.transpose_with(b, layers)
        a.ranking_index = bkp_a
        b.ranking_index = bkp_b
    return False


def space_out_ranking_indexes(all_segments: list[LinearSegment]):
    for segment in all_segments:
        segment.ranking_index *= 5


def collapse_ranking_indexes(all_segments: list[LinearSegment]):
    by_layer, _ = stratize_segments(all_segments, custom_key=lambda s: s.ranking_index)

    # initialize ranking indexes to 0
    for segment in all_segments:
        segment.ranking_index = 0

    changed = True
    while changed:
        changed = False

        for layer in by_layer:
            for i in range(1, len(layer)):
                a = layer[i - 1]
                b = layer[i]

                if a.ranking_index >= b.ranking_index:
                    b.ranking_index = a.ranking_index + 1
                    changed = True


def segment_by_segment_transpose_sweep(adjacent_segments: list[tuple[LinearSegment, LinearSegment, tuple[int, int]]], layers: list[list[Node]], all_segments: list[LinearSegment]):
    did_improve_inner = True
    escape = 10
    iters_since_improvement = 0
    while iters_since_improvement < 5 and escape >= 0:

        if adjacent_segments is None:
            index_nodes_by_ranking_index(layers)
            adjacent_segments = generate_adjacent_segments(all_segments)

        did_improve_inner = False
        escape -= 1

        did_improve = True
        while did_improve:
            did_improve = False

            # down sweep
            for a, b, overlap in adjacent_segments:
                # todo proper ibed support
                if any(n.is_ibed for n in a.nodes) or any(n.is_ibed for n in b.nodes):
                    continue
                space_out_ranking_indexes(all_segments)
                did_improve |= single_segment_transpose(a, b, overlap, layers)
                collapse_ranking_indexes(all_segments)

            # up sweep
            for a, b, overlap in reversed(adjacent_segments):
                # todo proper ibed support
                if any(n.is_ibed for n in a.nodes) or any(n.is_ibed for n in b.nodes):
                    continue
                space_out_ranking_indexes(all_segments)
                did_improve |= single_segment_transpose(a, b, overlap, layers)
                collapse_ranking_indexes(all_segments)

            did_improve_inner |= did_improve

        adjacent_segments = None
        if did_improve_inner:
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1

    if escape < 0:
        print("segment-by-segment transpose sweep: repeat limit escape condition reached")
    else:
        print(f"segment-by-segment transpose sweep: iters since improvement {iters_since_improvement}")


def index_nodes_by_ranking_index(layers: list[list[Node]]):
    for i, layer in enumerate(layers):
        def key(node: Node):
            return node.linear_segment.ranking_index, node.layer_index

        before = layer.copy()

        layer.sort(key=key)

        if before != layer:
            print(f"\tLayer {i} changed during indexing")
        index_layer(layer)


class EdgeLike(metaclass=abc.ABCMeta):
    @property
    @abstractmethod
    def edge_start_x(self) -> float: ...

    @property
    @abstractmethod
    def edge_end_x(self) -> float: ...

    @property
    @abstractmethod
    def id_(self) -> str: ...

    @property
    @abstractmethod
    def track_index(self) -> int: ...

    @track_index.setter
    @abstractmethod
    def track_index(self, track_index: int): ...


class EdgeLikeEdge(EdgeLike):
    def __init__(self, edge: Edge):
        self._edge: Edge = edge

    @property
    def edge_start_x(self) -> float:
        return self._edge.edge_start_x

    @property
    def edge_end_x(self) -> float:
        return self._edge.edge_end_x

    @property
    def id_(self) -> str:
        return self._edge.id_

    @property
    def track_index(self) -> int:
        return self._edge.track_index

    @track_index.setter
    def track_index(self, track_index: int):
        self._edge.track_index = track_index

    def __str__(self):
        return str(self._edge)

    def __repr__(self):
        return repr(self._edge)


class ObedEdge(EdgeLike):
    def __init__(self, obed: Node):
        self.obed = obed

    @property
    def edge_start_x(self) -> float:
        return min(self.obed.base_port_x, self.obed.obed_parent.base_port_x)

    @property
    def edge_end_x(self) -> float:
        return max(self.obed.base_port_x, self.obed.obed_parent.base_port_x)

    @property
    def id_(self) -> str:
        return str(self)

    @property
    def track_index(self) -> int:
        return self.obed.obed_edge_track_index

    @track_index.setter
    def track_index(self, track_index: int):
        self.obed.obed_edge_track_index = track_index

    def __str__(self):
        return f"ObedEdge(`{self.obed.name}` -> `{self.obed.obed_parent.name}`"

    def __repr__(self):
        return str(self)


class IbedEdge(EdgeLike):
    def __init__(self, ibed: Node):
        self.ibed = ibed

    @property
    def edge_start_x(self) -> float:
        return min(self.ibed.base_port_x, self.ibed.ibed_parent.ibed_port_x)

    @property
    def edge_end_x(self) -> float:
        return max(self.ibed.base_port_x, self.ibed.ibed_parent.ibed_port_x)

    @property
    def id_(self) -> str:
        return str(self)

    @property
    def track_index(self) -> int:
        return self.ibed.ibed_edge_track_index

    @track_index.setter
    def track_index(self, track_index: int):
        self.ibed.ibed_edge_track_index = track_index

    def __str__(self):
        return f"IbedEdge(`{self.ibed.ibed_parent.name}` -> `{self.ibed.name}`"

    def __repr__(self):
        return str(self)


def nodes_to_edges_timeline(nodes: list[Node]) -> list[EdgeLike]:
    timeline = sum((n.children for n in nodes), start=[])
    return [EdgeLikeEdge(edge) for edge in timeline if edge.edge_start_x != edge.edge_end_x]


def nodes_to_obeds_timeline(nodes: list[Node]) -> list[EdgeLike]:
    return [ObedEdge(n) for n in nodes if n.is_obed]


def nodes_to_ibeds_timeline(nodes: list[Node]) -> list[EdgeLike]:
    return [IbedEdge(n) for n in nodes if n.is_ibed]


def assign_tracks_internal(top: list[Node], timeline_producer: typing.Callable[[list[Node]], list[EdgeLike]],
                           index_offset: int = 0) -> tuple[int, list[RangeSet]]:
    top: list[Node] = sorted(top, key=lambda n: n.base_port_x)

    timeline: list[EdgeLike] = timeline_producer(top)

    # x: (name, is_start)
    events: dict[float, list[tuple[EdgeLike, str, bool]]] = {}

    ordered_events: list[float] = []

    for edge in timeline:
        if edge.edge_start_x not in events:
            events[edge.edge_start_x] = []
        if edge.edge_end_x not in events:
            events[edge.edge_end_x] = []

        events[edge.edge_start_x].append((edge, edge.id_, True))
        events[edge.edge_end_x].append((edge, edge.id_, False))
        ordered_events.append(edge.edge_start_x)
        ordered_events.append(edge.edge_end_x)

    for events_sequence in events.values():
        events_sequence.sort(key=lambda x: (-(x[0].edge_end_x if x[2] else x[0].edge_start_x), x[2]))

    ordered_events = list(set(ordered_events))

    ordered_events.sort()

    # if occupied, have id of edge
    start: dict[str, float] = {}
    track_occupancy: list[str|None] = []
    filled_ranges: list[RangeSet] = []

    for event_x in ordered_events:
        for edge, edge_id, is_start in events[event_x]:
            if is_start:
                # find a free track
                for i, track in enumerate(track_occupancy):
                    if track is None:
                        start[edge_id] = event_x
                        track_occupancy[i] = edge_id
                        edge.track_index = i + index_offset
                        break
                else:
                    start[edge_id] = event_x
                    track_occupancy.append(edge_id)
                    filled_ranges.append(RangeSet())
                    edge.track_index = (len(track_occupancy) - 1) + index_offset
            else:
                # find the track
                for i, track in enumerate(track_occupancy):
                    if track == edge_id:
                        track_occupancy[i] = None
                        filled_ranges[i] += Range(start[edge_id], event_x, False)
                        break
                else:
                    raise AssertionError(f"Track not found for edge {edge_id}")

    return len(track_occupancy), filled_ranges


def assign_tracks(layers: list[list[Node]]) -> tuple[list[int], list[list[RangeSet]]]:
    # todo actual values for leading area
    track_counts: list[int] = []
    range_sets: list[list[RangeSet]] = []

    leading_incoming_track_count, leading_incoming_filled_ranges = assign_tracks_internal(layers[0], nodes_to_ibeds_timeline)
    track_counts.append(leading_incoming_track_count)
    range_sets.append(leading_incoming_filled_ranges)

    for i in range(len(layers)):
        outgoing_track_count, outgoing_filled_ranges = assign_tracks_internal(layers[i], nodes_to_obeds_timeline)
        # invert the track indexes for the obeds
        for node in layers[i]:
            if node.is_obed:
                node.obed_edge_track_index = (outgoing_track_count-1) - node.obed_edge_track_index
        outgoing_filled_ranges.reverse()

        middle_track_count, middle_filled_ranges = assign_tracks_internal(layers[i], nodes_to_edges_timeline, outgoing_track_count)
        # count cross-overs
        def count_cross_overs():
            cross_overs = 0
            for node_ in layers[i]:
                for edge_ in node_.children:
                    if edge_.track_index is None:
                        continue

                    # check if descending end of edge crosses over any ranges under the assigned track
                    for range_idx in range(edge_.track_index - outgoing_track_count):
                        if edge_.source.port_x_for(edge_) in middle_filled_ranges[range_idx]:
                            cross_overs += 1

                    # check if ascending end of edge crosses over any ranges over the assigned track
                    for range_idx in range(edge_.track_index - outgoing_track_count + 1, len(middle_filled_ranges)):
                        if edge_.target.port_x_for(edge_) in middle_filled_ranges[range_idx]:
                            cross_overs += 1
            return cross_overs

        # flip around edge order in middle ranges
        def do_flip():
            nonlocal middle_filled_ranges
            for node_ in layers[i]:
                for edge_ in node_.children:
                    if edge_.track_index is None:
                        continue
                    relative_idx = edge_.track_index - outgoing_track_count
                    flipped_relative_idx = middle_track_count - relative_idx - 1
                    edge_.track_index = flipped_relative_idx + outgoing_track_count
            middle_filled_ranges = list(reversed(middle_filled_ranges))

        pre_flip_cross_overs = count_cross_overs()
        do_flip()
        post_flip_cross_overs = count_cross_overs()
        if post_flip_cross_overs > pre_flip_cross_overs: # if we made it worse, flip back
            print(f"Reversing ineffective flip ({pre_flip_cross_overs} -> {post_flip_cross_overs})")
            do_flip()
        else:
            print(f"Track flip was successful ({pre_flip_cross_overs} -> {post_flip_cross_overs})")

        if i < len(layers) - 1:
            incoming_track_count, incoming_filled_ranges = assign_tracks_internal(layers[i+1], nodes_to_ibeds_timeline, outgoing_track_count+middle_track_count)
        else:
            incoming_track_count, incoming_filled_ranges = 0, []

        track_counts.append(outgoing_track_count + middle_track_count + incoming_track_count)
        range_sets.append(outgoing_filled_ranges + middle_filled_ranges + incoming_filled_ranges)

    return track_counts, range_sets


def create_back_edge_dummies(all_nodes: list[Node], layers: list[list[Node]]):
    for i in range(len(layers) - 1):
        layer = layers[i]
        j = 0
        while j < len(layer):
            node = layer[j]
            if node.is_normal:
                for edge in node.children:
                    if edge.is_flipped:
                        assert is_loop_type(edge.typ)
                        # need to:
                        # 1. mark all intermediate dummies as removed
                        # 2. delete intermediate edges and dummies
                        # 3. insert ibed and obed
                        # 4. link up ibed and obed (with optional dummies in between)
                        edge.delete_me = True

                        end_target = edge.non_dummy_target
                        if end_target != edge.target: # there's dummies to remove
                            last = node
                            current = edge.target
                            while current != end_target:
                                current.delete_me = True
                                last = current
                                current = current.children[0].target
                            end_target.parents.remove(last.children[0])

                        ibed = Node(f"ibed_for {node.name}=>{end_target.name}", (0, 255, 0), depth=node.depth, is_ibed=True)
                        ibed.ibed_depth_diff = end_target.depth - node.depth
                        ibed.layer = node.layer
                        ibed.ibed_parent = node

                        node.ibeds.append(ibed)
                        node.ibeds.sort(key=lambda ib: ib.ibed_depth_diff, reverse=True)
                        for i_, ib_ in enumerate(node.ibeds):
                            ib_.ibed_idx = i_

                        layer.insert(j, ibed)
                        j += 1
                        all_nodes.append(ibed)

                        obed = Node(f"obed_for {node.name}=>{end_target.name}", (255, 0, 0), depth=end_target.depth, is_obed=True)
                        obed.layer = end_target.layer
                        layers[end_target.layer].insert(end_target.layer_index, obed)
                        all_nodes.append(obed)

                        end_target.back_edge_targets.append(node)

                        if end_target != edge.target:
                            # we need intermediate dummies
                            dummies = [
                                Node(f"dummy_{ibed.name}_{obed.name}_{i}", (255, 255, 255), depth = ibed.depth + 1, is_dummy=True)
                                for i in range(obed.layer - ibed.layer - 1)
                            ]

                            lyr = ibed.layer
                            for dummy in dummies:
                                lyr += 1
                                dummy.layer = lyr
                                layers[lyr].insert(int((ibed.layer_index + obed.layer_index)/2), dummy)
                            ibed.add_child(dummies[0], edge.typ)._flipped = True
                            for i_ in range(len(dummies) - 1):
                                # noinspection PyProtectedMember
                                dummies[i_].add_child(dummies[i_+1], edge.typ)._flipped = True
                            # noinspection PyProtectedMember
                            dummies[-1].add_child(obed, edge.typ)._flipped = True
                            print("pass")
                        else:
                            ibed_obed_edge = Edge(ibed, obed, edge.typ, flipped=True) # todo should flipped be true or false?
                            ibed.children.append(ibed_obed_edge)
                            obed.parents.append(ibed_obed_edge)
                node.children = [edge for edge in node.children if not hasattr(edge, "delete_me")]
            j += 1

    all_nodes_bkp = all_nodes.copy()
    all_nodes.clear()
    for node in all_nodes_bkp:
        if not hasattr(node, "delete_me"):
            all_nodes.append(node)

    for i in range(len(layers)):
        layers[i] = [node for node in layers[i] if not hasattr(node, "delete_me")]


def link_obeds(all_nodes: list[Node], layers: list[list[Node]]):
    for i in range(len(layers)):
        layer = layers[i]
        # ibed_parent : (obeds leading to it, normal nodes needing an obed from that parent)
        obed_groups: dict[Node, tuple[list[Node], list[Node]]] = {}
        for node in layer:
            if node.is_obed:
                parent_ibed = node.parents[0].non_dummy_source
                parent = parent_ibed.ibed_parent
                if parent not in obed_groups:
                    obed_groups[parent] = ([], [])
                obed_groups[parent][0].append(node)
            elif node.is_normal:
                for parent in node.back_edge_targets:
                    if parent not in obed_groups:
                        obed_groups[parent] = ([], [])
                    obed_groups[parent][1].append(node)

        for obed_nodes, normal_nodes in obed_groups.values():
            in_order = sorted(obed_nodes+normal_nodes, key=lambda n: n.layer_index)
            def assign_tmp_indexes():
                normal_idx = 0
                obed_idx = 0
                for j, n in enumerate(in_order):
                    n.tmp_idx = j
                    if n.is_normal:
                        n.tmp_normal_idx = normal_idx
                        normal_idx += 1
                    if n.is_obed:
                        n.tmp_obed_idx = obed_idx
                        obed_idx += 1

            assign_tmp_indexes()

            while len(in_order) > 0:
                count = len(obed_nodes)
                assert count == len(normal_nodes)

                # [row][column]
                # [obed][normal]
                distance_array: list[list[int]] = [[0 for _ in range(count)] for _ in range(count)]
                for obed_idx in range(count):
                    row = distance_array[obed_idx]
                    obed_node = obed_nodes[obed_idx]
                    for normal_idx in range(count):
                        normal_node = normal_nodes[normal_idx]
                        row[normal_idx] = abs(normal_node.tmp_idx - obed_node.tmp_idx)

                print("\nDistance array:")
                dbg_table = [[str(v) for v in row] for row in distance_array]
                for obed_idx in range(count):
                    dbg_table[obed_idx].insert(0, obed_nodes[obed_idx].name)

                dbg_table.insert(0, [""] + [normal_node.name for normal_node in normal_nodes])
                table_print(dbg_table)

                for obed_idx in range(count):
                    for normal_idx in range(count):
                        if distance_array[obed_idx][normal_idx] == 1:
                            obed_node = obed_nodes.pop(obed_idx)
                            normal_node = normal_nodes.pop(normal_idx)
                            obed_node.obed_parent = normal_node
                            print(f"Linked `{obed_node.name}` to `{normal_node.name}`")

                            a, b = max(obed_node.tmp_idx, normal_node.tmp_idx), min(obed_node.tmp_idx, normal_node.tmp_idx)

                            in_order.pop(a)
                            for higher_idx in range(a, len(in_order)):
                                in_order[higher_idx].tmp_idx -= 1

                            in_order.pop(b)
                            for higher_idx in range(b, len(in_order)):
                                in_order[higher_idx].tmp_idx -= 1

                            for higher_idx in range(obed_idx, len(obed_nodes)):
                                obed_nodes[obed_idx].tmp_obed_idx -= 1

                            for higher_idx in range(normal_idx, len(normal_nodes)):
                                normal_nodes[normal_idx].tmp_normal_idx -= 1

                            break
                    else: # if inner loop did not break, then continue
                        continue
                    break # if inner loop did break, then break here too


def create_temp_ibed_edges(all_segments: list[LinearSegment]):
    for segment in all_segments:
        first_node = segment.nodes[0]
        if first_node.is_ibed:
            normal_segment = first_node.ibed_parent.linear_segment
            for parent_segment in normal_segment.parents:
                parent_segment.children.append(segment)
                segment.parents.append(parent_segment)


def organize_graph(root: Node) -> tuple[list[list[Node]], list[LinearSegment], list[int], list[list[RangeSet]]]:
    print("Breaking cycles...")
    dfs_cycle_break(root)

    print("Assigning to layers...")
    all_nodes: list[Node] = get_all_nodes(root)
    layers = assign_to_layers(all_nodes)

    print("Layers:")
    for layer in layers:
        print(f"\t{layer}")

    print("Promoting vertices...")
    promote_vertices_heuristic(all_nodes)
    layers = rebuild_layers(all_nodes)
    print("Layers:")
    for layer in layers:
        print(f"\t{layer}")

    print("Inserting dummy nodes...")
    insert_dummy_nodes(all_nodes, layers)
    all_nodes = get_all_nodes(root)
    print("Dummy nodes inserted.")
    print("Layers:")
    for layer in layers:
        print(f"\t{layer}")

    # index layers
    list(map(index_layer, layers))

    # create ibed and obed nodes
    create_back_edge_dummies(all_nodes, layers)

    # index layers
    list(map(index_layer, layers))

    # noinspection PyUnreachableCode
    if False: # Crossing count debug
        print("Crossing count between l[1] and l[2]:")
        print(intersection_count(layers[1], layers[2], 1, 2))

    print("Layer by layer sweep...")
    layer_by_layer_sweep(layers)

    print("Layer by layer transpose sweep...")
    layer_by_layer_transpose_sweep(layers)

    print("Assigning to linear segments...")
    all_segments = assign_to_linear_segments(layers)
    print("Linear segments:")
    for segment in all_segments:
        print(f"\t{segment}")

    print("Ranking segments...")
    rank_segments(all_segments)

    print("Indexing nodes by ranking index...")
    index_nodes_by_ranking_index(layers)

    print("Generating segment adjacency pairs...")
    adjacent_segments = generate_adjacent_segments(all_segments)
    print("Adjacent segments:")
    for i, dat in enumerate(adjacent_segments):
        a, b, overlap = dat
        print(f"\t[{i}]: {a} and {b} overlap {overlap}")

    #a, b, _ = adjacent_segments[2]
    #a.transpose_with(b, layers)

    if DO_SEGMENT_TRANSPOSE:
        print("Segment by segment transpose sweep...")
        segment_by_segment_transpose_sweep(adjacent_segments, layers, all_segments)

    #print("Transposing segments...")
    #segment0 = next(segment for segment in all_segments if segment.nodes[0].name == "H")
    #segment1 = next(segment for segment in all_segments if segment.nodes[0].name == "F")
    #print(segment0.is_next_to(segment1))
    #segment0.transpose_with(segment1, layers)

    print("Assigning coordinates...")
    assign_coordinates(all_segments)

    print("Creating temp edges for ibeds")
    create_temp_ibed_edges(all_segments)

    if True:
        print("Pendulum balance...")
        try:
            tmp_print = print
            #globals()["print"] = lambda *args, **kwargs: None
            pendulum_balance(all_segments)
        except StopBalance as e:
            print(f"Balance forced to stop {e}")
        except AssertionError:
            print("Balance forced to stop")
        finally:
            globals()["print"] = tmp_print

    print("Assigning ports")
    for node in all_nodes:
        node.assign_ports()

    print("Linking obeds")
    link_obeds(all_nodes, layers)

    print("Assigning tracks")
    track_counts, range_sets = assign_tracks(layers)

    return layers, all_segments, track_counts, range_sets


# 1234567890
def main():
    global MOUSE_POS, SCALE, SELECTED_SEGMENTS, DO_SELECT
    #rt = example_cfg("dependency_minimal", prefix="")#example_cfg("paper3")
    rt = example_cfg("dependency", prefix="")
    #rt = example_cfg("basic_regions_1b")
    #rt = example_cfg("2c")
    #rt = example_cfg("3")
    #rt = example_cfg("back_edge_2c")
    rt = example_cfg("with_code_1")
    rt = example_cfg("ast_test_out", prefix="")
    print(rt)

    rt = rt.deepcopy()

    print("Organizing graph...")
    layers, all_segments, interlinear_track_counts, interlinear_occupied_ranges = organize_graph(rt)
    print("Graph organized.")

    max_segment_x = max(segment.x + segment.width for segment in all_segments)
    print(f"Max segment x: {max_segment_x}")

    if SVG_OUTPUT:
        # hide from PyCharm what we're doing
        globals()["pygame_bkp"] = globals()["pygame"]
        globals()["FONT_bkp"] = globals()["FONT"]
        globals()["pygame"] = globals()["pygame_svg_shim"]
        globals()["FONT"] = pygame.font.SysFont("FiraCode Nerd Font", 20)

        out = pygame.Surface((4500, 2500))
        out.fill((255, 255, 255))

        display_graph_with_layout(out, rt, layers=layers, use_linear_segment_coords=True, offset=50, vert_offset=20, interlinear=interlinear_track_counts, interlinear_ranges=interlinear_occupied_ranges)

        #print("Draw calls:")
        ## noinspection PyUnresolvedReferences
        #for draw_call in out.draw_calls:
        #    print(f"\t{draw_call}")

        pygame.image.save(out, "output.svg")

        globals()["pygame"] = globals()["pygame_bkp"]
        globals()["FONT"] = globals()["FONT_bkp"]
        del globals()["pygame_bkp"]
        del globals()["FONT_bkp"]
    else:
        out = pygame.Surface((4500, 2500))
        out.fill((255, 255, 255))

        display_graph_with_layout(out, rt, layers=layers, use_linear_segment_coords=True, offset=50, vert_offset=20, interlinear=interlinear_track_counts, interlinear_ranges=interlinear_occupied_ranges)

        pygame.image.save(out, "output.png")

    screen = pygame.display.set_mode((800, 700))

    offset = (0, 0)

    last_offset = (0, 0)
    middle_drag_start: tuple[int, int] | None = None

    only_layer = None

    kg = True
    while kg:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                kg = False
            elif event.type == pygame.MOUSEMOTION:
                MOUSE_POS = (event.pos[0]/SCALE, event.pos[1]/SCALE)
                if middle_drag_start is not None:
                    offset = event.pos[0]-middle_drag_start[0]+last_offset[0], event.pos[1]-middle_drag_start[1]+last_offset[1]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # left-click
                    try:
                        downsweep_pendulum(all_segments, only_layer=only_layer)
                    except StopBalance as e:
                        print(f"Balance forced to stop {e}")
                    normalize_segment_positions(all_segments)

                    sum_of_forces = sum(segment.force_from_above()+segment.force_from_below() for segment in all_segments)
                    print(f"Sum of forces: {sum_of_forces}")
                elif event.button == 2: # middle-click
                    last_offset = offset
                    middle_drag_start = event.pos
                elif event.button == 3: # right-click
                    try:
                        upsweep_pendulum(all_segments, only_layer=only_layer)
                    except StopBalance as e:
                        print(f"Balance forced to stop {e}")
                    normalize_segment_positions(all_segments)

                    sum_of_forces = sum(segment.force_from_above()+segment.force_from_below() for segment in all_segments)
                    print(f"Sum of forces: {sum_of_forces}")
                ####elif event.button == 4: # scroll up
                ####    SCALE += 0.1
                ####elif event.button == 5: # scroll down
                ####    SCALE -= 0.1
                ####else:
                ####    print("mouse press:", event.button)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2: # middle-click
                    middle_drag_start = None
                    last_offset = offset
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if only_layer is None:
                        only_layer = 1
                    else:
                        only_layer += 1
                elif event.key == pygame.K_DOWN:
                    if only_layer is not None and only_layer > 1:
                        only_layer -= 1
                    else:
                        only_layer = None
                elif event.key == pygame.K_x:
                    print("Move a segment")
                    target_segment = input("Enter target segment: ")
                    try:
                        target_segment = next(segment for segment in all_segments if segment.nodes[0].name == target_segment)
                    except StopIteration:
                        print("Segment not found")
                        continue
                    target_pos = float(input("Enter target position: "))
                    target_segment.x = target_pos
                elif event.key == pygame.K_s:
                    if SELECTED_SEGMENTS is not None:
                        DO_SELECT = True
                elif event.key == pygame.K_e:
                    if SELECTED_SEGMENTS is not None:
                        # PyCharm can't handle tuples very well
                        # noinspection PyArgumentList,PyTypeChecker
                        print(f"Command already in progress (need {SELECTED_SEGMENTS[0]-len(SELECTED_SEGMENTS[1])} more selections)")
                    else:
                        def check_adjacency(segments: list[LinearSegment]):
                            print(f"{segments[0]}.is_next_to({segments[1]}): {segments[0].is_next_to(segments[1])}")

                        def swap(segments: list[LinearSegment]):
                            segments[0].transpose_with(segments[1], layers)
                            print("Swapped")
                            print("Reassigning coordinates...")
                            assign_coordinates(all_segments)
                            print("Reassigned coordinates")

                        commands: list[tuple[str, int, typing.Callable[[list[LinearSegment]], None]]] = [
                            ("Check adjacency", 2, check_adjacency),
                            ("Swap", 2, swap)
                        ]
                        print("Commands:")
                        for i, (name, count, _) in enumerate(commands):
                            print(f"\t{i+1}: {name} ({count} selections)")

                        command = int(input("Enter command: ")) - 1
                        if command < 0 or command >= len(commands):
                            print("Invalid command")
                        else:
                            cmd = commands[command]
                            SELECTED_SEGMENTS = (cmd[1], [], cmd[2])
                            print("Make selections by pressing 's'")


        screen.fill((255, 255, 255))
        #flip_order = not flip_order
        tmp = pygame.Surface((round(screen.get_width()/SCALE), round(screen.get_height()/SCALE)))
        tmp.fill((255, 255, 255))
        display_graph_with_layout(tmp, rt, layers=layers, use_linear_segment_coords=True, offset=150+offset[0], vert_offset=20+offset[1], highlight_layer=only_layer, interlinear=interlinear_track_counts, interlinear_ranges=interlinear_occupied_ranges)
        screen.blit(pygame.transform.scale(tmp, screen.get_size()), (0, 0))

        only_layer_text = FONT.render(f"Only layer: {only_layer}", True, (0, 0, 0))
        only_layer_rect = only_layer_text.get_rect(bottomleft=(10, screen.get_height()-10))
        screen.blit(only_layer_text, only_layer_rect)

        pygame.display.flip()


if __name__ == "__main__":
    main()