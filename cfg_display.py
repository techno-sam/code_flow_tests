import json
import time

import pygame

pygame.init()
# link types:
# if-true (condition passed) -- green
# if-false (condition failed) -- red
# loop (at end of for/while loop, return to start) -- purple
# direct (end of if-block back to outer flow) -- blue


def decode_hex_color(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)


EDGE_COLORS: dict[str, tuple[int, int, int]] = {k: decode_hex_color(v) for k, v in
                                                {
                                                    "if-true": "#00FF00",
                                                    "if-false": "#FF0000",
                                                    "loop": "#800080",
                                                    "direct": "#0000FF"
                                                }.items()
                                                }


class Node:
    def __init__(self, name: str, color: tuple[int, int, int], children: list['Edge']|None = None, depth: int = 0):
        self.name: str = name
        self.color: tuple[int, int, int] = color
        self.children: list['Edge'] = children or []
        self.depth: int = depth

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("Node", self.name))

    def __str__(self) -> str:
        return f"Node({self.name}, {self.color}, #children: {len(self.children)}, depth: {self.depth})"

    def __repr__(self) -> str:
        return str(self)

    def add_child(self, child: 'Node', typ: str):
        if typ != "loop":
            child.depth = max(child.depth, self.depth + 1)
        self.children.append(Edge(self, child, typ))

    def deepest(self, seen: set['Node'] | None = None) -> int:
        if seen is None:
            seen = set()

        if self in seen:
            return self.depth

        seen.add(self)
        return max([child.target.deepest(seen) for child in self.children] + [self.depth])


class Edge:
    def __init__(self, source: Node, target: Node, typ: str):
        self.source: Node = source
        self.target: Node = target
        self.typ: str = typ

    @property
    def is_loop(self) -> bool: return self.typ == "loop"

    @property
    def is_single_parent(self) -> bool: return self.typ == "if-true" or self.typ == "if-false"


class LayoutElement:
    def __init__(self, name: str, x: int, y: int, layout: 'Layout'):
        self.name: str = name
        self.width: int = 150 + 15
        self.height: int = 50
        self.rect: pygame.Rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.center = (x, y)
        self.target_rect: pygame.Rect = self.rect.copy()
        self.target_rect0: pygame.Rect = self.target_rect.copy()
        self.layout: 'Layout' = layout

        self.side_applied_force: float = 0.0
        self.applied_force: float = 0.0

        self.above: list['LayoutElement'] = []
        self.below: list['LayoutElement'] = []
        self.primary_parent: 'LayoutElement|None' = None

        self.left: 'LayoutElement|None' = None
        self.right: 'LayoutElement|None' = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, LayoutElement):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(("LayoutElement", self.name))

    def _calc_above_force(self, my_rect: pygame.Rect) -> float:
        if len(self.above) == 0:
            return 0.0

        above_applied_forces: list[tuple[float, str|None]] = []
        counts: dict[str, int] = {}
        for above in self.above:
            name: str|None = above.primary_parent.name if above.primary_parent is not None else None
            above_applied_forces.append((above.rect.centerx - my_rect.centerx, name))
            if name is not None:
                counts[name] = counts.get(name, 0) + 1

        above_applied = 0.0
        has_none = False
        for force, name in above_applied_forces:
            if name is not None:
                force /= counts[name]*counts[name]
            else:
                has_none = True
            above_applied += force

        above_applied /= len(counts) + (1 if has_none else 0)

        return above_applied

    def _calc_below_force(self, my_rect: pygame.Rect) -> float:
        below_applied = 0.0
        for below in self.below:
            below_applied += below.rect.centerx - my_rect.centerx
        if len(self.below) > 0:
            below_applied /= len(self.below)

        below_applied *= self.layout.bottom_force_mul

        return below_applied

    def calculate_vertical_applied_force(self):
        self.applied_force = self.side_applied_force/2
        self.side_applied_force = 0.0

        above_applied = self._calc_above_force(self.rect)
        below_applied = self._calc_below_force(self.rect)

        print(f"Applying vertical-source force of {above_applied+below_applied} to {self.name}")

        self.applied_force += above_applied + below_applied

    def apply_force_step_0(self):
        self.target_rect = self.rect.copy()
        self.target_rect.centerx += self.applied_force
        self.target_rect0 = self.target_rect.copy()

    def apply_force_step_1(self):
        # move to target, taking into account collisions with left and right
        if True:
            if self.right is not None:
                if self.target_rect.right > self.right.target_rect0.left:
                    self.target_rect.right = (self.right.target_rect0.left + self.target_rect.right)/2
                    extra = (self._calc_above_force(self.target_rect) + self._calc_below_force(self.target_rect))/2
                    self.right.side_applied_force += extra
                    print(f"{self.name} applying force of {extra} to {self.right.name}")
            if self.left is not None:
                if self.target_rect.left < self.left.target_rect0.right:
                    self.target_rect.left = (self.left.target_rect0.right + self.target_rect.left)/2
                    extra = (self._calc_above_force(self.target_rect) + self._calc_below_force(self.target_rect))/2
                    self.left.side_applied_force += extra
                    print(f"{self.name} applying force of {extra} to {self.left.name}")

    def apply_force_step_2(self):
        self.rect = self.target_rect.copy()


class Layout:
    def __init__(self, root: Node):
        self._root = root
        self.bottom_force_mul: float = 1.0
        self.x_coordinates: dict[str, float] = {}
        self.x_coordinates0: dict[str, float] = {}
        self.y_values: list[int] = [
            50 * d + 15 * d for d in range(root.deepest() + 1)
        ]
        self._nodes_by_layer: list[list[Node]] = [[] for _ in range(root.deepest() + 1)]
        todo: list[Node] = [root]
        self.all_nodes: list[Node] = []

        all_edges: list[Edge] = []

        while len(todo) > 0:
            current = todo.pop(0)
            self.all_nodes.append(current)
            self._nodes_by_layer[current.depth].append(current)
            for edge in current.children:
                all_edges.append(edge)
                if edge.target not in todo and edge.target not in self.all_nodes:
                    todo.append(edge.target)

        self._elements_by_layers: list[list[LayoutElement]] = [
            [] for _ in self._nodes_by_layer
        ]

        self._elements_by_name: dict[str, LayoutElement] = {}

        for layer, nodes in enumerate(self._nodes_by_layer):
            needed_width = 150 * len(nodes) + 15 * (len(nodes) - 1)
            center = 800 // 2

            for i, node in enumerate(nodes):
                element = LayoutElement(node.name, center - needed_width // 2 + 150 * i + 15 * i, self.y_values[layer], self)
                self._elements_by_layers[layer].append(element)
                self._elements_by_name[node.name] = element

        self._update_x_coordinates()

        for edge in all_edges:
            source = self._elements_by_name[edge.source.name]
            target = self._elements_by_name[edge.target.name]
            if edge.is_single_parent:
                if source.rect.y < target.rect.y:
                    target.above.append(source)
                    target.primary_parent = source
                else:
                    target.below.append(source)
            else:
                if source.rect.y < target.rect.y:
                    if target.primary_parent is None:
                        target.primary_parent = source
                    source.below.append(target)
                    target.above.append(source)
                else:
                    source.above.append(target)
                    target.below.append(source)

        for nodes in self._elements_by_layers:
            for i, node in enumerate(nodes):
                if i > 0:
                    node.left = nodes[i - 1]
                if i < len(nodes) - 1:
                    node.right = nodes[i + 1]

    def _update_x_coordinates(self):
        self.x_coordinates.clear()
        self.x_coordinates0.clear()
        for layer, elements in enumerate(self._elements_by_layers):
            for element in elements:
                self.x_coordinates[element.name] = element.rect.centerx - 75
                self.x_coordinates0[element.name] = element.target_rect0.centerx - 75

    def apply_forces(self):
        print("\nVertical")
        for layer in reversed(self._elements_by_layers):
            for element in layer:
                element.calculate_vertical_applied_force()
                element.apply_force_step_0()

        print("\nSide")
        for layer in self._elements_by_layers:
            for element in layer:
                element.apply_force_step_1()

        print("\nApply")
        for layer in self._elements_by_layers:
            for element in layer:
                element.apply_force_step_2()

        self._update_x_coordinates()
        self.bottom_force_mul *= 0.9



def example_cfg(n: int) -> Node:
    with open(f"example_{n}.json") as f:
        data = json.load(f)

    raw_nodes: list[dict[str, str]] = data["nodes"]
    raw_edges: list[dict[str, str]] = data["edges"]

    nodes: dict[str, Node] = {data["name"]: Node(data["name"], decode_hex_color(data["color"])) for data in raw_nodes}
    for edge in raw_edges:
        nodes[edge["from"]].add_child(nodes[edge["to"]], edge["type"])

    return nodes["root"]


def display_graph_with_layout(surf: pygame.Surface, root: Node):
    # first we need to lay out each layer:
    # each node has a width of 150 and a height of 50
    # on each layer, there must be a gap of 15 between each node
    # between layers, there must be a gap of 30
    # the horizontal center of each node SHOULD be the average of the centers of its children
    if root.depth != 0:
        raise ValueError("Root node must have depth 0")

    y_values: list[int] = [
        50 * d + 15 * d for d in range(root.deepest() + 1)
    ]

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

    x_coordinates: dict[str, float] = {}
    layers: list[tuple[int, list[tuple[Node, int]]]] = []

    for layer, nodes in reversed(list(enumerate(nodes_by_layer))):
        needed_width = 150 * len(nodes) + 15 * (len(nodes) - 1)
        desired_center = 800 // 2
        if layer > 0:
            pass # this might need a two-pass algorithm or *shudder* some sort of spring simulation

        center = desired_center
        layer_x_coordinates = {node.name: center - needed_width // 2 + 150 * i + 15 * i for i, node in enumerate(nodes)}
        layers.append((center, [(node, layer_x_coordinates[node.name]) for node in nodes]))
        x_coordinates.update(layer_x_coordinates)

    display_graph(surf, all_nodes, x_coordinates, y_values)

    #print("\n".join(str(v) for v in nodes_by_layer))
    #raise StopIteration


def display_graph(surf: pygame.Surface, nodes: list[Node], x_coordinates: dict[str, float], y_values: list[int], x_coordinates0: dict[str, float]|None = None, reverse: bool = False):
    if reverse:
        nodes = nodes[::-1]
    for node in nodes:
        top_y = y_values[node.depth]
        left_x = x_coordinates[node.name]
        pygame.draw.rect(surf, node.color, (left_x, top_y, 150, 50))

        # draw edges
        p0 = (left_x + 75, top_y + 50)
        for edge in node.children:
            p1 = (x_coordinates[edge.target.name] + 75, y_values[edge.target.depth])
            pygame.draw.line(surf, EDGE_COLORS[edge.typ], p0, p1, 2)

    if x_coordinates0 is not None:
        for node in nodes:
            top_y = y_values[node.depth]
            left_x = x_coordinates0[node.name]

            pygame.draw.rect(surf, (0, 0, 0), (left_x-2, top_y-2, 150+4, 50+4), width=6)
            pygame.draw.rect(surf, node.color, (left_x, top_y, 150, 50), width=2)


def main():
    rt = example_cfg(4)
    print(rt)

    layout = Layout(rt)

    for element in layout._elements_by_name.values():
        print(element.name, element.primary_parent.name if element.primary_parent is not None else None)

    screen = pygame.display.set_mode((800, 600))

    flip_order = False
    last = time.time()
    holding = False

    kg = True
    while kg:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                kg = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                print("Applying forces")
                layout.apply_forces()
                print("Applied forces\n")
                last = time.time()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                holding = True
            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                holding = False

        now = time.time()
        if holding and now-last > 0.1:
            print("Applying forces")
            layout.apply_forces()
            print("Applied forces\n")
            last = time.time()

        screen.fill((255, 255, 255))
        #flip_order = not flip_order
        display_graph(screen, layout.all_nodes, layout.x_coordinates, layout.y_values, layout.x_coordinates0, reverse=flip_order)
        pygame.display.flip()


if __name__ == "__main__":
    main()