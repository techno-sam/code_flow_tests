import ast
import inspect
import json
import sys
import typing
from colorama import Fore, Back, Style

from example_functions import *


Edge: typing.TypeAlias = tuple[str, 'Node']


STRIP_COMMENTS = True # todo fix crash when comments not stripped

LINK_COUNT = 0


class LineChar:

    __mapping = {
        "": " ",
        "l": "╴",
        "r": "╶",
        "u": "╵",
        "d": "╷",
        "lr": "─",
        "ud": "│",
        "lu": "┘",
        "ld": "┐",
        "ru": "└",
        "rd": "┌",
        "lru": "┴",
        "lrd": "┬",
        "lud": "┤",
        "rud": "├",
        "lrud": "┼"
    }
    __reverse_mapping = {v: k for k, v in __mapping.items()}

    def __init__(self, left: bool = False, right: bool = False, up: bool = False, down: bool = False):
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    def __str__(self):
        key = ""
        if self.left:
            key += "l"
        if self.right:
            key += "r"
        if self.up:
            key += "u"
        if self.down:
            key += "d"
        return LineChar.__mapping[key]

    def __repr__(self):
        return f"LineChar({self.left}, {self.right}, {self.up}, {self.down} [{str(self)}])"

    def __add__(self, other: 'LineChar|str') -> 'LineChar':
        if type(other) == str:
            other_key = LineChar.__reverse_mapping[other]
            other = LineChar()
            other.left = "l" in other_key
            other.right = "r" in other_key
            other.up = "u" in other_key
            other.down = "d" in other_key
        return LineChar(
            left=self.left or other.left,
            right=self.right or other.right,
            up=self.up or other.up,
            down=self.down or other.down
        )


class Context:
    def __init__(self, function_name: str, source: str):
        self._function_name: str = function_name
        self._lines: list[str] = source.split('\n')
        self._used_lines: list[bool] = [False] * len(self._lines)
        self._multi_used_lines: list[bool] = [False] * len(self._lines)

        self._min_line: int = 0
        """Minimum line number that can be taken (inclusive)"""
        self._max_line: int = len(self._lines)
        """Maximum line number that can be taken (exclusive)"""

        assert self._min_line < self._max_line

        self._level: int = 0
        self._preemptive_end_of_block_node: 'Node|None' = None
        self._parent: 'Context|None' = None
        self._root: 'Context|None' = None
        self._return_node: 'Node|None' = None
        self._root_node: 'Node' = Node("root")

        self._continue_node: 'Node|None' = None
        self._break_node: 'Node|None' = None

    def take_line(self, line_number: int) -> str:
        if line_number < self._min_line or line_number >= self._max_line:
            raise IndexError(f"Line number {line_number} out of bounds [{self._min_line}, {self._max_line})")
        if self._used_lines[line_number]:
            #print(inspect.stack(), file=sys.stderr)
            print(f"Warning: Line {line_number} already taken", file=sys.stderr)
            self._multi_used_lines[line_number] = True
        self._used_lines[line_number] = True
        return self._lines[line_number]

    def take_lines(self, start: int, end: int) -> list[str]:
        return [self.take_line(i) for i in range(start, end)]

    @property
    def root_node(self) -> 'Node': return self._root_node

    @property
    def function_name(self) -> str: return self._function_name

    @property
    def level(self) -> int: return self._level

    @property
    def parent(self) -> 'Context|None': return self._parent

    @property
    def root(self) -> 'Context|None': return self._root

    @property
    def is_root(self) -> bool: return self._root is None

    @property
    def return_node(self) -> 'Node':
        if self.is_root:
            if self._return_node is None:
                self._return_node = Node("end-of-function")
            return self._return_node
        else:
            return self.root.return_node

    @property
    def continue_node(self) -> 'Node|None':
        current = self
        while current._continue_node is None and current._parent is not None:
            current = current._parent
        return current._continue_node

    @property
    def break_node(self) -> 'Node|None':
        current = self
        while current._break_node is None and current._parent is not None:
            current = current._parent
        return current._break_node

    def inner(self, start_line: int, end_line: int, continue_node: 'Node|None' = None, break_node: 'Node|None' = None) -> 'Context':
        if start_line < self._min_line or end_line > self._max_line:
            raise IndexError(f"Line range [{start_line}, {end_line}) out of bounds [{self._min_line}, {self._max_line})")

        new = Context(self._function_name, "")

        new._lines = self._lines
        # intentionally NOT copying used lines
        new._used_lines = self._used_lines
        new._multi_used_lines = self._multi_used_lines
        new._min_line = start_line
        new._max_line = end_line
        new._level = self._level + 1
        new._parent = self
        new._root = self._root or self
        new._root_node = self._root_node
        new._continue_node = continue_node
        new._break_node = break_node

        return new

    def create_preemptive_eob(self, may_create: bool = True, ignore_parent: bool = False) -> 'Node|None':
        if self._preemptive_end_of_block_node is not None:
            if may_create:
                print("Warning: Pre-emptive end-of-block node already exists", file=sys.stderr)
        elif not ignore_parent and self.parent is not None and self.parent._preemptive_end_of_block_node is not None:
            print("Using pre-emptive end-of-block node from parent")
            return self.parent.create_preemptive_eob(may_create=False)
        elif may_create:
            self._preemptive_end_of_block_node = Node("preemptive_eob", started_as_eob=True)
        return self._preemptive_end_of_block_node

    def set_preemptive_eob(self, eob: 'Node|None'):
        self._preemptive_end_of_block_node = eob

    def node(self, name: str, children: list['Edge']|None = None, lines: list[str]|None = None, skip_eob: bool = False) -> 'Node':
        if not skip_eob and self._preemptive_end_of_block_node is not None:
            self._preemptive_end_of_block_node.name = name
            self._preemptive_end_of_block_node.children = children or []
            self._preemptive_end_of_block_node.lines = lines or []
            eob = self._preemptive_end_of_block_node
            self._preemptive_end_of_block_node = None
            return eob
        else:
            return Node(name, children=children, lines=lines)

    def has_unused_lines(self) -> bool:
        return any(not v for v in self._used_lines[self._min_line:self._max_line])

    def has_disjoint_unused_lines(self) -> bool:
        if not self.has_unused_lines():
            return False
        # check if there are multiple separate unused line ranges
        last_used = None
        unused_ranges = 0
        for i in range(self._min_line, self._max_line):
            current_used = self._used_lines[i]
            if last_used is None:
                last_used = current_used
            elif last_used != current_used:
                if not current_used:
                    unused_ranges += 1
                last_used = current_used
                if unused_ranges > 1:
                    return True
        return unused_ranges > 1

    def take_trailing_unused_lines(self) -> list[str]:
        if not self.has_unused_lines():
            return []
        assert not self.has_disjoint_unused_lines()
        assert not self._used_lines[self._max_line-1],\
            str(list(zip(
                self._lines[self._min_line:self._max_line],
                self._used_lines[self._min_line:self._max_line]
            )))
        for i in range(self._max_line-1, self._min_line-1, -1):
            if self._used_lines[i]:
                return self.take_lines(i+1, self._max_line)
        return []

    def print_line_usage(self):
        used_char = "x"
        unused_char = "o"
        for i, line in enumerate(self._lines):
            if self._used_lines[i]:
                print(Fore.GREEN, end="")
            else:
                print(Back.RED, end="")
            if self._multi_used_lines[i]:
                print(Fore.YELLOW+Back.RESET, end="")
            print(f"{i:3d} [{used_char if self._used_lines[i] else unused_char}] {line}{Style.RESET_ALL}")


class Node:
    __EXISTING_NAMES: dict[str, int] = {}
    def __init__(self, name: str, children: list['Edge']|None = None, lines: list[str]|None = None, started_as_eob: bool = False):
        if name not in Node.__EXISTING_NAMES:
            Node.__EXISTING_NAMES[name] = 0

        Node.__EXISTING_NAMES[name] += 1
        self._type: str = name
        name = f"{name} {Node.__EXISTING_NAMES[name]}"

        self._name: str = name
        self.children: list[Edge] = children or []
        self.lines: list[str] = lines or []
        self._started_as_eob: bool = started_as_eob
        self.parents: list[Edge] = []
        self._pin_to_bottom: bool = False

    def pin_to_bottom(self, pin: bool = True):
        self._pin_to_bottom = pin

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        # strip of trailing numeral
        #Node.__EXISTING_NAMES[self._type] -= 1 # this breaks things BIG TIME
        if name not in Node.__EXISTING_NAMES:
            Node.__EXISTING_NAMES[name] = 0
        Node.__EXISTING_NAMES[name] += 1
        self._type = name
        self._name = f"{name} {Node.__EXISTING_NAMES[name]}"

    @property
    def started_as_eob(self) -> bool:
        return self._started_as_eob

    def consume_started_as_eob(self) -> bool:
        v = self._started_as_eob
        self._started_as_eob = False
        return v

    def link_to(self, edge_type: str, other: 'Node'):
        global LINK_COUNT
        print(f"[{LINK_COUNT:3d}] Linking '{self.name}' to '{other.name}'")
        LINK_COUNT += 1
        self.children.append((edge_type, other))
        other.parents.append((edge_type, self))

        if self.name == "stmt 1" and other.name == "if 3":
            a = 0

    def __str__(self):
        joiner = '\\n'
        return f"Node({self.name}, {self.children}, {joiner.join(l.lstrip() for l in self.lines)})"

    def __repr__(self):
        return str(self)

    def display_as_tree(self, extra_indent: int = 1, _prefix: list[LineChar] | None = None, _already_visited: set['Node']|None = None):
        _prefix = _prefix or []#[LineChar(right=True, up=True), LineChar(left=True, right=True)]
        _already_visited = _already_visited or set()
        if self in _already_visited:
            print(self.name, "• (already visited)")
            return
        _already_visited.add(self)
        print(self.name, "•", "\\n".join(l.lstrip() for l in self.lines))
        for i, child in enumerate(self.children):
            next_prefix = _prefix.copy()
            next_char = LineChar(right=True, up=True)
            if i < len(self.children) - 1:
                next_char.down = True
            next_prefix.append(next_char)
            for _ in range(extra_indent):
                next_prefix.append(LineChar(left=True, right=True))
            print("".join(str(v) for v in next_prefix), end="")
            child[1].display_as_tree(
                extra_indent=extra_indent,
                _prefix=_prefix + [LineChar(up=next_char.down, down=next_char.down)] + [LineChar()]*extra_indent,
                _already_visited=_already_visited
            )

    def _get_all_nodes(self) -> list['Node']:
        all_nodes: list[Node] = [self]
        todo: list[Node] = [self]
        done: list[Node] = []

        while len(todo) > 0:
            current = todo.pop()
            for _, child in current.children:
                if child in done:
                    continue
                todo.append(child)
                all_nodes.append(child)
                done.append(child)

        return all_nodes

    def _simplify_internal(self) -> bool:
        if self.type == "root":
            return False
        elif self.type == "preemptive_eob":
            assert len(self.children) == 1
            assert len(self.lines) == 0

            child = self.children[0][1]
            did_improve = False

            for _, parent in self.parents:
                for i, data in enumerate(parent.children):
                    if data[1] is self:
                        parent.children[i] = (data[0], child)
                        child.parents.append((data[0], parent))
                        child.parents = [p for p in child.parents if p[1] is not self]
                        did_improve = True

            return did_improve
        else:
            if len(self.children) == 1:
                edge_type, child = self.children[0]
                if len(child.parents) == 1 and edge_type == "direct" and child.type in {"stmt", "preemptive_eob"}:
                    self.lines.extend(child.lines)
                    self.children = child.children
                    for _, child_new in self.children:
                        for i, data in enumerate(child_new.parents):
                            et, par = data
                            if par == child:
                                child_new.parents[i] = (et, self)
                    return True
            return False

    def simplify(self):
        all_nodes = self._get_all_nodes()
        improved = True
        while improved:
            improved = False
            for node in all_nodes:
                improved |= node._simplify_internal()

    def export(self) -> dict[str, ...]:
        nodes: list[dict[str, ...]] = []
        edges: list[dict[str, ...]] = []

        all_nodes = self._get_all_nodes()

        for node in all_nodes:
            node_desc = {
                "name": node.name,
                "color": "#ffaaff",
                "lines": node.lines
            }
            if node.type == "end-of-function" or node._pin_to_bottom:
                node_desc["pin_to_bottom"] = True
            nodes.append(node_desc)
            for edge_type, target in node.children:
                edges.append({
                    "from": node.name,
                    "to": target.name,
                    "type": edge_type
                })

        return {
            "root": "root 1",
            "nodes": nodes,
            "edges": edges
        }

    def __eq__(self, other: 'Node') -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(("Node", id(self)))


class FileCodeObjectSham:
    def __init__(self, filename: str):
        self.co_filename = filename
        self.co_firstlineno = 1
        self.__code__ = self


def get_function_source(func) -> str:
    out_lines: list[str] = []
    code_object = func.__code__
    with open(code_object.co_filename, 'r') as file:
        source_code = file.read()
        lines = source_code.split('\n')
        start_line = code_object.co_firstlineno - 1
        lines = lines[start_line:]

        base_indent = len(lines[0]) - len(lines[0].lstrip())
        lines = [line[base_indent:] for line in lines]

        out_lines.append(lines[0])

        for i, line in enumerate(lines):
            if i == 0:
                continue
            o_line = line

            if STRIP_COMMENTS:
                # strip out comments
                comment_index = line.find('#')
                if comment_index != -1:
                    line = line[:comment_index]

            line = line.strip()
            if line == '':
                continue

            # check for de-indent
            if len(o_line) - len(o_line.lstrip()) <= base_indent:
                break
            out_lines.append(o_line)

    return "\n".join(out_lines)


POP_RETURN = "POP_RETURN"
POP_BREAK = "POP_BREAK"
POP_CONTINUE = "POP_CONTINUE"


def analyze_function(ast_tree: ast.FunctionDef, src: str):
    global LINK_COUNT
    assert type(ast_tree) == ast.FunctionDef
    print(ast.dump(ast_tree, indent=4))

    """
    How to traverse:
    don't worry about loops for now
    return statements should instantly link to the end of the function
    if statements should branch off and then pop back to previous scope
    """
    ctx = Context(ast_tree.name, src)
    root = ctx.root_node

    root.lines.extend(
        ctx.take_lines(ast_tree.lineno-1, ast_tree.body[0].lineno-1)
    )

    LINK_COUNT = 0
    fct, ln = analyze_internal(ast_tree.body, root, ctx)
    if fct == POP_RETURN:
        ln.pin_to_bottom()
    root.simplify()
    root.display_as_tree()

    if ctx.has_unused_lines():
        print("Warning: Unused lines in source code")
    print("Line usage summary:")
    ctx.print_line_usage()

    with open("ast_test_out.json", "w") as f:
        json.dump(root.export(), f, indent=2)


def contains_recursive_call_to(func: str, stmt: ast.AST) -> bool:
    if isinstance(stmt, ast.Call):
        if isinstance(stmt.func, ast.Name):
            if stmt.func.id == func:
                return True
    for child in ast.iter_child_nodes(stmt):
        if contains_recursive_call_to(func, child):
            return True
    return False


def analyze_internal(
        body: list[ast.stmt],
        parent_node: Node,
        context: Context,
        parent_link_type: str = "direct",
        first_node_lines: list[str]|None = None) -> tuple[str | None, Node | None]:
    """Test

    :param body:             list of statements inside the current node
    :param parent_node:      the node surrounding the current statements
    :param context:          the context of the current node
    :param parent_link_type: edge type for link from parent
    :param first_node_lines: lines to prepend to first node
    :return: (flow control type: POP_RETURN|etc...|None, last_node)
    """

    is_first_node = True

    last_node = None
    after_complex_block = False
    next_permitted_parent_link = -1

    # extra complexity needed to allow with statements to extend body dynamically
    i = -1
    while i + 1 < len(body):#for i, stmt in enumerate(body):
        i += 1
        stmt = body[i]
        if isinstance(stmt, ast.If):
            after_complex_block = True
            child_node = context.node("if")

            if is_first_node and first_node_lines is not None:
                child_node.lines.extend(first_node_lines)
                is_first_node = False

            eob = context.create_preemptive_eob(ignore_parent=True) # this seems to be causing some problems w/ excessive eob linakge
            last_node = eob

            # add source code lines to the node
            child_node.lines.extend(
                context.take_lines(stmt.lineno - 1, stmt.body[0].lineno-1)
            )
            if i >= next_permitted_parent_link:
                parent_node.link_to(parent_link_type, child_node)
            if_inner_ctx = context.inner(stmt.body[0].lineno - 1, stmt.body[-1].end_lineno)
            fct, ln = analyze_internal(
                stmt.body,
                child_node,
                if_inner_ctx,
                parent_link_type="if-true"
            )
            ln = ln or child_node
            if fct == POP_RETURN:
                ln.link_to("direct", context.return_node)
            elif fct == POP_CONTINUE:
                ln.link_to("loop", context.continue_node)
            elif fct == POP_BREAK:
                ln.link_to("direct", context.break_node)
            else:
                ln.link_to("direct", eob)

            if if_inner_ctx.has_unused_lines():
                # add source code lines to the node
                ln.lines.extend(if_inner_ctx.take_trailing_unused_lines())
                assert not if_inner_ctx.has_unused_lines()

            if stmt.orelse:
                ###child_node_i = context.node("else", skip_eob=True)
                #### add source code lines to the node
                ###child_node_i.lines.extend(
                ###    context.lines[stmt.body[-1].lineno:stmt.orelse[0].lineno-1]
                ###)
                ###child_node.link_to("if-false", child_node_i)
                fct_i, ln_i = analyze_internal(
                    stmt.orelse,
                    child_node,
                    context.inner(stmt.orelse[0].lineno-1, stmt.orelse[-1].end_lineno),
                    parent_link_type="if-false",
                    first_node_lines=context.take_lines(stmt.body[-1].end_lineno, stmt.orelse[0].lineno-1)
                )
                ln_i = ln_i or child_node
                if fct_i == POP_RETURN:
                    ln_i.link_to("direct", context.return_node)
                elif fct_i == POP_CONTINUE:
                    ln_i.link_to("loop", context.continue_node)
                    next_permitted_parent_link = i + 2
                elif fct_i == POP_BREAK:
                    ln_i.link_to("direct", context.break_node)
                    next_permitted_parent_link = i + 2
                else:
                    ln_i.link_to("direct", eob)
                    next_permitted_parent_link = i + 2
            else:
                child_node.link_to("if-false", eob)
                next_permitted_parent_link = i + 2
        elif isinstance(stmt, ast.For) or isinstance(stmt, ast.While):
            after_complex_block = True
            if isinstance(stmt, ast.For):
                loop_type = "for"
            else:
                loop_type = "while"
            child_node = context.node(f"loop-{loop_type}")

            if is_first_node and first_node_lines is not None:
                child_node.lines.extend(first_node_lines)
                is_first_node = False

            # add source code lines to the node
            child_node.lines.extend(
                context.take_lines(stmt.lineno-1, stmt.body[0].lineno-1)
            )
            if i >= next_permitted_parent_link:
                parent_node.link_to(parent_link_type, child_node)

            eob = context.create_preemptive_eob(ignore_parent=True)
            context.set_preemptive_eob(None)

            fct, ln = analyze_internal(
                stmt.body,
                child_node,
                context.inner(
                    start_line=stmt.body[0].lineno-1,
                    end_line=stmt.body[-1].end_lineno,
                    continue_node=child_node,
                    break_node=eob
                ),
                parent_link_type="if-true"
            )
            ln = ln or child_node

            context.set_preemptive_eob(eob)
            last_node = eob

            next_permitted_parent_link = i + 2

            if stmt.orelse:
                child_node_i = context.node("else", skip_eob=True)
                # add source code lines to the node
                child_node_i.lines.extend(
                    context.take_lines(stmt.body[-1].lineno, stmt.orelse[0].lineno-1)
                )
                child_node.link_to("if-false", child_node_i)
                fct_i, ln_i = analyze_internal(
                    stmt.orelse,
                    child_node_i,
                    context.inner(stmt.orelse[0].lineno-1, stmt.orelse[-1].end_lineno),
                    parent_link_type="direct"
                )
                ln_i = ln_i or child_node_i
                if fct_i == POP_RETURN:
                    ln_i.link_to("direct", context.return_node)
                else:
                    ln_i.link_to("direct", eob)
            else:
                child_node.link_to("if-false", eob)

            if fct == POP_RETURN:
                ln.link_to("direct", context.return_node)
            elif fct == POP_CONTINUE:
                ln.link_to("loop", child_node)
            elif fct == POP_BREAK:
                ln.link_to("direct", eob)
            else:
                ln.link_to("loop", child_node)
        elif isinstance(stmt, ast.With):
            child_node = context.node("with")

            if is_first_node and first_node_lines is not None:
                child_node.lines.extend(first_node_lines)
                is_first_node = False

            # add source code lines to the node
            child_node.lines.extend(
                context.take_lines(stmt.lineno - 1, stmt.body[0].lineno - 1)
            )
            if not after_complex_block or parent_node.type == "stmt" or parent_node.consume_started_as_eob():
                if i >= next_permitted_parent_link:
                    parent_node.link_to(parent_link_type, child_node)
                parent_node = child_node
                parent_link_type = "direct"
            elif child_node.started_as_eob:
                parent_node = child_node
                parent_link_type = "direct"
            last_node = child_node

            # insert stmt.body right after stmt
            if i == len(body) - 1:
                body.extend(stmt.body)
            else:
                body = body[:i + 1] + stmt.body + body[i+1:]
        else:
            if isinstance(stmt, ast.Return):
                node_id = "return"
                ret_type = POP_RETURN
            elif isinstance(stmt, ast.Continue):
                node_id = "continue"
                ret_type = POP_CONTINUE
            elif isinstance(stmt, ast.Break):
                node_id = "break"
                ret_type = POP_BREAK
            else:
                node_id = "stmt"
                ret_type = None
            child_node = context.node(node_id)

            if is_first_node and first_node_lines is not None:
                child_node.lines.extend(first_node_lines)
                is_first_node = False

            # add source code lines to the node
            child_node.lines.extend(
                context.take_lines(stmt.lineno - 1, stmt.end_lineno)
            )
            if not after_complex_block or parent_node.type == "stmt" or parent_node.consume_started_as_eob():
                if i >= next_permitted_parent_link:
                    parent_node.link_to(parent_link_type, child_node)
                parent_node = child_node
                parent_link_type = "direct"
            elif child_node.started_as_eob:
                parent_node = child_node
                parent_link_type = "direct"
            last_node = child_node

            if contains_recursive_call_to(context.function_name, stmt):
                print(f"Found recursive call to '{context.function_name}'")
                child_node.link_to("recursion", context.root_node)

            if ret_type is not None:
                return ret_type, last_node

    if context.level == 0 and last_node is not None:
        last_node.link_to("direct", context.return_node)

    return None, last_node


def main():
    src = get_function_source(FileCodeObjectSham("ast_tests.txt"))#analyze_internal)
    # noinspection PyTypeChecker
    ast_tree: ast.FunctionDef = ast.parse(src, "<string>").body[0]
    analyze_function(ast_tree, src)

    a = Node("a")

    b = Node("b")
    c = Node("c")

    d = Node("d")
    e = Node("e")

    f = Node("f")
    g = Node("g")

    # a.children = [b, c]
    a.link_to("direct", b)
    a.link_to("direct", c)

    # b.children = [d]
    b.link_to("direct", d)

    # c.children = [e]
    c.link_to("direct", e)

    # d.children = [f, g]
    d.link_to("direct", f)
    d.link_to("direct", g)

    a.display_as_tree()


if __name__ == "__main__":
    main()