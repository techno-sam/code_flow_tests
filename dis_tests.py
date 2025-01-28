import dis
import typing
from typing import TextIO, Iterable, AnyStr
from example_functions import *


#import octrace.octree_builder
#dis.dis(octrace.octree_builder.octree_array_to_bytes)

# noinspection PyAbstractClass
class StringBufferIO(typing.TextIO):

    def __init__(self):
        self.buf: str = ""

    def close(self) -> None: pass

    def flush(self) -> None: pass

    def isatty(self) -> bool: return False

    def readable(self) -> bool: return False

    def writable(self) -> bool: return True

    def write(self, __s: AnyStr) -> int:
        self.buf += __s
        return len(__s)

    def writelines(self, __lines: Iterable[AnyStr]) -> None:
        for line in __lines:
            self.write(line+"\n")

    def __enter__(self) -> TextIO: pass

f = StringBufferIO()

dis.dis(test, file=f)

print("\n\ntest(a)\n\n")
print(f.buf)

print("\n\nmain()\n\n")
dis.dis(main)