# noinspection PyPep8Naming
class _Sliceable:
    def __init__(self, func):
        self._func = func

    def __getitem__(self, item):
        assert(isinstance(item, slice))
        assert(item.step is None)
        return self._func(item.start, item.stop)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def sliceable(func):
    return _Sliceable(func)


@sliceable
def test_func(start: int, end: int):
    for i in range(start, end):
        print(i)
    print("Done")

print("Normal mode")
test_func(1, 5)
print("Indexed mode")
test_func[1:5]