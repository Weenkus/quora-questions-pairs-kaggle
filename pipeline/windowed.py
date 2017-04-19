class WindowedIterable(object):
    def __init__(self, iterable, window_size):
        assert window_size >= 1
        self._iterable = iter(iterable)
        self._previous = list()
        self._current = None
        self._next = list()
        self._window_size = window_size
        self._is_current_set = False

        self.__get_next(window_size)

    def __get_next(self, number_of_elements):
        for _ in range(0, number_of_elements):
            try:
                self._next.append(self._iterable.__next__())
            except StopIteration:
                return

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._next) == 0:
            raise StopIteration

        if self._is_current_set:
            self._previous.append(self._current)

        self._current = self._next[0]
        self._is_current_set = True

        self._next.pop(0)
        self.__get_next(1)

        if len(self._previous) > self._window_size:
            self._previous.pop(0)

        return self._previous[:], self._current, self._next[:]


def windowify(window_size):
    def windowed(iterable):
        return WindowedIterable(iterable, window_size)
    return windowed


def dewindowify(iterable):
    for _previous, current, _next in iterable:
        yield current
