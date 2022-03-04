import math
from collections import MutableMapping

# implementation from here: https://stackoverflow.com/questions/53138434/use-range-as-a-key-value-in-a-dictionary-most-efficient-way

# empty slot marker
_EMPTY = object()

class RangeMap(MutableMapping):
    """Map points to values, and find values for points in O(1) constant time

    The map requires a fixed minimum lower and maximum upper bound for
    the ranges. Range boundaries are quantized to a fixed step size. Gaps
    are permitted, when setting overlapping ranges last range set wins.

    """
    def __init__(self, map=None, lower=0.0, upper=1.0, step=0.05):
        self._mag = 10 ** -round(math.log10(step) - 1)  # shift to integers
        self._lower, self._upper = round(lower * self._mag), round(upper * self._mag)
        self._step = round(step * self._mag)
        self._steps = (self._upper - self._lower) // self._step
        self._table = [_EMPTY] * self._steps
        self._len = 0
        if map is not None:
            self.update(map)

    def __len__(self):
        return self._len

    def _map_range(self, r):
        low, high = r
        start = round(low * self._mag) // self._step
        stop = round(high * self._mag) // self._step
        if not self._lower <= start < stop <= self._upper:
            raise IndexError('Range outside of map boundaries')
        return range(start - self._lower, stop - self._lower)

    def __setitem__(self, r, value):
        for i in self._map_range(r):
            self._len += int(self._table[i] is _EMPTY)
            self._table[i] = value

    def __delitem__(self, r):
        for i in self._map_range(r):
            self._len -= int(self._table[i] is not _EMPTY)
            self._table[i] = _EMPTY

    def _point_to_index(self, point):
        point = round(point * self._mag)
        if not self._lower <= point <= self._upper:
            raise IndexError('Point outside of map boundaries')
        return (point - self._lower) // self._step

    def __getitem__(self, point_or_range):
        if isinstance(point_or_range, tuple):
            low, high = point_or_range
            r = self._map_range(point_or_range)
            # all points in the range must point to the same value
            value = self._table[r[0]]
            if value is _EMPTY or any(self._table[i] != value for i in r):
                raise IndexError('Not a range for a single value')
        else:
            value = self._table[self._point_to_index(point_or_range)]
            if value is _EMPTY:
                raise IndexError('Point not in map')
        return value

    def __iter__(self):
        low = None
        value = _EMPTY
        for i, v in enumerate(self._table):
            pos = (self._lower + (i * self._step)) / self._mag
            if v is _EMPTY:
                if low is not None:
                    yield (low, pos)
                    low = None
            elif v != value:
                if low is not None:
                    yield (low, pos)
                low = pos
                value = v
        if low is not None:
            yield (low, self._upper / self._mag)

class RangeMapDict(RangeMap):
    def __init__(self, dic):
        self.range_dic = dic
        self.range_map = RangeMap(dic)
