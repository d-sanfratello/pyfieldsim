class Kernel:
    def __init__(self, name, width_x, width_y, size):
        self._name = name
        self._width_x = width_x
        self._width_y = width_y
        self._size_x = int(size * self.width_x)
        self._size_y = int(size * self.width_y)

    @property
    def name(self):
        return self._name

    @property
    def width_x(self):
        return self._width_x

    @property
    def width_y(self):
        return self._width_y

    @property
    def size_x(self):
        if self._size_x % 2 == 0:
            return self._size_x + 1
        else:
            return self._size_x

    @property
    def size_y(self):
        if self._size_y % 2 == 0:
            return self._size_y + 1
        else:
            return self._size_y

    @property
    def kernel(self):
        return self._generate_kernel()

    def _generate_kernel(self):
        return None
