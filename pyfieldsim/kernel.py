class Kernel:
    def __init__(self, name, width_x, width_y, size):
        """
        Class that initializates a generic kernel for a Point Spread Function (psf).

        When initialized, it stores a name for the kernel and typical widths on the x and y axis (`width_x` and
        `width_y`) and the size of the kernel as the multiplying factor with respect to the typical widths.

        Parameters
        ----------
        name:
            The name associated with the kernel.
        width_x:
            The typical width of the kernel along the x axis.
        width_y:
            The typical width of the kernel along the y axis.
        size:
            The multiplying factor to obtain the shape of the matrix defining the kernel.

        Examples
        --------
        >>> # Example on the `size` argument. In this case the kernel has not central simmetry and the matrix
        >>> # representing it will be an `int(2.5 * width_x)` x `int(2.5 * width_y)` matrix.
        >>> generic_kernel = Kernel(name='generic', width_x=10, width_y=8, size=2.5)

        See Also
        --------
        `psf.kernels`.
        """
        self._name = name
        self._width_x = width_x
        self._width_y = width_y
        self._size_x = int(size * self.width_x)
        self._size_y = int(size * self.width_y)

    @property
    def name(self):
        """
        The name of the kernel.
        """
        return self._name

    @property
    def width_x(self):
        """
        The typical width of the kernel along the x axis.
        """
        return self._width_x

    @property
    def width_y(self):
        """
        The typical width of the kernel along the y axis.
        """
        return self._width_y

    @property
    def size_x(self):
        """
        The size of the matrix representing the kernel, along the x axis.
        """
        if self._size_x % 2 == 0:
            return self._size_x + 1
        else:
            return self._size_x

    @property
    def size_y(self):
        """
        The size of the matrix representing the kernel, along the y axis.
        """
        if self._size_y % 2 == 0:
            return self._size_y + 1
        else:
            return self._size_y

    @property
    def kernel(self):
        """
        The `Kernel.size_x` x `Kernel.size_y` array representing the kernel.
        """
        return self._generate_kernel()

    def _generate_kernel(self):
        """
        Placeholder method for the definition of specific kernels.
        """
        return None
