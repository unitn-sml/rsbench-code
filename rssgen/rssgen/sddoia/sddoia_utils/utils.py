import random
import numpy as np
import os
import sys
from contextlib import contextmanager


def set_random_seed(seed):
    """
    Set random seed for Blender, Python's random module, and numpy.
    """
    # Set random seed for Python's random module
    random.seed(seed)
    # Set random seed for numpy
    np.random.seed(seed)


class HashableDict(dict):
    def __init__(self, dict_scene):
        in_dict, scene, ys, g = dict_scene
        super().__init__(in_dict)
        self.scene = scene
        self.ys = ys
        self.g = g

    def __eq__(self, other):
        return (
            frozenset(self.items()) == frozenset(other.items())
            and self.scene == other.scene
        )

    def __hash__(self):
        key_value_strings = [f"{key}{value}" for key, value in self.items()]
        key_value_strings.append(self.scene)
        return hash("".join(key_value_strings))

    def get_scene(self):
        return self.scene

    def get_y(self):
        return self.ys

    def get_instance(self):
        return self.g


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
