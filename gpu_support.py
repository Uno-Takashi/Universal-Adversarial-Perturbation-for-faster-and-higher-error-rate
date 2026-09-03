"""Make TensorFlow find the CUDA libraries shipped by the ``cuda`` extra.

``tensorflow[and-cuda]`` installs CUDA and cuDNN as ``nvidia-*`` wheels under
``site-packages/nvidia/*/lib``. TensorFlow is supposed to pick those up on its own, but when
it does not the only symptom is a one-line warning and ``list_physical_devices("GPU") == []``
-- the run then silently proceeds on the CPU, which is easy to miss and very slow.

Exporting ``LD_LIBRARY_PATH`` fixes it, but the dynamic loader reads that variable once at
process start, so setting it from Python is too late. Loading each library explicitly with
``RTLD_GLOBAL`` before TensorFlow's own ``dlopen`` calls has the same effect and needs no
wrapper script: by the time TensorFlow looks, the libraries are already in the global symbol
namespace.

:func:`enable_cuda_wheels` must therefore run *before* ``import tensorflow``. It is a no-op
when the wheels are absent (a CPU-only install, or a machine like the dev container whose
image already provides CUDA system-wide), and it never raises.
"""

import ctypes
import glob
import os
import site
import sysconfig

DISABLE_ENV_VAR = "UAP_DISABLE_CUDA_PRELOAD"

# Repeated passes let a library load once its dependencies have been loaded by an earlier one,
# which saves having to hard-code the dependency order between the CUDA libraries.
_LOAD_PASSES = 3

_done = False


def _site_package_dirs():
    dirs = []
    try:
        dirs.extend(site.getsitepackages())
    except AttributeError:  # pragma: no cover - only on unusual interpreters
        pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        dirs.append(purelib)
    return dirs


def find_cuda_libraries():
    """Return the CUDA shared objects shipped by the ``nvidia-*`` wheels, if any."""
    libraries = set()
    for directory in _site_package_dirs():
        nvidia_root = os.path.join(directory, "nvidia")
        if os.path.isdir(nvidia_root):
            libraries.update(glob.glob(os.path.join(nvidia_root, "*", "lib", "*.so*")))
    return sorted(libraries)


def enable_cuda_wheels():
    """Preload the wheel-provided CUDA libraries. Returns how many were loaded.

    Safe to call more than once; only the first call does any work.
    """
    global _done
    if _done or os.environ.get(DISABLE_ENV_VAR):
        return 0

    _done = True
    libraries = find_cuda_libraries()
    loaded = 0
    for _ in range(_LOAD_PASSES):
        for library in libraries:
            try:
                ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                # A library whose dependencies are not loaded yet, or one that does not apply
                # to this machine. Neither is worth failing an import over.
                continue
    return loaded
