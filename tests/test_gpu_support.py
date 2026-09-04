"""Tests for the CUDA preload shim.

None of these need a GPU: the filesystem and ctypes are stubbed, so the logic is checked on
any machine.
"""

import os

import pytest

import gpu_support


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(gpu_support, "_done", False)
    monkeypatch.delenv(gpu_support.DISABLE_ENV_VAR, raising=False)


@pytest.fixture
def fake_wheels(tmp_path, monkeypatch):
    """Lay out a site-packages tree shaped like the nvidia-* wheels."""
    site_dir = tmp_path / "site-packages"
    for package, name in [
        ("cuda_runtime", "libcudart.so.12"),
        ("cudnn", "libcudnn.so.9"),
        ("cublas", "libcublas.so.12"),
    ]:
        lib_dir = site_dir / "nvidia" / package / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / name).write_bytes(b"")
    monkeypatch.setattr(gpu_support, "_site_package_dirs", lambda: [str(site_dir)])
    return site_dir


def test_find_cuda_libraries_discovers_the_wheel_layout(fake_wheels):
    found = gpu_support.find_cuda_libraries()

    assert len(found) == 3
    assert all(path.endswith((".so.12", ".so.9")) for path in found)
    assert found == sorted(found), "results should be deterministic"


def test_find_cuda_libraries_is_empty_without_wheels(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_support, "_site_package_dirs", lambda: [str(tmp_path)])
    assert gpu_support.find_cuda_libraries() == []


def test_find_cuda_libraries_deduplicates_across_site_dirs(fake_wheels, monkeypatch):
    monkeypatch.setattr(
        gpu_support, "_site_package_dirs", lambda: [str(fake_wheels), str(fake_wheels)]
    )
    assert len(gpu_support.find_cuda_libraries()) == 3


def test_enable_loads_every_library_once_per_pass(fake_wheels, monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_support.ctypes, "CDLL", lambda path, mode=0: calls.append(path))

    loaded = gpu_support.enable_cuda_wheels()

    assert loaded == 3 * gpu_support._LOAD_PASSES
    assert len(calls) == 3 * gpu_support._LOAD_PASSES


def test_enable_uses_rtld_global(fake_wheels, monkeypatch):
    modes = []

    def fake_cdll(path, mode=0):
        modes.append(mode)

    monkeypatch.setattr(gpu_support.ctypes, "CDLL", fake_cdll)
    gpu_support.enable_cuda_wheels()

    assert set(modes) == {gpu_support.ctypes.RTLD_GLOBAL}


def test_enable_survives_libraries_that_will_not_load(fake_wheels, monkeypatch):
    """A library whose dependencies are missing must not break the import."""

    def picky(path, mode=0):
        if "cudnn" in path:
            raise OSError("libcudnn needs something else first")

    monkeypatch.setattr(gpu_support.ctypes, "CDLL", picky)

    loaded = gpu_support.enable_cuda_wheels()

    assert loaded == 2 * gpu_support._LOAD_PASSES


def test_enable_is_a_no_op_without_wheels(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_support, "_site_package_dirs", lambda: [str(tmp_path)])
    monkeypatch.setattr(
        gpu_support.ctypes, "CDLL", lambda *a, **k: pytest.fail("should not load anything")
    )
    assert gpu_support.enable_cuda_wheels() == 0


def test_enable_only_runs_once(fake_wheels, monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_support.ctypes, "CDLL", lambda path, mode=0: calls.append(path))

    first = gpu_support.enable_cuda_wheels()
    second = gpu_support.enable_cuda_wheels()

    assert first > 0
    assert second == 0, "a second call must not reload the libraries"
    assert len(calls) == first


def test_enable_respects_the_disable_switch(fake_wheels, monkeypatch):
    monkeypatch.setenv(gpu_support.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(
        gpu_support.ctypes, "CDLL", lambda *a, **k: pytest.fail("preload was disabled")
    )
    assert gpu_support.enable_cuda_wheels() == 0


def test_site_package_dirs_returns_real_directories():
    dirs = gpu_support._site_package_dirs()
    assert dirs
    assert any(os.path.isdir(d) for d in dirs)


def test_classifiers_enables_cuda_before_importing_tensorflow():
    """Ordering is the whole point: TensorFlow resolves CUDA during its own import."""
    import ast

    tree = ast.parse(open("classifiers.py").read())
    order = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tensorflow":
                    order.append(("import_tf", node.lineno))
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "enable_cuda_wheels":
            order.append(("enable", node.lineno))

    enable_lines = [line for kind, line in order if kind == "enable"]
    tf_lines = [line for kind, line in order if kind == "import_tf"]
    assert enable_lines and tf_lines
    assert min(enable_lines) < min(tf_lines)
