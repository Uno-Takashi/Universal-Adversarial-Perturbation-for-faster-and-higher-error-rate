import os
import tempfile

import matplotlib
import pytest

# The test suite must not try to open a window.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def isolated_image_cache(monkeypatch):
    """Keep the on-disk image cache out of the working tree during tests."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("UAP_IMAGE_CACHE", os.path.join(tmp, "imagenet"))
        yield
