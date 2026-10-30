"""Package metadata must agree with the installed distribution."""
from importlib.metadata import version
import sxdm


def test_version():
    assert sxdm.__version__ == version("id01-sxdm")
