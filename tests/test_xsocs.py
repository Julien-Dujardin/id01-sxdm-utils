"""Optional XSOCS integration tests using the external ESRF example dataset.

These tests do not download beamline data and write only to pytest's temporary
directory. The synthetic core regressions run independently of these fixtures.
"""

from pathlib import Path

import numpy as np
import pytest

import sxdm


SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "doc/source/examples/data/MA1234/id01/20230710/RAW_DATA"
    / "InGaN/InGaN_0001/InGaN_0001.h5"
)

if not SAMPLE_DATASET.is_file():
    pytest.skip(
        "External ESRF InGaN example dataset is unavailable", allow_module_level=True
    )

pytest.importorskip("xsocs", reason="XSOCS integration tests require XSOCS")
pytest.importorskip("id01lib", reason="XSOCS integration tests require id01-core")
pytest.importorskip("silx", reason="XSOCS integration tests require silx")
pytest.importorskip(
    "xrayutilities", reason="XSOCS integration tests require xrayutilities"
)


@pytest.fixture(scope="module")
def shifted_master(tmp_path_factory):
    path_out = tmp_path_factory.mktemp("xsocs-integration")
    shifts = np.array([[0.0, 0.0], [4.0, 0.0], [9.0, -1.0], [14.0, -1.0], [17.0, -3.0]])
    sxdm.utils.bliss.make_xsocs_links(str(SAMPLE_DATASET), str(path_out), None)
    path_master = path_out / "InGaN_0001_master.h5"
    result = sxdm.process.xsocs.shift_xsocs_data(
        str(path_master), str(path_out), shifts, overwrite=True
    )
    return path_out / "InGaN_0001_master_shifted.h5", result


def test_xsocs_shift(shifted_master):
    path_master, result = shifted_master
    assert result is None
    assert path_master.is_file()


def test_xsocs_qconv(shifted_master, tmp_path):
    path_master, _ = shifted_master
    path_qspace = tmp_path / "InGaN_qspace_shift.h5"
    offsets = {"eta": 0, "delta": 0, "phi": 0, "roby": 1, "nu": 0.5}

    sxdm.process.xsocs.grid_qspace_xsocs(
        str(path_qspace),
        str(path_master),
        (10, 10, 10),
        overwrite=True,
        offsets=offsets,
    )
    qx, qy, qz = sxdm.utils.get_qspace_coords(path_qspace)

    assert qx.shape == (10,)
    assert qy.shape == (10,)
    assert qz.shape == (10,)
