"""Exercise imports and real worker startup outside pytest's interpreter."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _run_python(script, *args):
    env = os.environ.copy()
    # Prevent BLAS from starting a thread pool in every detector worker.
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=REPOSITORY,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_imports_do_not_load_notebook_or_beamline_packages(tmp_path):
    script = tmp_path / "check_imports.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import sxdm
            from sxdm.io import bliss, utils, scan_positions
            from sxdm.plot import interactive

            forbidden = {"ipywidgets", "IPython", "id01lib", "xsocs"}
            imported = {name.split(".")[0] for name in sys.modules}
            assert not (forbidden & imported), sorted(forbidden & imported)
            assert sxdm.__version__
            """
        ),
        encoding="utf-8",
    )
    _run_python(script)


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_detector_sums_with_real_worker_processes(tmp_path, start_method):
    """Spawn/forkserver work on newer Python and don't depend on fork state."""
    import multiprocessing

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable on this platform")
    script = tmp_path / "check_workers.py"
    script.write_text(
        textwrap.dedent(
            """
            import multiprocessing as mp
            from pathlib import Path
            import sys

            import h5py
            import numpy as np
            from sxdm.io.bliss import get_sxdm_frame_sum, get_sxdm_pos_sum


            def main():
                mp.set_start_method(sys.argv[2], force=True)
                path = Path(sys.argv[1])
                data = np.arange(6 * 3 * 4, dtype=np.uint16).reshape(6, 3, 4)
                with h5py.File(path, "w") as h5:
                    h5["1.1/title"] = "sxdm x 0 2 3 y 0 1 2 0.1"
                    h5["1.1/instrument/det/data"] = data
                    h5["1.1/measurement/det"] = h5["1.1/instrument/det/data"]

                for workers in (1, 2):
                    frames = get_sxdm_frame_sum(
                        path, "1.1", detector="det", n_proc=workers, pbar=False
                    )
                    points = get_sxdm_pos_sum(
                        path, "1.1", detector="det", n_proc=workers, pbar=False
                    )
                    np.testing.assert_array_equal(frames, data.sum(axis=0))
                    np.testing.assert_array_equal(points, data.sum(axis=(1, 2)))


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    _run_python(script, tmp_path / "detector.h5", start_method)
