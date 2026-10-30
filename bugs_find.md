# Bug audit: id01-sxdm-utils-main

Audit date: 2026-10-23. Scope: supplied package, numerical processing, HDF5 IO,
plotting and widget initialization/navigation. All 17 confirmed issues below are
fixed. This is an evidence-based audit, not a claim that every possible bug has
been found.

## Git and review workflow

The supplied package had no tracked files. A baseline commit was created in a
working copy and fixes developed on `fix/package-bugs`. The orchestrator integrated
the changes; a coder investigated numerical processing and reviewed IO/plot fixes;
an independent reviewer investigated chunking/widgets and reviewed the coder's
changes. Existing files in the surrounding analysis project were preserved.

## Confirmed issues

| ID | Location | Trigger and previous failure | Fix / regression coverage |
| --- | --- | --- | --- |
| 01 | `sxdm/io/utils.py::_get_chunk_indexes` | More workers than frames, empty data, or a single chunk caused division by zero or an index error; partitions could exceed the requested count. | Balanced bounded ranges, CPU fallback, positive worker validation; `tests/test_io_chunks.py`. |
| 02 | `sxdm/io/utils.py::_get_chunk_indexes_detector` | Small dimensions crashed; `n_chunks=1` returned an incompatible structure and lost nonzero ROI origins. | Consistent row/column ranges, preserved ROI coordinates and checked bounds; `tests/test_io_chunks.py`. |
| 03 | `sxdm/io/bliss.py::get_sxdm_pos_sum` | Irregular masks only selected a bounding rectangle, so excluded bright pixels inside it contributed to sums. | Apply the actual mask within the cropped data; `tests/test_io_regressions.py`. |
| 04 | `sxdm/io/bliss.py::get_sxdm_pos_sum` | An all-True detector mask called min/max on empty coordinates. | Empty detector slice yields zero sums; `tests/test_io_regressions.py`. |
| 05 | `sxdm/io/bliss.py::get_sxdm_frame_sum_multi` | Requested detector and custom data path were ignored for each scan, allowing wrong detector sums or path errors. | Forward detector and the original path template for every scan; `tests/test_io_regressions.py`. |
| 06 | `sxdm/io/bliss.py` sum docstrings | Documentation described True as included even though the established implementation excludes True values. | Corrected both detector/sample mask descriptions to match behavior. |
| 07 | `sxdm/process/math.py::_calc_com_qspace3d` | A reused worker cache retained coordinates from a different file, mask or spherical mode, producing incorrect COMs. | Key cached coordinates by file, mask and mode; `tests/test_process_regressions.py`. |
| 08 | `sxdm/process/math.py::calc_coms_qspace2d` | Irregular detector masks used only their bounding box, contaminating COM calculations. | Filter both frames and coordinate arrays by the actual mask; public API regression in `tests/test_process_regressions.py`. |
| 09 | `sxdm/process/math.py::calc_com_3d` | `n_pix` larger than the array raised an out-of-bounds error; zero silently selected every pixel. | Cap oversized selections and reject nonpositive/noninteger counts; `tests/test_process_regressions.py`. |
| 10 | `sxdm/process/math.py::ang_between` | Floating-point roundoff gave cosine slightly above 1 for parallel vectors, yielding NaN. | Clamp cosine to [-1, 1]; `tests/test_process_regressions.py`. |
| 11 | `sxdm/utils/general.py::get_shift_dset` | `np.log(where=...)` without `out` left nonpositive pixels uninitialized, making registration input nondeterministic. | Initialize those pixels to zero; `tests/test_process_regressions.py`. |
| 12 | `sxdm/plot/utils.py::make_hsv` | Integer magnitudes failed in-place division; zero magnitudes or constant stretched magnitudes generated NaNs. | Normalize a floating array only when its maximum is positive; `tests/test_plot_regressions.py`. |
| 13 | `sxdm/widgets/Inspect4DArray.py` | Passing an HDF5 dataset raised AttributeError because projections were only initialized for NumPy arrays. | Compute HDF5 projections one sample row at a time; `tests/test_widget_regressions.py`. |
| 14 | `sxdm/widgets/Inspect4DArray.py` | Arrow keys before clicking used missing state; moving beyond array bounds caused invalid indexing. | Initialize navigation state and clamp indices for the selected space; NumPy and HDF5 regressions in `tests/test_widget_regressions.py`. |
| 15 | `sxdm/widgets/bliss/GetShift.py::GetShift` | Supplying existing shifts left `self.shifts` unset and crashed widget construction. | Copy supplied shifts before widget initialization; `tests/test_widget_regressions.py`. |
| 16 | `sxdm/widgets/bliss/GetShift.py::GetShiftCustom` | `np.bool_ is True` was false, so supplied nonzero initial shifts were overwritten. | Use boolean truth testing; `tests/test_widget_regressions.py`. |
| 17 | `sxdm/widgets/bliss/GetShift.py` | Both shift widgets reduced an empty nonzero-pixel array for all-zero images. Layout also used unsupported `align-items`, ignoring alignment and emitting a traitlets warning. | Safe [0, 1] display limits and supported `align_items` trait; both widgets covered in `tests/test_widget_regressions.py`. |

## Validation and limits

- Synthetic regressions cover chunk boundaries, irregular and fully excluded masks,
  coordinate cache changes, numerical limits, HSV inputs, and widget behavior.
- Before fixes, the coder's five numerical regressions all failed or errored. The
  reviewer's original 32 chunk/widget cases produced 21 failures on the baseline.
- Run the complete verified subset with:
  `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sxdm-mpl python -m pytest -q tests --ignore=tests/test_xsocs.py`.
- Verified subset result: **49 passed** (including 43 new regression cases).
- Static validation: `ruff check sxdm tests` and `git diff --check`.
- The original two XSOCS integration tests are stateful: the second consumes files
  written by the first under the example data directory. Their multiprocessing
  shift stage did not finish during the bounded verification run. No pass is
  claimed for those integration tests; the timeout alone does not identify a
  package bug. They need a separate investigation in a suitable runtime.
- Third-party Matplotlib/pyparsing and HDF5 deprecation warnings remain.
- Q-space coordinate caching assumes a given file's coordinates are not rewritten
  in place during a running analysis.
