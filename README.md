# ID01-SXDM · modern visual analysis

Analyze scanning X-ray diffraction microscopy data from [ID01 at ESRF](https://www.esrf.fr/UsersAndScience/Experiments/XNP/ID01). This fork adds a browser-native workflow for **Python 3.11–3.14**, **uv**, and **marimo**. The BLISS readers, numerical core, and new interactive views run without Jupyter or ipywidgets.

## Install with uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Julien-Dujardin/id01-sxdm-utils.git
cd id01-sxdm-utils
# For this development version: git switch feat/modern-python-marimo
uv sync --locked --extra marimo
```

The repository defaults to Python 3.14; `uv sync --locked --python 3.12 --extra marimo` selects another supported version. `uv.lock` records dependencies for all supported Python versions. Core installation and the marimo extra use package-index releases and require no GitLab access.

| Installation | Includes |
| --- | --- |
| `uv sync --locked` | NumPy/SciPy, HDF5 readers/compression, reductions, development checks |
| `uv sync --locked --extra visual` | Core + Plotly figures and standalone interactive HTML |
| `uv sync --locked --extra marimo` | Core + Plotly + marimo scan browser |
| `uv sync --locked --extra analysis` | Matplotlib, pandas, image registration, fitting and geometry libraries |
| `uv sync --locked --extra analysis --extra jupyter` | Optional legacy notebook interfaces; some also need ESRF packages below |

When using `uv run`, include the same extras in the command; uv synchronizes the environment. Standard `pip install '.[marimo]'` also works from a checkout. Until this fork is published, do not expect the PyPI `id01-sxdm` release to contain these changes.

## Launch the scan browser

```bash
uv run --extra marimo marimo edit examples/scan_browser.py -- --dataset /path/to/dataset.h5
```

Or try a small, synthetic dataset (the generator refuses to overwrite an existing file):

```bash
uv run examples/create_demo.py /tmp/sxdm-demo.h5
uv run --extra marimo marimo edit examples/scan_browser.py -- --dataset /tmp/sxdm-demo.h5
```

The app provides a spatial scan overview, scan/counter selectors, log counter colors, and a detector-frame slider. Moving the slider reads **one detector frame**, rather than the entire 4D dataset. An explicit button integrates detector intensity across the sample. Refresh reopens the file to reload acquisition metadata. To use a read-only app interface, replace `marimo edit` with `marimo run`.

Axes are inferred from recognized 2D scan commands. For line/point-only datasets, or to choose another coordinate pair:

```bash
uv run --extra marimo marimo edit examples/scan_browser.py -- \
    --dataset /path/to/dataset.h5 --x-motor samx --y-motor samy
```

## Use a display function in your own marimo notebook

A separate application is optional. Add this cell after defining `path_dset`:

```python
@app.cell
def _(mo, path_dset, sxdm):
    mo.ui.plotly(sxdm.io.bliss.display_scan_positions(
        path_dset,
        rectangle=(0, 1, 0, 1),  # display xmin, xmax, ymin, ymax
    ))
    return
```

`display_scan_positions` maps all selected scans onto the same normalized rectangle. The default native bounds come from the full acquired coordinate arrays before display sampling. One global transform preserves relative scan locations; individual scans are never independently stretched to fill the rectangle. Hover shows scan number/name, command, timestamp, original motor coordinates/units, and mapped coordinates. Axes are labeled as mapped coordinates because independent normalization changes physical aspect ratios.

Use `rectangle=(0, 2, 0, 1)` for a wider reference rectangle. Pass `motor_bounds=(native_xmin, native_xmax, native_ymin, native_ymax)` to reuse an identical mapping when selecting subsets or comparing datasets. Points outside explicit bounds are extrapolated without clipping. An inferred constant axis is placed at the rectangle midpoint.

`ascan`, `amesh`, and ID01 `asxdm` trajectories use recorded positions. As with `get_scan_map`, datasets containing only line/point scans need explicit `x_motor="...", y_motor="..."`. Requested command bounds are not substituted for missing measurements. See the [BLISS command cheat sheet](https://bliss.gitlab-pages.esrf.fr/bliss/2.0.x/img/cheat_sheet/cheat_sheet_fundamental.pdf) for standard `ascan`/`amesh` syntax.

To retain the scan table in another cell without loading ipywidgets:

```python
mo.Html(sxdm.io.bliss.get_scan_table(path_dset, as_widget=False))
```

A minimal notebook matching this workflow is included:

```bash
uv run --extra marimo marimo edit examples/display_scan_positions.py -- --dataset /path/to/dataset.h5
```

## Spatial alternative to `get_scan_table`

```python
from sxdm.io.bliss import get_scan_map

fig = get_scan_map(path_dset)  # Plotly Figure, no FigureWidget or ipywidgets
fig
```

Each scan is drawn as **lines and measured points**. Hover shows its scan number/name, full command, timestamp, acquisition index, and motor coordinates. Pan/zoom and legend toggles work in the browser. In marimo:

```python
import marimo as mo
from sxdm.plot import get_scan_map

scan_view = mo.ui.plotly(get_scan_map(path_dset, x_motor="samx", y_motor="samy"))
scan_view
# Read scan_view.value in another cell to react to point selections.
```

Use the data API independently of plotting:

```python
from sxdm.io import get_scan_positions

scans = get_scan_positions(path_dset, x_motor="samx", y_motor="samy")
# Each ScanPositions has scan_no, title, x, y, point_index, and motor units.
```

Motor measurement channels take precedence over initial snapshots. Stationary scalar snapshots can locate point/line scans. Missing moving-motor trajectories are skipped with warnings; requested command bounds are never presented as measured positions. Incomplete coordinate arrays use their shared acquired prefix. The map keeps one common motor pair across scans, preserves gaps, and rejects inconsistent units instead of silently mixing them. Coordinates remain in native units; no coordinate-frame transform is applied.

Only numeric BLISS scan groups (`1.1`, `2.1`, etc.) are considered. Non-spatial scans or scans without usable coordinates cannot be located in 2D and are skipped with a warning. Positions come from `measurement/<motor>`, `<motor>_position`, or `instrument/positioners/` equivalents. Supply explicit axes for unfamiliar commands.

For large scans, at most 5,000 points per scan are displayed by default, plus gap separators; raw arrays remain available through `get_scan_positions`. Use `max_points=None` for all acquired points, or `scan_nos=["1.1", "3.1"]` to limit the overview. This bounds browser rendering; the coordinate channels themselves are read in full. Detector stacks are never loaded to build the overview.

## Visualize counters and inspect detector data

```python
from sxdm.plot import get_counter_map
from sxdm.io.bliss import get_detector_frame, get_sxdm_pos_sum, get_sxdm_frame_sum

# Colored measured points preserve snake/irregular/incomplete scans.
fig = get_counter_map(path_dset, "1.1", "mpx1x4_roi1", log=True)

# Read one frame, optionally crop before loading (exclusive stop bounds).
frame = get_detector_frame(path_dset, "1.1", frame_index=10,
                           roi=(20, 100, 30, 150))

sample_intensity = get_sxdm_pos_sum(path_dset, "1.1", pbar=False)
summed_detector = get_sxdm_frame_sum(path_dset, "1.1", pbar=False)
```

Reductions stream batches of approximately 32 MiB or less per worker (at least one frame) and apply detector ROIs during reading. They default to in-process execution to avoid notebook/process startup overhead. Set `n_proc=2` or another positive integer for parallel processing; workers are limited to available tasks. `True` mask entries continue to mean **exclude**. Parallel calls from scripts require the usual `if __name__ == "__main__":` guard for spawn/forkserver platforms. No speedup factor is claimed; throughput depends on detector compression, storage and ROI size.

Export an interactive figure for sharing without a Python server:

```python
fig.write_html("scans.html")  # embeds Plotly JavaScript by default
```

marimo and uv behavior follow their [plotting](https://docs.marimo.io/guides/working_with_data/plotting/) and [environment synchronization](https://docs.astral.sh/uv/concepts/projects/sync/) documentation.

## Legacy interfaces and ESRF integrations

Existing `sxdm.widgets` interfaces remain **Jupyter-only optional features**. `get_scan_table()` defaults to its legacy Jupyter widget; `as_widget=False` returns plain HTML for marimo. The marimo workflow replaces scan/counter browsing and detector-frame inspection; advanced widget interactions such as manual alignment and XSOCS fitting have not been ported to marimo. Computational functions remain usable from ordinary Python cells with their required dependencies.

SPEC geometry, XSOCS conversion, and some legacy widgets still need ESRF's `id01-core`/`xsocs`. These are intentionally outside the universal uv lock so GitLab availability and compiled legacy extensions cannot block installation of the modern core. If needed, install into an analysis/Jupyter environment with versions suitable for your beamline:

```bash
uv sync --locked --python 3.12 --extra analysis --extra jupyter
uv pip install --python .venv/bin/python \
  'id01-core @ git+https://gitlab.esrf.fr/id01-science/id01-core.git@663c3b34eec2f94fc614044811978523def7ffb9' \
  'xsocs @ git+https://gitlab.esrf.fr/zatterin/xsocs.git@f85001026679b7980722adceeef2c32b038e1b4a'
# Preserve these manually installed legacy dependencies during subsequent runs:
uv run --no-sync python your_legacy_analysis.py
```

Those historical revisions are provided for reproducibility, not as a Python 3.14 compatibility guarantee. Real beamline/XSOCS integration tests require external ESRF datasets and are skipped when those fixtures are absent. The synthetic suite covers the new workflow and prior numerical regressions.

Original [upstream documentation](https://id01-science.gitlab-pages.esrf.fr/id01-sxdm-utils/) and [SXDM publication](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.18.064015) remain useful for scientific context.
