# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- Python 3.11–3.14 support, PEP 621 packaging, uv lockfile and optional dependency groups.
- Self-contained BLISS readers, with compressed HDF5 support and no mandatory GitLab dependencies.
- `get_scan_positions`, browser-native `get_scan_map` with scan-name hover, and `get_counter_map`.
- Notebook-callable `display_scan_positions` with shared normalized-rectangle mapping and native coordinate hover; `amesh`/`asxdm` aliases.
- Plain HTML scan tables via `get_scan_table(as_widget=False)` and a minimal marimo display notebook.
- Shared overview point budgets and compact hover metadata to keep large scan collections below marimo output limits.
- Scan-family filters (`sxdm`, `mesh`, `ascan`) on readers, overview figures and tables, with matching dropdowns in both marimo examples.
- `get_detector_frame` and a reactive marimo scan/counter/detector browser with a synthetic demo.
- GitHub Actions Python matrix and synthetic tests for imports, measured scan geometry and multiprocessing.

### Changed

- Core imports no longer load ipywidgets, XSOCS or beamline geometry packages.
- Detector reductions stream bounded batches, default to in-process execution and cap worker counts to tasks.
- Terminal progress bars work outside Jupyter; legacy widgets remain optional Jupyter interfaces.
- Fitting/geometry libraries are optional; ESRF-specific integrations are documented separately.

### Validation limits

- External ESRF/XSOCS integration fixtures are not bundled. These tests skip explicitly when unavailable.
