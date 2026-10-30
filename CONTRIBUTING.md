# Contributing

Develop against the GitHub fork and keep changes small enough to review.

```bash
uv sync --locked --extra marimo --extra analysis --extra jupyter
uv run --no-sync pytest -q
uv run --no-sync ruff check sxdm tests examples
uv run --no-sync marimo check examples/scan_browser.py
uv build
```

Python 3.11, 3.12, 3.13, and 3.14 are checked by GitHub Actions. The core job installs no extras; synthetic tests must not depend on ESRF data or optional notebook imports. Integration tests should explicitly skip absent external data and write outputs only under temporary directories.

Update `uv.lock` with `uv lock` when changing dependencies. Use PEP 621 dependencies/extras and PEP 735 dependency groups; keep GitLab packages out of the universal lock. Avoid eager imports of optional GUI or beamline packages. Plotting functions should return ordinary figures, not create servers or force a notebook backend.

For the marimo example, also test execution using synthetic data:

```bash
uv run examples/create_demo.py /tmp/sxdm-contributor-demo.h5
uv run --extra marimo marimo export html examples/scan_browser.py \
    -o /tmp/sxdm-browser.html -- --dataset /tmp/sxdm-contributor-demo.h5
```

A static export checks startup; interactive slider/button behavior should also be checked in `marimo edit`. Real facility data should remain local.

The legacy GitLab CI configuration uses uv too; notebook execution is disabled during documentation builds because the original tutorials require external beamline data.
