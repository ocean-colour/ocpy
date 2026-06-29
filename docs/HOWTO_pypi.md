# HOWTO: Release ocpy to PyPI

This guide walks through publishing the **ocpy** package
(distribution name `ocpy-ocean`, import name `ocpy`) to the Python
Package Index. It reflects the current state of the repo after the
packaging work in `prompts/pip.md` (Tasks 1–3).

---

## 0. Current state (what is already done)

These are in place and need no further work:

- ✅ `pyproject.toml` — PEP 621 metadata, trimmed dependencies, dynamic
  version from `ocpy/__init__.py:__version__`, the `ocpy_view` and
  `ocpy_plot_rrs` console entry points, and `package-data` globs that
  bundle `ocpy/data/**`.
- ✅ `setup.py` — thin shim (all metadata lives in `pyproject.toml`).
- ✅ `MANIFEST.in` — controls the source distribution (sdist) contents.
- ✅ `.github/workflows/tests.yml` — CI runs the unit tests on 3.10/3.11/3.12.
- ✅ `README.md` — long description rendered on the PyPI project page.
- ✅ `LICENSE` — BSD.
- ✅ The pre-rename `oceancolor` imports have all been migrated to `ocpy`
  (Task 2), so a fresh `pip install` will not fail on a missing
  `oceancolor` package.
- ✅ The previously failing `test_ph` / `test_ls2` are fixed (Task 3).
- ✅ Build verified: `python -m build` produces a valid sdist + wheel
  (`ocpy_ocean-0.1.0.tar.gz` / `ocpy_ocean-0.1.0-py3-none-any.whl`) with
  the `ocpy/data/**` files and both console entry points included.

Unlike the sibling `bing` package, `ocpy` has **no git-only hard
dependency** — every runtime dependency (numpy, scipy, pandas, xarray,
scikit-learn, etc.) is on PyPI, so `pip install ocpy-ocean` is
self-contained.

---

## 1. Before the first upload

### 1a. Confirm the distribution name

`ocpy-ocean` is set in `pyproject.toml`. The bare name `ocpy` is already
taken on PyPI by an unrelated project, which is why we distribute under
`ocpy-ocean`. Confirm the chosen name is still free / owned by you:

```bash
pip index versions ocpy-ocean   # "No matching distribution found" => available
pip index versions ocpy         # shows the unrelated project (do NOT use)
```

As of this writing `ocpy-ocean` returns "No matching distribution
found" (i.e. available). If it has since been taken, pick another name
and update `[project].name`.

### 1b. Pick the release version

Currently `ocpy/__init__.py:__version__ = "0.1.0"`. Decide the real
first-release version and set it there (the single source of truth).
PyPI versions are **immutable** — you cannot re-upload the same version.

> Tip: clean any stale editable install first. A previous `setup.py`
> install may linger as `ocpy 0.1.dev0`; run
> `pip uninstall -y ocpy ocpy-ocean` and remove `*.egg-info` so the build
> picks up the correct name/version.

### 1c. Runtime data is NOT bundled (document, not a blocker)

Some modules need external datasets that are intentionally **not** shipped
in the wheel:

- `OS_COLOR` env var → Loisel et al. (2023) Hydrolight data
  (Dryad doi:10.6076/D1630T).
- Tara parquet tables → see `ocpy/data/Tara/README.md`.

These are already documented in `README.md`; just make sure the release
notes mention them so users are not surprised by "file not found".

### 1d. Cosmetic: the WOPP GUI

`ocpy/water/WOPP/WOPP_gui.py` is Python-2 only (`Tkinter` / `tkFileDialog`)
and is shipped in the wheel. Per the Q&A it is inherited and unused, so it
is harmless — it only errors if someone imports it. No action required;
exclude it from the package later if you want a cleaner wheel.

---

## 2. One-time account setup

1. Create accounts on **TestPyPI** (https://test.pypi.org) and
   **PyPI** (https://pypi.org). Enable 2FA on both.
2. Create an **API token** for each (Account Settings → API tokens).
   Scope it to "Entire account" for the first upload, then re-scope to the
   project afterwards.
3. Store credentials in `~/.pypirc` (or use the token at the `twine`
   prompt, or env vars `TWINE_USERNAME=__token__` / `TWINE_PASSWORD=<token>`):

   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-AgEI...        # your PyPI token

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-AgEI...        # your TestPyPI token
   ```

   `chmod 600 ~/.pypirc` to protect it.

---

## 3. Install the build tooling

In the `ocean14` env, `build` is already installed but `twine` is **not**:

```bash
conda activate ocean14
pip install --upgrade build twine
```

(`python -m build` uses an isolated build env and fetches `setuptools` /
`wheel` automatically, so you do not need `wheel` installed globally.)

---

## 4. Build the distributions

From the repo root:

```bash
# Clean any stale artifacts first
rm -rf dist build *.egg-info

python -m build        # writes dist/ocpy_ocean-<version>.tar.gz and .whl
```

---

## 5. Validate the artifacts

```bash
twine check dist/*     # verifies metadata + checks the README renders
```

Optionally confirm the data files and entry points are bundled:

```bash
unzip -l dist/*.whl | grep -E 'ocpy/data/|entry_points'
```

You should see e.g. `ocpy/data/LS2/LS2_LUT.npz` and the
`ocpy_view` / `ocpy_plot_rrs` console scripts.

---

## 6. Test on TestPyPI first (strongly recommended)

```bash
twine upload --repository testpypi dist/*
```

Then install from TestPyPI into a fresh environment and smoke-test. Use
the main PyPI as the fallback index so the real dependencies resolve
(TestPyPI does not mirror numpy/scipy/etc.):

```bash
conda create -n ocpy-test python=3.11 -y
conda activate ocpy-test
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  ocpy-ocean

python -c "import ocpy; print(ocpy.__version__)"
ocpy_view --help
ocpy_plot_rrs --help
```

---

## 7. Upload to the real PyPI

Once TestPyPI looks good:

```bash
twine upload dist/*
```

Verify the project page at `https://pypi.org/project/ocpy-ocean/` and a
clean install:

```bash
pip install ocpy-ocean
```

---

## 8. Tag the release in git

Keep the git tag in sync with the published version:

```bash
git tag -a v0.1.0 -m "ocpy 0.1.0 — first PyPI release"
git push origin v0.1.0
```

Optionally create a GitHub Release from the tag (changelog + attach the
sdist/wheel). The repo already has a Zenodo DOI badge, so a GitHub Release
will also mint a new Zenodo archive version.

---

## 9. (Optional) Automate releases with GitHub Actions

Use PyPI **Trusted Publishing** (OIDC, no long-lived tokens):

1. On PyPI: project → Settings → Publishing → add a trusted publisher
   pointing at `ocean-colour/ocpy`, workflow `publish.yml`,
   environment `pypi`.
2. Add `.github/workflows/publish.yml` that triggers on
   `release: types: [published]`, runs `python -m build`, and uses
   `pypa/gh-action-pypi-publish` (which needs `permissions: id-token:
   write`). No API token is stored in the repo.

---

## 10. Post-release housekeeping

- Bump `__version__` to the next dev version (e.g. `0.1.1.dev0`) so future
  builds are not mistaken for the release.
- Update the changelog (`docs/changelog.rst`).
- Confirm the ReadTheDocs build (`ocpy`) picks up the tag.

---

## Quick reference (the happy path)

```bash
conda activate ocean14
pip install --upgrade build twine
rm -rf dist build *.egg-info
python -m build
twine check dist/*
twine upload --repository testpypi dist/*    # test first
twine upload dist/*                          # then real PyPI
git tag -a v<version> -m "ocpy <version>" && git push origin v<version>
```
