# Prompt: package ocpy's data files so they ship on install

## Goal

Make ocpy's bundled data files (under `ocpy/data/**`, 34 files) install with the
package, so code that reads them via `importlib.resources.files('ocpy')` works on
a fresh `pip install` (wheel) — not just from a source checkout.

## The problem

`ocpy/satellites/pace.py::gen_noise_vector` does:

```python
from importlib.resources import files
pace_file = files('ocpy').joinpath('data/satellites/PACE_error.csv')
PACE_errors = pandas.read_csv(pace_file)
```

On a pip-installed ocpy (e.g. `pip install git+https://github.com/ocean-colour/ocpy`)
that file is **absent** → `FileNotFoundError: .../site-packages/ocpy/data/satellites/PACE_error.csv`.
Same for every other reader of `ocpy/data/**` (MODIS/SeaWiFS matchups, water
coefficients, Bricaud/phytoplankton tables, COASTLOOC, …).

## Root cause

`setup.py` ships only Python packages and declares no data:

- `packages = find_packages()` — `ocpy/data/` has **no `__init__.py`**, so it is
  not a package and `find_packages()` skips it entirely.
- There is **no** `package_data`, **no** `include_package_data=True`, and **no**
  `MANIFEST.in`.

So the install contains only `.py` files; the 34 data files are dropped.

## The fix

1. **Add `MANIFEST.in`** at the repo root (controls the sdist + is honored by
   `include_package_data`):

   ```
   recursive-include ocpy/data *
   ```

2. **Edit `setup.py`** — add data-file inclusion to `setup_keywords`:

   ```python
   setup_keywords['include_package_data'] = True
   setup_keywords['package_data'] = {'ocpy': ['data/**/*']}
   ```

   (`package_data` makes the wheel include the files even though `ocpy/data` is
   not itself a package; `include_package_data` + `MANIFEST.in` covers the sdist.
   Belt-and-suspenders so both `pip install .` and `pip install git+…` work.)

3. *(Optional hygiene, while you're in `setup.py`)* remove the long-deprecated
   `use_2to3` and `setup_requires=['pytest-runner']` keys — both are obsolete on
   modern setuptools.

## Verify

Build a wheel and confirm the data file is inside it, then install clean and
exercise the reader:

```bash
# 1) the data file is in the built wheel
python -m pip install --upgrade build
python -m build --wheel
python - <<'PY'
import zipfile, glob
whl = sorted(glob.glob('dist/ocpy-*.whl'))[-1]
names = zipfile.ZipFile(whl).namelist()
hits = [n for n in names if n.startswith('ocpy/data/')]
print(f'{len(hits)} data files in wheel; PACE present:',
      any(n.endswith('data/satellites/PACE_error.csv') for n in hits))
PY

# 2) fresh install + the actual reader works
python -m venv /tmp/ocpy_check && /tmp/ocpy_check/bin/pip install -q dist/ocpy-*.whl
/tmp/ocpy_check/bin/python - <<'PY'
import numpy as np
from importlib.resources import files
print('file installed:', files('ocpy').joinpath('data/satellites/PACE_error.csv').is_file())
from ocpy.satellites import pace
print('gen_noise_vector OK:', pace.gen_noise_vector(np.array([443., 555.])))
PY
```

Both should report `True` / print a noise vector.

## Why it matters (downstream)

IOPtics' `pace` noise model (the L23 first-pass) calls `gen_noise_vector`, so
without this fix the model can't run on any fresh install, and IOPtics CI has to
skip-guard its PACE test (`needs_pace`). Fixing ocpy lets that path run for real.

## Prompts

1. Perform the necessary changes to properly package ocpy data.  If you have any questions, write them in the Q&A section below. Log your work.

## Q&A

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-27 (Packaged ocpy's bundled data files)

Executed Polishing/packaging prompt 1. Made the `ocpy/data/**` files install
with the package so `importlib.resources.files('ocpy')` resolves them on a
fresh `pip install`, not just from a source checkout.

**Changes.**
- Added `MANIFEST.in` at the repo root with `recursive-include ocpy/data *`
  (covers the sdist).
- Edited `setup.py`:
  - `include_package_data = True` and
    `package_data = {'ocpy': ['data/**/*']}` so both sdist and wheel ship the
    34 data files even though `ocpy/data` has no `__init__.py`.
  - Removed the deprecated `use_2to3 = False` and
    `setup_requires = ['pytest-runner']` keys (obsolete on modern
    setuptools), per the optional-hygiene note.

**Adjacent fix (needed to meet the goal).** The doc's verification —
`pace.gen_noise_vector(...)` on a fresh install — failed even after the data
file was present, with `ModuleNotFoundError: No module named 'IPython'`.
Cause: `ocpy/satellites/pace.py` (and 15 other modules) do a top-level
`from IPython import embed` as a debugging idiom, but `ipython` was not in
`install_requires`, so a fresh `pip install` never pulled it in (it only
worked in `ocean14` because IPython happens to be installed there). The
import runs at module-load time, so the module can't even be imported without
it.

Per JXP's direction, the fix is to **declare the dependency** rather than
strip the imports: added `'ipython'` to `install_requires` in `setup.py`.
This unblocks all 16 modules at once. (An earlier pass had instead removed
the unused import from `pace.py`; that was reverted so the file stays
consistent with the rest of the codebase.)

**Verification (all green).**
- Built the wheel (`python -m build --wheel`): **35** `ocpy/data/` entries
  inside, `PACE_error.csv` present.
- Built the sdist: data files present too (MANIFEST.in path).
- Fresh `python -m venv` + `pip install dist/ocpy-*.whl`:
  `files('ocpy').joinpath('data/satellites/PACE_error.csv').is_file()` →
  `True`; `ipython` is now pulled in automatically (v9.15.0); and
  `pace.gen_noise_vector([443., 555.])` → `[0.00040105 0.00018155]`.

No questions for the Q&A section — the prompt's plan was complete and
unambiguous.
