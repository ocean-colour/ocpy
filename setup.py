# Thin shim retained for backward-compatible / editable installs.
#
# All project metadata and dependencies now live in pyproject.toml,
# which is the single source of truth for packaging.  setuptools reads
# that file automatically; this file exists only so that tooling which
# still invokes `python setup.py ...` continues to work.

from setuptools import setup

if __name__ == '__main__':
    setup()
