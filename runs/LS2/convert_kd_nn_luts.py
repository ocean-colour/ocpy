"""Convert the authors' Kd neural-network LUTs and reference vectors for ocpy.

Regenerates, from a clone of the SIO Ocean Optics Research Laboratory's
``Kd_NN_Distribution`` (MIT licence):

* ``ocpy/data/LS2/Kd_NN_LUT_MODIS_v1.1.npz`` -- commit ``a5c7ec2`` (2023-10-10),
  byte-identical in its weights to the ``weights_1/2.csv`` and
  ``train_switch.csv`` ocpy has always shipped.
* ``ocpy/data/LS2/Kd_NN_LUT_MODIS_v1.3.npz`` -- commit ``19c2501`` (2025-04-15),
  the authors' current MODIS network.
* ``ocpy/data/LS2/Kd_NN_LUT_PACE_v2.3.npz`` -- commit ``982b52c`` (2025-04-15),
  the authors' PACE network (reference vector from ``fef4d15``, 2025-04-20).
* ``ocpy/tests/files/Kd_NN_test_run_{MODIS_v1.1,MODIS_v1.3,PACE_v2.3}.csv`` --
  the matching reference vectors, as dependency-free CSVs.

The ``.mat`` files hold MATLAB ``table`` objects, which ``scipy.io`` cannot
decode, so this script needs ``mat-io`` and ``xlrd``.  Neither is an ocpy
dependency; install them somewhere private, e.g.::

    pip install --target /tmp/pylib mat-io xlrd
    PYTHONPATH=/tmp/pylib python runs/LS2/convert_kd_nn_luts.py /path/to/Kd_NN_Distribution

The clone must have full history (``git fetch --unshallow``) because the v1.1
files are read from an old commit.

Each NPZ holds, per branch (``clear_`` and ``turbid_`` prefixes), the
NaN-stripped LUT columns ``b1``, ``b2``, ``bout``, ``w1``, ``w2`` and
``wout`` in MATLAB storage order, so that ``reshape(w1, nc1, ne)`` in MATLAB is
``w1.reshape((nc1, ne), order='F')`` in numpy.  ``train_mean`` and
``train_std`` are the full ``train_switch`` columns.  ``version`` and
``source_commit`` record provenance.  The architectures, input orders and
activation functions are code, not data, and live in ``ocpy.ls2.kd_nn``.
"""

import argparse
import pathlib
import subprocess
import tempfile

import numpy as np

#: (output tag, sensor, LUT commit, reference-vector commit, version string)
RELEASES = (
    ('MODIS_v1.1', 'MODIS', 'a5c7ec2', 'a5c7ec2', '1.1'),
    ('MODIS_v1.3', 'MODIS', '19c2501', '982b52c', '1.3'),
    # The PACE reference at 982b52c carries MODIS column headers and only five
    # Rrs columns; fef4d15 corrects the sheet without changing its Kd.
    ('PACE_v2.3', 'PACE', '982b52c', 'fef4d15', '2.3'),
)

OCPY = pathlib.Path(__file__).resolve().parents[2]


def _git_show(repo, commit, path, dest):
    """Write ``commit:path`` from ``repo`` to ``dest``."""
    blob = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{path}'],
                          check=True, capture_output=True).stdout
    dest.write_bytes(blob)


def convert(repo):
    """Write the NPZ LUTs and the reference CSVs.

    Parameters
    ----------
    repo : pathlib.Path
        A full-history clone of ``Kd_NN_Distribution``.
    """
    from matio import load_from_mat   # dev-only dependency
    import pandas

    data_dir = OCPY / 'ocpy' / 'data' / 'LS2'
    test_dir = OCPY / 'ocpy' / 'tests' / 'files'

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for tag, sensor, lut_commit, ref_commit, version in RELEASES:
            mat = tmp / f'{tag}.mat'
            _git_show(repo, lut_commit, f'Kd_NN_LUT_{sensor}.mat', mat)
            lut = load_from_mat(str(mat))[f'Kd_NN_LUT_{sensor}']

            arrays = {}
            for branch, key in (('clear', 'weights_1'), ('turbid', 'weights_2')):
                table = np.ravel(lut[key])[0]
                for col in ('b1', 'b2', 'bout', 'w1', 'w2', 'wout'):
                    values = table[col].to_numpy(dtype=float)
                    arrays[f'{branch}_{col}'] = values[np.isfinite(values)]
            switch = np.ravel(lut['train_switch'])[0]
            arrays['train_mean'] = switch['MEAN'].to_numpy(dtype=float)
            arrays['train_std'] = switch['STD'].to_numpy(dtype=float)
            arrays['version'] = np.array(version)
            arrays['source_commit'] = np.array(lut_commit)
            np.savez(data_dir / f'Kd_NN_LUT_{tag}.npz', **arrays)

            xls = tmp / f'{tag}.xls'
            _git_show(repo, ref_commit, f'Kd_NN_test_run_{sensor}.xls', xls)
            ref = pandas.read_excel(xls, engine='xlrd')
            ref.to_csv(test_dir / f'Kd_NN_test_run_{tag}.csv', index=False,
                       float_format='%.17g')
            print(f'{tag}: LUT {lut_commit}, reference {ref_commit}, '
                  f'{len(ref)} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('repo', type=pathlib.Path,
                        help='full-history clone of Kd_NN_Distribution')
    convert(parser.parse_args().repo)
