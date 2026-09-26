"""Tests for the LS2 inversion model.

The oracle is ``files/LS2_test_run.csv``, the authors' own reference vector
(``LS2_test_run.xls`` from the SIO Ocean Optics Research Laboratory's
``LS2_Distribution``) converted once into a dependency-free CSV so the tests
need neither ``xlrd`` nor ``openpyxl``.  It holds 60 cells: 10 samples at each
of 412, 443, 490, 510, 555 and 670 nm, with every input column alongside the
MATLAB outputs.  The inputs are read back out of it rather than duplicated as
literals, so the test and its oracle cannot drift apart.

The reference was produced by the authors' single-pass code, so every
comparison against it pins ``max_iter=1``.  Convergence of the iterated mode is
tested separately, by self-consistency.

Reference:

Loisel, H., D. Stramski, D. Dessailly, C. Jamet, L. Li and R. A. Reynolds
(2018), An inverse model for estimating the optical absorption and
backscattering coefficients of seawater from remote-sensing reflectance over a
broad range of oceanic and coastal marine environments, *J. Geophys. Res.
Oceans*, 123, 2141-2171, doi:10.1002/2017JC013632.

Original test script: M. Kehrli, R. A. Reynolds and D. Stramski, October 2022.
"""

import pathlib
import time

import numpy as np
import pandas
import pytest

from ocpy.ls2.io import load_LUT
from ocpy.ls2.ls2_main import (LS2_calc_kappa, LS2_main, LS2_seek_pos,
                               ls2_invert)

#: Wavelengths of the reference vector, in the order its sheets were written.
WAVES = (412., 443., 490., 510., 555., 670.)


def data_path(filename):
    data_dir = pathlib.Path(__file__).parent.absolute().joinpath('files')
    # TODO: This really should have the `.resolve()`, but it crashes the
    #       Windows/python3.9 CI test (only that one).  When PypeIt advances
    #       to python>=3.10, reinstate the last part of the following line:
    return str(data_dir.joinpath(filename))


@pytest.fixture(scope='module')
def lut():
    """The LS2 look-up tables, materialised out of the lazy npz once."""
    npz = load_LUT()
    return {key: np.asarray(npz[key]) for key in npz.files}


@pytest.fixture(scope='module')
def reference():
    """The authors' reference vector as ``(inputs, outputs)`` ``(10, 6)`` blocks.

    Returns
    -------
    inputs : dict
        ``Rrs``, ``Kd``, ``bp`` of shape ``(10, 6)``; ``aw``, ``bw``, ``wave``
        of shape ``(6,)``; ``sza`` of shape ``(10,)``.
    outputs : dict
        ``a``, ``anw``, ``bb``, ``bbp``, ``kappa``, each ``(10, 6)``, with the
        blanks of the CSV carried through as NaN.
    """
    df = pandas.read_csv(data_path('LS2_test_run.csv'))
    blocks = [df[df['Input wavelength [nm]'] == w].reset_index(drop=True)
              for w in WAVES]

    def _col(name):
        return np.column_stack([b[name].values for b in blocks])

    inputs = dict(
        Rrs=_col('Input Rrs [1/sr]'),
        Kd=_col('Input Kd [1/m]'),
        bp=_col('Input bp [1/m]'),
        aw=np.array([b['Input aw [1/m]'][0] for b in blocks]),
        bw=np.array([b['Input bw [1/m]'][0] for b in blocks]),
        sza=blocks[0]['Input sza [deg]'].values,
        wave=np.array(WAVES),
    )
    outputs = dict(
        a=_col('Ouput a [1/m]'),      # the CSV inherits the original typo
        anw=_col('Output anw [1/m]'),
        bb=_col('Output bb [1/m]'),
        bbp=_col('Output bbp [1/m]'),
        kappa=_col('Output kappa [dim]'),
    )
    return inputs, outputs


def _invert(inputs, lut, **kwargs):
    """Call :func:`ls2_invert` on the reference inputs."""
    return ls2_invert(inputs['Rrs'], inputs['Kd'], inputs['aw'], inputs['bw'],
                      inputs['bp'], inputs['sza'], inputs['wave'], lut,
                      **kwargs)


def test_ls2_run(reference, lut):
    """All four coefficients reproduce the authors' reference vector.

    This is the test the stale-corner defect failed: the earlier port
    recomputed only ``a00`` in the Raman branch and reused three stale corners,
    so ``bb`` agreed and ``a`` did not.
    """
    inputs, ref = reference
    res = _invert(inputs, lut, raman=True, max_iter=1, clip_negative=False)

    np.testing.assert_allclose(res.a, ref['a'], rtol=1e-9)
    np.testing.assert_allclose(res.bb, ref['bb'], rtol=1e-9)


def test_ls2_run_kappa(reference, lut):
    """kappa is finite exactly where the reference is, and NaN in its 14 blanks."""
    inputs, ref = reference
    res = _invert(inputs, lut, raman=True, max_iter=1, clip_negative=False)

    finite = np.isfinite(ref['kappa'])
    assert finite.sum() == 46 and (~finite).sum() == 14

    np.testing.assert_allclose(res.kappa[finite], ref['kappa'][finite],
                               rtol=1e-9)
    assert np.all(np.isnan(res.kappa[~finite]))
    # Every blank is a cell whose bb/a left the table's admissible range.
    assert np.array_equal(res.kappa_out_of_range, ~finite)


def test_ls2_run_nonwater(reference, lut):
    """anw = a - aw and bbp = bb - bw/2, including where the reference is blank.

    The MATLAB test script replaced negative ``anw`` by NaN before writing --
    hence 11 blanks -- but wrote three negative ``bbp`` values through.  With
    ``clip_negative=False`` we reproduce both: the three negatives exactly, and
    a negative value under each of the 11 blanks.
    """
    inputs, ref = reference
    res = _invert(inputs, lut, raman=True, max_iter=1, clip_negative=False)

    # bbp has no blanks, and its three negative entries must match.
    assert np.isfinite(ref['bbp']).all()
    assert (ref['bbp'] < 0).sum() == 3
    np.testing.assert_allclose(res.bbp, ref['bbp'], rtol=1e-9)
    np.testing.assert_allclose(res.bbp, res.bb - inputs['bw'] / 2., rtol=1e-12)

    finite = np.isfinite(ref['anw'])
    assert finite.sum() == 49 and (~finite).sum() == 11
    np.testing.assert_allclose(res.anw[finite], ref['anw'][finite], rtol=1e-9)
    assert np.all(res.anw[~finite] < 0.)
    assert np.all(res.negative[~finite])


def test_ls2_clip_negative(reference, lut):
    """``clip_negative`` NaNs each coefficient independently, as MATLAB does."""
    inputs, ref = reference
    raw = _invert(inputs, lut, raman=True, max_iter=1, clip_negative=False)
    clipped = _invert(inputs, lut, raman=True, max_iter=1, clip_negative=True)

    assert np.all(np.isnan(clipped.anw[raw.anw < 0.]))
    assert np.all(np.isnan(clipped.bbp[raw.bbp < 0.]))
    # a and bb are positive everywhere here, so they survive untouched.
    np.testing.assert_allclose(clipped.a, raw.a, rtol=1e-12)
    np.testing.assert_allclose(clipped.bb, raw.bb, rtol=1e-12)


def test_scalar_wrapper_matches_vectorized(reference, lut):
    """``LS2_main`` is a faithful thin wrapper, and returns its 5-tuple."""
    inputs, _ = reference
    res = _invert(inputs, lut, raman=True, max_iter=1, clip_negative=True)

    scalar = np.array([[LS2_main(inputs['sza'][i], inputs['wave'][j],
                                 inputs['Rrs'][i, j], inputs['Kd'][i, j],
                                 inputs['aw'][j], inputs['bw'][j],
                                 inputs['bp'][i, j], lut, 1)
                        for j in range(len(WAVES))]
                       for i in range(len(inputs['sza']))])
    assert scalar.shape == (10, 6, 5)

    for k, name in enumerate(('a', 'anw', 'bb', 'bbp', 'kappa')):
        np.testing.assert_allclose(scalar[:, :, k], getattr(res, name),
                                   rtol=1e-12, equal_nan=True,
                                   err_msg=f'scalar {name} differs')


def test_raman_off_is_the_first_pass(reference, lut):
    """``raman=False`` returns the uncorrected solution, with kappa of 1."""
    inputs, _ = reference
    off = _invert(inputs, lut, raman=False)
    zero = _invert(inputs, lut, raman=True, max_iter=0)

    np.testing.assert_allclose(off.a, zero.a, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(off.bb, zero.bb, rtol=1e-12, equal_nan=True)
    assert np.all(off.kappa == 1.)
    assert np.all(off.n_iter == 0)
    assert not off.not_converged.any()


def test_raman_iteration_converges(reference, lut):
    """The Raman iteration converges well inside its cap, and is a fixed point.

    Measured on the reference vector: every cell whose kappa stays in range
    converges in 2 to 5 passes against a cap of 10, and a further pass then
    moves ``bb/a`` by less than the tolerance.  The planning round estimated
    four passes; five is what the stated criterion actually costs.
    """
    inputs, _ = reference
    tol = 1e-3
    res = _invert(inputs, lut, raman=True, tol=tol, max_iter=10)
    usable = ~res.kappa_out_of_range

    assert usable.sum() == 43
    assert not res.not_converged[usable].any()
    assert res.n_iter[usable].min() >= 2
    assert res.n_iter[usable].max() <= 6      # measured 5, cap is 10

    # One more pass than the iteration took must move bb/a by less than tol.
    extra = _invert(inputs, lut, raman=True, tol=tol,
                    max_iter=int(res.n_iter[usable].max()) + 1)
    moved = np.abs(extra.bb / extra.a - res.bb / res.a) / np.abs(res.bb / res.a)
    assert np.nanmax(moved[usable]) < tol

    # Iterating changes the answer enough to be worth reporting: the authors'
    # single pass leaves bb about 0.3% (median) low against convergence.
    single = _invert(inputs, lut, raman=True, max_iter=1)
    shift = np.abs(res.bb - single.bb) / np.abs(single.bb)
    assert 1e-4 < np.nanmedian(shift[usable]) < 1e-1


def test_kappa_leaving_range_on_a_later_pass(reference, lut):
    """A late kappa failure halts that cell and keeps the last completed pass.

    Iterating moves ``bb/a``, so a cell admissible on the first pass can leave
    the table's range on a later one.  Three of the reference vector's cells do
    -- all at 670 nm, where ``bb/a`` is smallest -- taking the usable count from
    46 at ``max_iter=1`` down to 43 under iteration.  The documented behaviour
    is to stop there, keep the coefficients of the last completed pass, report
    ``kappa`` as NaN and set ``kappa_out_of_range``.  On the first pass that
    reduces exactly to the authors' behaviour, which
    :func:`test_ls2_run_kappa` pins.
    """
    inputs, _ = reference
    single = _invert(inputs, lut, raman=True, max_iter=1)
    iterated = _invert(inputs, lut, raman=True, max_iter=10)

    late = iterated.kappa_out_of_range & ~single.kappa_out_of_range
    assert late.sum() == 3
    assert np.all(inputs['wave'][np.argwhere(late)[:, 1]] == 670.)

    assert np.all(np.isnan(iterated.kappa[late]))
    assert np.all(iterated.n_iter[late] >= 1)      # a correction was applied
    assert np.all(np.isfinite(iterated.a[late]))   # and its result was kept
    assert np.all(iterated.not_converged[late])


def test_kappa_is_applied_to_the_original_rrs(reference, lut):
    """The correction is not cumulative (Loisel & Stramski 2000, Eq. 22).

    One converged pass from the final kappa must reproduce the final answer;
    if kappa were applied to an already-corrected Rrs it would not.
    """
    inputs, _ = reference
    res = _invert(inputs, lut, raman=True, max_iter=10)
    usable = ~res.kappa_out_of_range

    redo = ls2_invert(inputs['Rrs'] * res.kappa, inputs['Kd'], inputs['aw'],
                      inputs['bw'], inputs['bp'], inputs['sza'],
                      inputs['wave'], lut, raman=False)
    np.testing.assert_allclose(redo.a[usable], res.a[usable], rtol=1e-12)
    np.testing.assert_allclose(redo.bb[usable], res.bb[usable], rtol=1e-12)


def _single(lut, *, sza=30., wave=490., Rrs=0.004, Kd=0.05, aw=0.015,
            bw=0.003, bp=0.15, **kwargs):
    """One cell through :func:`ls2_invert`, for the edge-case tests."""
    return ls2_invert([[Rrs]], [[Kd]], [aw], [bw], [[bp]], [sza], [wave],
                      lut, **kwargs)


def test_grid_edges_are_finite(lut):
    """eta on its last node and sza at 70 degrees both return a solution.

    Both used to raise ``UnboundLocalError``: the bracketing scan never
    assigned an index for a value sitting exactly on the final node.  ``sza =
    70`` is worse than that -- the stored ``muw`` nodes are rounded to six
    decimals, so it lands 4.9e-7 *outside* the table and has to be snapped back
    onto the edge.
    """
    # eta = bw / (bp + bw) = 0.2 exactly, the last eta node.
    at_eta = _single(lut, bw=0.003, bp=4 * 0.003, raman=False)
    assert np.isfinite(at_eta.a).all() and not at_eta.off_grid.any()

    at_muw = _single(lut, sza=70., raman=False)
    assert np.isfinite(at_muw.a).all() and not at_muw.off_grid.any()

    # And the scalar helper agrees.
    assert LS2_seek_pos(0.2, lut['eta'], 'eta') == 19
    assert LS2_seek_pos(lut['muw'].ravel()[-1], lut['muw'], 'muw') == 6


def test_off_grid_returns_nan_not_none(lut):
    """Past the edge every output is NaN and ``off_grid`` says so.

    The earlier port used a bare ``return`` here, handing the caller ``None``
    and breaking its 5-tuple unpacking.
    """
    # eta > 0.2 requires bp < 4 bw; clear blue water routinely fails this.
    res = _single(lut, bw=0.003, bp=0.001, raman=True)
    assert res.off_grid.all()
    for name in ('a', 'anw', 'bb', 'bbp', 'kappa'):
        assert np.all(np.isnan(getattr(res, name))), name

    out = LS2_main(30., 490., 0.004, 0.05, 0.015, 0.003, 0.001, lut, 1)
    assert len(out) == 5
    assert all(np.isnan(v) for v in out)

    # sza beyond the muw table is off-grid too.
    assert np.isnan(LS2_seek_pos(0.5, lut['muw'], 'muw'))
    assert _single(lut, sza=80., raman=False).off_grid.all()


def test_kappa_is_nan_above_the_table(lut):
    """Above 702 nm kappa is NaN, not the clamped value ``np.interp`` gives.

    The Raman table stops at 702 nm.  A hyperspectral corpus running to 750 nm
    must be told the correction is unavailable there rather than handed the
    702 nm row.
    """
    lam = lut['kappa'][:, 0]
    assert lam.max() == 702.

    res = _single(lut, wave=750., Rrs=5e-4, Kd=0.5, aw=2.8, bw=6e-4, bp=0.1,
                  raman=True, max_iter=1)
    assert np.all(np.isnan(res.kappa))
    assert res.kappa_out_of_range.all()

    # The scalar helper too, at a bb/a that is admissible at 702 nm.
    ratio = float(np.mean(lut['kappa'][-1, 5:7]))
    assert np.isfinite(LS2_calc_kappa(ratio, 702., lut['kappa']))
    assert np.isnan(LS2_calc_kappa(ratio, 703., lut['kappa']))
    assert np.isnan(LS2_calc_kappa(ratio, 301., lut['kappa']))


def test_lut_limiting_relation(lut):
    """``a1 * muw = 1`` at every LUT node, the paper's Rrs -> 0 limit.

    Eq. 9 gives ``a -> Kd / a1`` as ``Rrs -> 0``, and the paper's limiting
    relation is ``a -> Kd * muw``.  The published table satisfies this to
    4.4e-6 across all 21 x 8 = 168 nodes, which is a useful invariant to hold
    any re-derived coefficients to.
    """
    a1 = lut['a'][:, :, 0]
    muw = lut['muw'].ravel()
    assert a1.shape == (21, 8)

    product = a1 * muw[None, :]
    assert np.abs(product - 1.).max() < 1e-5
    assert np.abs(product - 1.).max() > 1e-7   # it is not exact; do not assume so


def test_vectorized_speed(lut):
    """800k cells run in seconds, not the 18.5 minutes the scalar path costs.

    The scalar entry point re-reads the npz, rebuilds two
    ``RegularGridInterpolator`` objects and evaluates the kappa cubic at all
    101 table rows for every cell.  The wall-clock bound below is loose enough
    for a slow shared runner and still two orders of magnitude under that.
    """
    rng = np.random.default_rng(0)
    n_samples, n_waves = 10_000, 80
    wave = np.linspace(400., 700., n_waves)

    start = time.time()
    res = ls2_invert(rng.uniform(1e-4, 8e-3, (n_samples, n_waves)),
                     rng.uniform(0.02, 0.5, (n_samples, n_waves)),
                     rng.uniform(0.005, 2., n_waves),
                     rng.uniform(5e-4, 7e-3, n_waves),
                     rng.uniform(0.02, 0.4, (n_samples, n_waves)),
                     rng.uniform(0., 60., n_samples), wave, lut,
                     raman=True, max_iter=10)
    elapsed = time.time() - start

    assert res.a.shape == (n_samples, n_waves)
    assert res.counts()['n_cells'] == n_samples * n_waves
    assert elapsed < 60., f'800k cells took {elapsed:.1f} s'


def test_no_warning_storm(lut, recwarn):
    """Negative, off-grid and out-of-range cells are counted, never warned about.

    One warning per cell is 800k warnings on the benchmark corpus, which hides
    everything else in the log.
    """
    rng = np.random.default_rng(1)
    res = ls2_invert(rng.uniform(-1e-3, 8e-3, (200, 20)),
                     rng.uniform(0.02, 0.5, (200, 20)),
                     rng.uniform(0.005, 2., 20),
                     rng.uniform(5e-4, 7e-3, 20),
                     rng.uniform(0.0005, 0.4, (200, 20)),
                     rng.uniform(0., 75., 200),
                     np.linspace(400., 750., 20), lut, raman=True)

    counts = res.counts()
    assert counts['off_grid'] > 0
    assert counts['negative'] > 0
    assert counts['kappa_out_of_range'] > 0
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]
