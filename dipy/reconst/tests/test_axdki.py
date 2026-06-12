#!/usr/bin/python
"""Tests for the Axial Symmetric Diffusion Kurtosis Imaging (AXDKI) model.

Follows dipy testing conventions: numpy.testing assertions, module-level
fixtures, and isolated single-concern test functions.
"""

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere

from dipy.reconst.axdki import (
    AxialSymmetricDiffusionKurtosisFit,
    AxialSymmetricDiffusionKurtosisModel,
    _get_powder_average,
    _get_principal_eigvec,
    axdki_predictions,
    design_matrix_A1,
    design_matrix_A2,
    fast_vectorize_solve,
    ols_fit_axdki,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_gtab():
    """Multi-shell GradientTable with isotropically distributed directions.

    Uses dipy's repulsion100 sphere to avoid rank-deficient design matrices.
    Acquisition: 4 b0 + 10 directions at b=1000/2000/3000 s/mm² = 34 volumes.
    """
    sphere = get_sphere(name="repulsion100")
    vecs = sphere.vertices  # (100, 3), well-distributed unit vectors

    bvals = np.concatenate(
        [np.zeros(4), np.ones(10) * 1000, np.ones(10) * 2000, np.ones(10) * 3000]
    )
    bvecs = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (4, 1)),  # b0 placeholder directions
            vecs[:10],
            vecs[10:20],
            vecs[20:30],
        ]
    )
    return gradient_table(bvals, bvecs=bvecs)


def _make_data(gtab, shape=(3, 3, 3), seed=42):
    """Synthetic signal following a monoexponential decay with kurtosis.

    Parameters
    ----------
    gtab : GradientTable
    shape : tuple of int
        Spatial dimensions of the volume.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : ndarray, shape (*shape, n_directions)
    """
    rng = np.random.default_rng(seed)
    bvals = gtab.bvals
    n_dirs = len(bvals)
    data = np.zeros((*shape, n_dirs))

    for i, b in enumerate(bvals):
        if b == 0:
            data[..., i] = 100.0 + rng.normal(0, 0.5, shape)
        else:
            # Simple isotropic DKI signal: S = S0 * exp(-b*D + b²/6 * D²*K)
            D, K = 1e-3, 1.0
            decay = np.exp(-b * D + (b ** 2 / 6.0) * D ** 2 * K)
            data[..., i] = 100.0 * decay + rng.normal(0, 0.5, shape)

    return data


# Module-level gtab and data reused across tests
_GTAB = _make_gtab()
_DATA = _make_data(_GTAB)


# ---------------------------------------------------------------------------
# Design matrix tests
# ---------------------------------------------------------------------------

class TestDesignMatrices:
    """Unit tests for design_matrix_A1 and design_matrix_A2."""

    def test_A1_shape(self):
        """A1 must have shape (X, Y, Z, n_dirs, 6)."""
        nx, ny, nz = _DATA.shape[:3]
        n_dirs = len(_GTAB.bvals)
        eigvec = np.zeros((nx, ny, nz, 3))
        eigvec[..., 0] = 1.0  # x-aligned principal eigenvector

        A1 = design_matrix_A1(eigvec, _GTAB)

        npt.assert_equal(A1.shape, (nx, ny, nz, n_dirs, 6))

    def test_A1_intercept_column_is_ones(self):
        """First column of A1 (intercept) must be identically 1."""
        nx, ny, nz = _DATA.shape[:3]
        eigvec = np.zeros((nx, ny, nz, 3))
        eigvec[..., 0] = 1.0

        A1 = design_matrix_A1(eigvec, _GTAB)

        npt.assert_array_equal(A1[..., 0], 1.0)

    def test_A1_b0_rows_are_intercept_only(self):
        """For b=0 directions all decay columns must be zero."""
        nx, ny, nz = _DATA.shape[:3]
        eigvec = np.zeros((nx, ny, nz, 3))
        eigvec[..., 0] = 1.0

        A1 = design_matrix_A1(eigvec, _GTAB)
        b0_mask = _GTAB.bvals == 0

        # Columns 1-5 encode b-dependent terms — must vanish at b=0
        npt.assert_array_almost_equal(A1[..., b0_mask, 1:], 0.0)

    def test_A2_shape(self):
        """A2 must have shape (n_unique_shells, 3)."""
        ubvals = np.array([0, 1000, 2000, 3000], dtype=float)
        A2 = design_matrix_A2(ubvals)
        npt.assert_equal(A2.shape, (4, 3))

    def test_A2_first_column_ones(self):
        """First column of A2 (S0 term) must be all ones."""
        ubvals = np.array([0, 1000, 2000, 3000], dtype=float)
        A2 = design_matrix_A2(ubvals)
        npt.assert_array_almost_equal(A2[:, 0], np.ones(4))

    def test_A2_diffusivity_column(self):
        """Second column of A2 must equal -b (negative b-values)."""
        ubvals = np.array([0, 1000, 2000, 3000], dtype=float)
        A2 = design_matrix_A2(ubvals)
        npt.assert_array_almost_equal(A2[:, 1], -ubvals)

    def test_A2_kurtosis_column(self):
        """Third column of A2 must equal b²/6."""
        ubvals = np.array([0, 1000, 2000, 3000], dtype=float)
        A2 = design_matrix_A2(ubvals)
        npt.assert_array_almost_equal(A2[:, 2], ubvals ** 2 / 6.0)


# ---------------------------------------------------------------------------
# Powder average tests
# ---------------------------------------------------------------------------

class TestPowderAverage:
    """Tests for _get_powder_average."""

    def test_output_shape(self):
        """logS shape must be (n_unique_shells, n_voxels_flat)."""
        n_unique = len(np.unique(_GTAB.bvals))
        expected_voxels = int(np.prod(_DATA.shape[:3]))

        logS = _get_powder_average(_DATA, _GTAB.bvals)

        npt.assert_equal(logS.shape, (n_unique, expected_voxels))

    def test_no_nan_in_output(self):
        """Powder-averaged log-signal must not contain NaN values."""
        logS = _get_powder_average(_DATA, _GTAB.bvals)
        assert not np.isnan(logS).any(), "NaN found in powder-averaged log-signal."

    def test_b0_log_signal_close_to_log100(self):
        """Mean powder-averaged log-signal at b=0 must be near log(100)."""
        logS = _get_powder_average(_DATA, _GTAB.bvals)
        # b=0 is the first unique shell
        npt.assert_array_less(np.abs(logS[0] - np.log(100.0)), 0.1)

    def test_signal_decreases_with_b(self):
        """Powder-averaged signal must decrease monotonically with b-value."""
        logS = _get_powder_average(_DATA, _GTAB.bvals)
        # Each row is a shell; mean log-signal should decrease
        shell_means = logS.mean(axis=1)
        for i in range(len(shell_means) - 1):
            assert shell_means[i] > shell_means[i + 1], (
                f"Signal did not decrease between shells {i} and {i+1}: "
                f"{shell_means[i]:.4f} vs {shell_means[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# OLS fit (part 1) tests
# ---------------------------------------------------------------------------

class TestOlsFitAxdki:
    """Tests for ols_fit_axdki."""

    def test_output_tuple_length(self):
        """Must return exactly 5 parameter maps."""
        result = ols_fit_axdki(_DATA, _GTAB)
        assert len(result) == 5

    def test_output_shapes_match_spatial(self):
        """All 5 parameter maps must match the spatial volume shape."""
        result = ols_fit_axdki(_DATA, _GTAB)
        for i, param in enumerate(result):
            npt.assert_equal(
                param.shape,
                _DATA.shape[:3],
                err_msg=f"Parameter map {i} has wrong shape.",
            )

    def test_no_nan_in_parameters(self):
        """OLS fit must not produce NaN in any parameter map."""
        result = ols_fit_axdki(_DATA, _GTAB)
        for i, param in enumerate(result):
            assert not np.isnan(param).any(), f"NaN found in parameter map {i}."


# ---------------------------------------------------------------------------
# Fast powder-average solve (part 2) tests
# ---------------------------------------------------------------------------

class TestFastVectorizeSolve:
    """Tests for fast_vectorize_solve."""

    def test_output_tuple_length(self):
        """Must return exactly 2 parameter maps (Dpowder, Wpowder_raw)."""
        from dipy.core.gradients import unique_bvals_magnitude

        ubvals = unique_bvals_magnitude(_GTAB.bvals)
        result = fast_vectorize_solve(_DATA, ubvals, _GTAB.bvals)
        assert len(result) == 2

    def test_output_shapes_match_spatial(self):
        """Both powder parameter maps must match the spatial volume shape."""
        from dipy.core.gradients import unique_bvals_magnitude

        ubvals = unique_bvals_magnitude(_GTAB.bvals)
        result = fast_vectorize_solve(_DATA, ubvals, _GTAB.bvals)
        for i, param in enumerate(result):
            npt.assert_equal(param.shape, _DATA.shape[:3])

    def test_Dpowder_plausible_range(self):
        """Dpowder must be in the physiologically plausible range (0.3–3 µm²/ms)."""
        from dipy.core.gradients import unique_bvals_magnitude

        ubvals = unique_bvals_magnitude(_GTAB.bvals)
        Dpowder, _ = fast_vectorize_solve(_DATA, ubvals, _GTAB.bvals)
        # In mm²/s units: [0.3e-3, 3e-3]; in s/mm² convention the bvals are ~1000
        assert np.nanmean(Dpowder) > 0, "Mean Dpowder must be positive."


# ---------------------------------------------------------------------------
# Principal eigenvector tests
# ---------------------------------------------------------------------------

class TestGetPrincipalEigvec:
    """Tests for _get_principal_eigvec."""

    def test_output_shape(self):
        """Principal eigenvector must have shape (X, Y, Z, 3)."""
        eigvec = _get_principal_eigvec(_DATA, _GTAB)
        npt.assert_equal(eigvec.shape, (*_DATA.shape[:3], 3))

    def test_unit_norm(self):
        """Each principal eigenvector must be approximately unit-length."""
        eigvec = _get_principal_eigvec(_DATA, _GTAB)
        norms = np.linalg.norm(eigvec, axis=-1)
        npt.assert_array_almost_equal(norms, np.ones(_DATA.shape[:3]), decimal=5)


# ---------------------------------------------------------------------------
# Model and fit tests (end-to-end)
# ---------------------------------------------------------------------------

class TestAxialSymmetricDiffusionKurtosisModel:
    """Integration tests for model instantiation and fitting."""

    def test_fit_returns_correct_type(self):
        """fit() must return an AxialSymmetricDiffusionKurtosisFit instance."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        assert isinstance(fit, AxialSymmetricDiffusionKurtosisFit)

    def test_model_params_count(self):
        """model_params must contain exactly 7 maps (5 from A1, 2 from A2)."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        assert len(fit.model_params) == 7

    def test_parameter_shapes(self):
        """All model_params maps must have the spatial volume shape."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        for i, param in enumerate(fit.model_params):
            npt.assert_equal(
                param.shape,
                _DATA.shape[:3],
                err_msg=f"model_params[{i}] has wrong shape.",
            )

    def test_derived_property_shapes(self):
        """Derived scalar properties must all match the spatial volume shape."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        spatial = _DATA.shape[:3]

        for name in ("Dperp", "Dpara", "dmean", "Dpowder"):
            npt.assert_equal(
                getattr(fit, name).shape,
                spatial,
                err_msg=f"fit.{name} has wrong shape.",
            )

    def test_normalized_kurtosis_no_nan(self):
        """Normalized kurtosis maps must not contain NaN (eps guards division)."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)

        for name in ("Wperp", "Wpara", "Wmean", "Wpowder"):
            assert not np.isnan(getattr(fit, name)).any(), (
                f"NaN found in fit.{name}."
            )

    def test_dmean_formula(self):
        """dmean must equal 1/3 * Dpara + 2/3 * Dperp everywhere."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        expected = 1.0 / 3.0 * fit.Dpara + 2.0 / 3.0 * fit.Dperp
        npt.assert_array_almost_equal(fit.dmean, expected)


# ---------------------------------------------------------------------------
# Masking tests
# ---------------------------------------------------------------------------

class TestAxdkiMasking:
    """Verify that the boolean mask is correctly propagated through the fit."""

    def setup_method(self):
        self.model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        self.mask = np.ones(_DATA.shape[:3], dtype=bool)
        self.mask[0, :, :] = False  # blank out the first slice

    def test_masked_voxels_are_zero(self):
        """Voxels excluded by the mask must have all parameters set to zero."""
        fit = self.model.fit(_DATA, mask=self.mask)
        for i, param in enumerate(fit.model_params):
            npt.assert_array_almost_equal(
                param[~self.mask],
                0.0,
                err_msg=f"model_params[{i}] non-zero inside masked region.",
            )

    def test_unmasked_voxels_are_nonzero(self):
        """At least some unmasked voxels must have non-zero Dperp."""
        fit = self.model.fit(_DATA, mask=self.mask)
        assert np.any(fit.Dperp[self.mask] != 0.0), (
            "All unmasked Dperp values are zero — fit may have failed silently."
        )


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestAxdkiPredictions:
    """Tests for axdki_predictions and model.predict."""

    def _get_pred_params(self, fit):
        eigvec = _get_principal_eigvec(_DATA, _GTAB)
        return (
            fit.Dperp,
            fit.Dpara,
            fit.Wperp_raw,
            fit.Wpara_raw,
            fit.Wmean_raw,
            eigvec,
        )

    def test_predict_shape(self):
        """Predicted signal must have shape (X, Y, Z, n_dirs)."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        pred = model.predict(self._get_pred_params(fit), S0=100.0)
        npt.assert_equal(pred.shape, _DATA.shape)

    def test_predict_all_finite(self):
        """Predicted signal must be finite everywhere."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        pred = model.predict(self._get_pred_params(fit), S0=100.0)
        assert np.isfinite(pred).all(), "Non-finite values in predicted signal."

    def test_predict_b0_equals_S0(self):
        """At b=0 the predicted signal must equal S0."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        S0 = 100.0
        pred = model.predict(self._get_pred_params(fit), S0=S0)
        b0_mask = _GTAB.bvals == 0
        npt.assert_array_almost_equal(pred[..., b0_mask], S0, decimal=4)

    def test_predict_positive(self):
        """All predicted signal values must be strictly positive."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        fit = model.fit(_DATA)
        pred = model.predict(self._get_pred_params(fit), S0=100.0)
        assert (pred > 0).all(), "Non-positive values in predicted signal."


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestAxdkiErrorHandling:
    """Tests for model validation and error conditions."""

    def test_single_shell_raises_value_error(self):
        """Model instantiation must raise ValueError for single non-zero shell."""
        bvals = np.ones(6) * 1000  # only one non-zero b-value; no b=0
        sphere = get_sphere(name="repulsion100")
        bvecs = sphere.vertices[:6]
        invalid_gtab = gradient_table(bvals, bvecs=bvecs)

        with pytest.raises(ValueError):
            AxialSymmetricDiffusionKurtosisModel(invalid_gtab)

    def test_fit_with_all_zero_mask_returns_zero_params(self):
        """Fitting with an all-False mask must yield all-zero parameter maps."""
        model = AxialSymmetricDiffusionKurtosisModel(_GTAB)
        mask = np.zeros(_DATA.shape[:3], dtype=bool)
        fit = model.fit(_DATA, mask=mask)

        for i, param in enumerate(fit.model_params):
            npt.assert_array_almost_equal(
                param,
                0.0,
                err_msg=f"model_params[{i}] non-zero with all-False mask.",
            )