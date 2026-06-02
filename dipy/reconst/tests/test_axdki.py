#!/usr/bin/python
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_equal, assert_raises

from dipy.core.gradients import gradient_table
from dipy.core.onetime import auto_attr

# Import the newly implemented functions and classes from your module
# (Assuming the script is saved as dipy/reconst/axdki.py)
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


def setup_synthetic_axdki_data():
    """Helper utility to generate structured multi-shell data for tests.

    Creates a 3-shell acquisition scheme along 32 directions (plus b0s)
    and populates a tiny 3D volumetric grid (2, 2, 2).
    """
    # 1. Build an acquisition scheme containing 3 distinct shells
    bvals = np.array(
        [0, 0, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 3000, 3000, 3000, 3000]
    )

    # Unit vectors corresponding to basic spatial alignments
    bvecs = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.707, 0.707, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.707, 0.707, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.707, 0.707, 0],
        ]
    )

    gtab = gradient_table(bvals, bvecs)

    # 2. Build a stable mock array: shape=(2, 2, 2, 14)
    np.random.seed(42)
    data = np.zeros((2, 2, 2, len(bvals)))

    # Compute signal decay pattern based on a baseline S0=100
    for i in range(len(bvals)):
        if bvals[i] == 0:
            data[..., i] = 100.0 + np.random.normal(0, 0.1, (2, 2, 2))
        else:
            # Signal decreases predictably with higher b-values
            decay = np.exp(-bvals[i] * 0.001 + (bvals[i] ** 2 / 6.0) * 0.0002)
            data[..., i] = 100.0 * decay + np.random.normal(0, 0.1, (2, 2, 2))

    return data, gtab


def test_design_matrices():
    """Verify both part-1 (directional) and part-2 (powder average) matrices."""
    data, gtab = setup_synthetic_axdki_data()

    # Test matrix A1 construction
    mock_eigvec = np.zeros((2, 2, 2, 3))
    mock_eigvec[..., 0] = 1.0  # Orient along x-axis

    A1 = design_matrix_A1(mock_eigvec, gtab)
    # Target shape needs to match: (Voxels_dims, directions, 6 parameters)
    assert_equal(A1.shape, (2, 2, 2, len(gtab.bvals), 6))
    assert_array_almost_equal(A1[..., 0, 0], 1.0)  # Check baseline intercept column

    # Test matrix A2 construction
    ubvals = np.array([0, 1000, 2000, 3000])
    A2 = design_matrix_A2(ubvals)
    assert_equal(A2.shape, (4, 3))  # 4 unique shells, 3 components
    assert_array_almost_equal(A2[:, 0], np.ones(4))
    assert_array_almost_equal(A2[:, 1], -ubvals)


def test_powder_average_pipeline():
    """Check directional pooling inside powder average calculator."""
    data, gtab = setup_synthetic_axdki_data()
    unique_shells = np.unique(gtab.bvals)

    logS = _get_powder_average(data, gtab.bvals)

    # Output dimension must map: (number of unique shells, total flat voxels)
    expected_voxels = data.shape[0] * data.shape[1] * data.shape[2]
    assert_equal(logS.shape, (len(unique_shells), expected_voxels))
    assert_equal(np.isnan(logS).any(), False)


def test_axdki_model_and_fit():
    """End-to-end evaluation of the double fitting workflow execution."""
    data, gtab = setup_synthetic_axdki_data()

    # Construct the model wrapper
    model = AxialSymmetricDiffusionKurtosisModel(gtab)

    # Standard complete fit execution without an active mask
    fit = model.fit(data)

    assert isinstance(fit, AxialSymmetricDiffusionKurtosisFit)

    # Ensure shape maps to 7 output indices matching the structural metric properties
    # 5 from Matrix A1 (Dperp, Dpara, Wperp_raw, Wpara_raw, Wmean_raw)
    # 2 from Matrix A2 (Dpowder, Wpowder_raw)
    assert_equal(len(fit.model_params), 7)
    assert_equal(fit.model_params[0].shape, data.shape[:-1])

    # Test metric property calls to guarantee equations are running properly
    assert_equal(fit.Dperp.shape, data.shape[:-1])
    assert_equal(fit.Dpara.shape, data.shape[:-1])
    assert_equal(fit.dmean.shape, data.shape[:-1])

    # Validate normalization equations (W_norm = W_raw / D^2)
    # Ensure no NaN errors due to divisions via the epsilon anchor
    assert_equal(np.isnan(fit.Wperp).any(), False)
    assert_equal(np.isnan(fit.Wpara).any(), False)
    assert_equal(np.isnan(fit.Wmean).any(), False)
    assert_equal(np.isnan(fit.Wpowder).any(), False)


def test_axdki_masking():
    """Verify parameter assignment restriction when applying masks."""
    data, gtab = setup_synthetic_axdki_data()
    model = AxialSymmetricDiffusionKurtosisModel(gtab)

    # Construct active tracking mask blocking exactly half the volume
    mask = np.ones(data.shape[:-1], dtype=bool)
    mask[0, :, :] = False

    fit = model.fit(data, mask=mask)

    # Reconstructions inside masked out spatial pockets should yield zero values
    for parameter_map in fit.model_params:
        assert_array_almost_equal(parameter_map[~mask], 0.0)


def test_insufficient_bvals_exception():
    """Check configuration checks fail if data has fewer than 2 b-shells."""
    single_shell_bvals = np.array([0, 0, 1000, 1000])
    single_shell_bvecs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
    invalid_gtab = gradient_table(single_shell_bvals, single_shell_bvecs)

    # Instantiation should crash as check_multi_b thresholds at 2 shells
    with assert_raises(ValueError):
        AxialSymmetricDiffusionKurtosisModel(invalid_gtab)