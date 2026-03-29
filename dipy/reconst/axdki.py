#!/usr/bin/python
"""Classes and functions for fitting the axial symmetric signal diffusion kurtosis model"""

import numpy as np
from dipy.reconst.base import ReconstModel
from dipy.core.gradients import check_multi_b, round_bvals, unique_bvals_magnitude

from dipy.testing.decorators import warning_for_keywords
from dipy.reconst.dti import TensorModel

@warning_for_keywords()
def axdki_predictions(axdki_params, gtab, *, S0=1):
    """
    Predict the axial symmetric signal given the parameters of the axial symmetric DKI, an GradientTable object and S0 signal.

    Parameters
    ----------
    axdki_params : ndarray ([X,Y,Z,...], 2)
        Array containing the axial symmetric diffusivity and axial symmetric signal kurtosis in its last axis.
        gtab : a GradientTable class instance
            The gradient table for this prediction
        S0 : float or ndarray, optional
            The non diffusion-weighted signal in every voxel, or across all voxels.

    References
    ----------
    .. footbibliography::
    """

    

@warning_for_keywords()
def _get_principal_eigvec(gtab, mask):
    """
    Compute the principal eigenvector for the first part of the algorithm.

    Parameters
    ----------
        gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
        m
        mask : ndarray, optional
            A boolean array used to mark the coordinates in the data that should be
            analyzed that has the same shape of the axdki parameters
    """

    tenmodel = TensorModel(gtab, fit_method="WLS")

    tenfit = tenmodel.fit(mask)

    principal_eigenvector = tenfit.evecs.astype(np.float32)[:,:,:,:,0] # extract first eigenvector

    return principal_eigenvector

class AxialSymmetricDiffusionKurtosisModel(ReconstModel):
    """Axial Symmetric Diffusion Kurtosis Model"""

    def __init__(self, gtab, *args, bmag=None, return_S0_hat=False, **kwargs):
        """
        Axial Symmetric Diffusion Kurtosis Model
        
        Parameters
        ----------
        gtab : GradientTable class instance
            Gradient table.
        
        args, kwargs : arguments and keyword arguments passed to the fit_method. See msdki.wls_fit_msdki for details.

        References
        ----------
        .. footbibliography::

        """

        ReconstModel.__init__(self, gtab)

        self.return_S0_hat = return_S0_hat
        self.ubvals = unique_bvals_magnitude(gtab.bvals, bmag=bmag)
        self.design_matrix_A1 = design_matrix_A1(self.ubvals)
        self.design_matrix_A2 = design_matrix_A2(self.ubvals)
        self.bmag = bmag
        self.args = args
        self.kwargs = kwargs

        # Check if at least three b-values are given
        enough_b = check_multi_b(self.gtab, 2, non_zero=False, bmag=bmag)
        if not enough_b:
            mes = "The `min_signal` key-word argument needs to be strictly"
            e_s += " postiive."
            raise ValueError(e_s)


    @warning_for_keywords()
    def fit(self, data, *, mask=None):
        """Fit method for the first part of of the AXDKI model class
        
        Parameters
        ----------
        data : ndarray ([X,Y,Z, ...], g)
            ndarray containing the data signals in its last dimension.

        mask : array
            A boolean array used to mark the coordinates in the data that should be analyzed that has the shape data.shape[:-1]
        """

        principal_eigenvector = _get_principal_eigvec(data, mask)

        params = ols_fit_axdki(
            
        )
    
    @warning_for_keywords()
    def predict(self, axdki_params, *, S0=1.0):
        """
        Predict a signal for this AxialSymmetricDiffusionKurtosisModel class instance given parameters.

        Parameters
        ----------
        axdki_params : ndarray
            The parameters of the axial symmetric signal diffusion kurtosis model
        S0 : float or ndarray, optional
            The non diffusion-weighted signal is every voxel, or across all voxels.

        Returns
        -------
        S : (..., N) ndarray
            Simulated axial symmetric signal based on the axial symmetric signal diffusion kurtosis model

        References
        ----------
        .. footbibliography::
        """
        return axdki_predictions(axdki_params, self.gtab, S0=S0)

class AxialSymmetricDiffusionKurtosisFit:
    @warning_for_keywords()
    def __init__(self, model, model_params, *, model_S0=None):
        """Initialize a AxialSymmetricDiffusionKurtosisFit class instance."""
        self.model = model
        self.model_params = model_params
        self.model_S0 = model_S0
        


@warning_for_keywords()
def ols_fit_axdki(mask):
    r"""
    Fit the axial symmetric diffusion kurtosis imaging based on a weighted least square solution.
    """

    nx, ny, nz, nt = mask.shape
    Nvox = nx * ny * nz

    S = np.log(np.clip(mask, 1e-6, None))
    S = S.reshape(Nvox, nt)

    A = A.reshape(Nvox, nt, 6)
    mask_flat = mask.reshape(Nvox)

    ATA = np.einsum('vti,vtj->vij', A, A)

    ATy = np.einsum('vti,vt->vi', A, S)

    I = np.eye(6)[None, :, :]  # (1,6,6)
    ATA_reg = ATA + 1e-6 * I

    X = np.linalg.solve(ATA_reg, ATy[..., None])[..., 0]  # (Nvox,6)

    X[~mask_flat] = 0

    X = X.reshape(nx, ny, nz, 6)

    Dperp=X[:,:,:,1]
    Dpara=X[:,:,:,2]
    Wperp=X[:,:,:,3]
    Wpara=X[:,:,:,4]
    Wmean=X[:,:,:,5]

    return None

def design_matrix_A1(ubvals, principal_eigenvec, bvecs):
    """Constructs design matrix for the axial symmetric signal diffusion kurtosis model
    
    Parameters
    ----------
    ubvals : array
        Containing the unique b-values of the data.

    Returns
    -------
    design_matrix : array
    """
    # calculus of cos(theta)
    cos_theta = np.einsum('xyzi,ti->xyzt', principal_eigenvec, bvecs)

    b = ubvals[None, None, None, :]
    c = cos_theta
    c2 = c**2
    c4 = c**4

    A = np.empty(c.shape + (6,), dtype=c.dtype)

    A[..., 0] = 1
    A[..., 1] = -b * (1 - c2)
    A[..., 2] = -b * c2
    A[..., 3] = (b**2 / 6) * (5*c4 - 6*c2 + 1)
    A[..., 4] = (b**2 / 6) * (0.5 * c2 * (5*c2 - 3))
    A[..., 5] = (b**2 / 6) * (-15/2 * (c4 - c2))

    return A

def design_matrix_A2(ubvals, principal_eigenvec, bvecs):
    """Constructs design matrix for the axial symmetric signal diffusion kurtosis model
    
    Parameters
    ----------
    ubvals : array
        Containing the unique b-values of the data.

    Returns
    -------
    design_matrix : array
    """
    # calculus of cos(theta)
    return None