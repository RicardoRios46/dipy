#!/usr/bin/python
"""Classes and functions for fitting the axial symmetric signal diffusion kurtosis model"""

import numpy as np
from dipy.reconst.base import ReconstModel
from dipy.core.gradients import round_bvals

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

    A = design_matrix(round_bvals(gtab.bvals))

@warning_for_keywords()
def _get_principal_eigvec(gtab, mask):
    """
    Compute the principal eigenvector for the first part of the algorithm.

    """

    tenmodel = TensorModel(gtab, fit_method="WLS")

    tenfit = tenmodel.fit(mask)

    principal_eigenvector = tenfit.evecs.astype(np.float32)[:,:,:,:,0] # extract first eigenvector

    return principal_eigenvector

class AxialSymmetricDiffusionKurtosisModel(ReconstModel):
    """Axial Symmetric Diffusion Kurtosis Model"""

    def __init__(self, gtab, *args, **kwargs):
        """Axial Symmetric Diffusion Kurtosis Model
        
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
        self.args = args
        self.kwargs = kwargs

    @warning_for_keywords()
    def fit(self, data, *, mask=None):
        """Fit method of the AXDKI model class
        
        Parameters
        ----------
        data : ndarray ([X,Y,Z, ...], g)
            ndarray containing the data signals in its last dimension.

        mask : array
            A boolean array used to mark the coordinates in the data that should be analyzed that has the shape data.shape[:-1]
        """

        principal_eigenvector = _get_principal_eigvec(data, mask)

        params = wls_fit_axdki(
            
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
def wls_fit_axdki(
    gtab,
    mask=None
):
    r"""
    Fit the axial symmetric diffusion kurtosis imaging based on a weighted least square solution.
    """

    tenmodel = TensorModel(gtab, fit_method="WLS")

    return tenmodel.fit(mask)

def design_matrix(ubvals):
    """Constructs design matrix for the axial symmetric signal diffusion kurtosis model
    
    Parameters
    ----------
    ubvals : array
        Containing the unique b-values of the data.

    Returns
    -------
    design_matrix : array
    """
    nb = ubvals.shape
    B = np.zeros(nb + (3,))
    return B