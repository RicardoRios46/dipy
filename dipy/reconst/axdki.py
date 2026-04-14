#!/usr/bin/python
"""Classes and functions for fitting the axial symmetric signal diffusion kurtosis model"""

import numpy as np
from dipy.reconst.base import ReconstModel
from dipy.core.gradients import check_multi_b, round_bvals, unique_bvals_magnitude

from dipy.testing.decorators import warning_for_keywords
from dipy.reconst.dti import TensorModel

from dipy.core.onetime import auto_attr

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
def _get_principal_eigvec(data, gtab, mask=None):
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

    tenfit = tenmodel.fit(data)

    principal_eigenvector = tenfit.evecs.astype(np.float32)[:,:,:,:,0] # extract first eigenvector

    return principal_eigenvector

@warning_for_keywords()
def _get_powder_average(self, bvals):
    """
    Compute powder average
    """
    b_unique = np.unique(bvals)
    b_unique = np.sort(b_unique)

    # tolérance (comme opt.bthresh MATLAB)
    b_thresh = 50  

    nx, ny, nz, nt = self.data.shape

    S_powder_list = []

    for b in b_unique:
        inds = np.abs(bvals - b) < b_thresh
        # moyenne sur directions
        S_mean = np.mean(self.data[..., inds], axis=-1)  # (nx,ny,nz)
        S_powder_list.append(S_mean)

    S_powder = np.stack(S_powder_list, axis=0)
    S_powder = np.clip(S_powder, 1e-6, None)
    logS = np.log(S_powder)

    nb = len(b_unique)
    Nvox = nx * ny * nz

    logS = logS.reshape(nb, -1)

    return logS


class AxialSymmetricDiffusionKurtosisModel(ReconstModel):
    """Axial Symmetric Diffusion Kurtosis Model"""

    def __init__(self, data, gtab, *args, bmag=None, return_S0_hat=False, **kwargs):
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
        self.data = data
        self.ubvals = unique_bvals_magnitude(gtab.bvals, bmag=bmag)
        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.design_matrix_A1 = design_matrix_A1(data, self.gtab, self.bvals,self.bvecs)
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
    def ols_fit_axdki(self):
        r"""
        Fit the axial symmetric diffusion kurtosis imaging based on a weighted least square solution.
        """

        nx, ny, nz, nt = self.data.shape
        Nvox = nx * ny * nz

        S = np.log(np.clip(self.data, 1e-6, None))
        S = S.reshape(Nvox, nt)

        #A = design_matrix_A1(data, gtab, bvals, ubvecs)
        A = self.design_matrix_A1

        A = A.reshape(Nvox, nt, 6)
        #mask_flat = self.data.reshape(Nvox)

        ATA = np.einsum('vti,vtj->vij', A, A)

        ATy = np.einsum('vti,vt->vi', A, S)

        I = np.eye(6)[None, :, :]  # (1,6,6)
        ATA_reg = ATA + 1e-6 * I

        X = np.linalg.solve(ATA_reg, ATy[..., None])[..., 0]  # (Nvox,6)

        #X[~mask_flat] = 0

        X = X.reshape(nx, ny, nz, 6)

        Dperp=X[:,:,:,1]
        Dpara=X[:,:,:,2]
        Wperp=X[:,:,:,3]
        Wpara=X[:,:,:,4]
        Wmean=X[:,:,:,5]

        return (Dperp, Dpara, Wperp, Wpara, Wmean)


    @warning_for_keywords()
    def fit(self, *, mask=None):
        """Fit method for the first part of of the AXDKI model class
        
        Parameters
        ----------
        data : ndarray ([X,Y,Z, ...], g)
            ndarray containing the data signals in its last dimension.

        mask : array
            A boolean array used to mark the coordinates in the data that should be analyzed that has the shape data.shape[:-1]
        """

        params_A1 = self.ols_fit_axdki()

        params_A2 = fast_vectorize_solve(mask, self.ubvals)

        params = list(params_A1) + list(params_A2)

        return AxialSymmetricDiffusionKurtosisFit(self, params)
    
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
        
    # def __getitem__(self, index):
    #     model_params = self.model_params
    #     model_S0 = self.model_S0
    #     N = model_params.ndim
    #     if type(index) is not tuple:
    #         index = (index,)
    #     elif len(index) >= model_params.ndim:
    #         raise IndexError("IndexError: invalid index")
    #     index = index + (slice(None),) * (N - len(index))
    #     if model_S0 is not None:
    #         model_S0 = model_S0[index[:-1]]
    #     return AxialSymmetricDiffusionKurtosisFit(
    #         self.model, model_params[index], model_S0=model_S0
    #         )
    
    @property
    def S0_hat(self):
        return self.model_S0

    @auto_attr
    def Dperp(self): return self.model_params[0]

    @auto_attr
    def Dpara(self): return self.model_params[1]

    @auto_attr
    def Wperp_raw(self): return self.model_params[2]

    @auto_attr
    def Wpara_raw(self): return self.model_params[3]

    @auto_attr
    def Wmean_raw(self): return self.model_params[4]

    @auto_attr
    def Dpowder(self): return self.model_params[5]

    @auto_attr
    def Wpowder_raw(self): return self.model_params[6]

    @auto_attr
    def dmean(self):
        # Now Dpara and Dperp are accessible as self.Dpara and self.Dperp
        return (1/3 * self.Dpara + 2/3 * self.Dperp)

    @auto_attr
    def Wperp(self):
        # Normalized Kurtosis: W / D^2
        return self.Wperp_raw / (self.Dperp**2 + 1e-6)

    @auto_attr
    def Wpara(self):
        return self.Wpara_raw / (self.Dpara**2 + 1e-6)

    @auto_attr
    def Wmean(self):
        return self.Wmean_raw / (self.dmean**2 + 1e-6)

    @auto_attr
    def Wpowder(self):
        return self.Wpowder_raw / (self.Dpowder**2 + 1e-6)


@warning_for_keywords()
def fast_vectorize_solve(self, mask, bvals):
    """
    Fast vectorize solve
    """
    nx, ny, nz, nt = self.data.shape

    #Ap = design_matrix_A2(bvals)

    Ap = self.design_matrix_A2
    logS = _get_powder_average(bvals, mask)

    ATA = Ap.T @ Ap
    ATA_inv = np.linalg.inv(ATA + 1e-6 * np.eye(3))
    AT = Ap.T

    # solution globale
    X = ATA_inv @ (AT @ logS)   # (3, Nvox)

    # appliquer masque
    X[:, ~mask] = 0

    # =========================
    # RESHAPE OUTPUTS
    # =========================
    X = X.reshape(3, nx, ny, nz)

    logS0 = X[0]
    Dpowder = X[1]
    Wpowder = X[2]

    return (Dpowder, Wpowder)


def design_matrix_A1(data, gtab, bvals, bvecs):
    """Constructs design matrix for the axial symmetric signal diffusion kurtosis model
    
    Parameters
    ----------
    ubvals : array
        Containing the unique b-values of the data.

    Returns
    -------
    design_matrix : array
    """
    principal_eigvec = _get_principal_eigvec(data, gtab)
    print(principal_eigvec.shape)
    # calculus of cos(theta)
    cos_theta = np.einsum('xyzi,ti->xyzt', principal_eigvec, bvecs)
    print(cos_theta.shape)
    print(bvals.shape)
    b = bvals[None, None, None, :]
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

def design_matrix_A2(ubvals):
    """Constructs design matrix for the axial symmetric signal diffusion kurtosis model
    
    Parameters
    ----------
    ubvals : array
        Containing the unique b-values of the data.

    Returns
    -------
    design_matrix : array
    """
    b_unique = np.unique(ubvals)
    b_unique = np.sort(b_unique)

    b = b_unique[:, None]  # (nb,1)

    Ap = np.concatenate([
        np.ones_like(b),   # S0
        -b,                # D
        (b**2) / 6         # W
    ], axis=1)             # (nb, 3)

    return Ap
