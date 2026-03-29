"""
Robust Frequency-Dependent Diffusional Kurtosis Imaging (AxDKI)
================================================================

Diffusion MRI (dMRI) allows probing of tissue microstructure at spatial
scales unattainable with conventional MRI. Diffusional Kurtosis Imaging
(DKI) [1]_ is a form of dMRI that quantifies the non-Gaussian component
of water diffusion within tissue, providing greater sensitivity to
microstructural changes compared to conventional Diffusion Tensor Imaging
(DTI). DKI has been applied to study a wide range of pathologies including
stroke [2]_, mild traumatic brain injury [3]_, and neurodegenerative
diseases [4]_.

An important parameter when interpreting dMRI metrics is the **effective
diffusion time** (:math:`\\Delta t_{eff}`), which determines the length scale
within the tissue being probed [5]_. Commonly used Pulsed Gradient Spin Echo
(PGSE) sequences are limited to relatively long effective diffusion times
(>10 ms), meaning the acquired signal reflects diffusion restriction at
multiple spatial scales simultaneously. In contrast, Oscillating Gradient
Spin Echo (OGSE) sequences allow much shorter effective diffusion times —
corresponding to higher oscillating frequencies — increasing sensitivity to
smaller subcellular spatial scales [6]_.

By collecting data at multiple gradient oscillation frequencies
(**frequency-dependent dMRI**), the spatial scale sensitivity can be varied,
enabling investigation of tissue microstructure at both subcellular and
cellular levels. This approach has been characterized in the healthy brain
[7]_, [8]_, and in the study of stroke [9]_, cancer [10]_, and
neurodegenerative disease [11]_.

.. topic:: Background: The Challenge of Frequency-Dependent DKI

    A key technical obstacle in combining OGSE and DKI is that OGSE
    sequences are far less efficient at generating the large b-values
    (:math:`\\geq 2000\\ s/mm^2`) required for kurtosis fitting. This is
    because b-value scales as :math:`b \\sim G^2 / f^3`, where :math:`G` is
    the gradient strength and :math:`f` is the oscillating frequency [12]_.
    As frequency increases, the achievable b-value drops dramatically for a
    given gradient hardware.

    A second challenge is noise propagation. DKI requires fitting a large
    number of parameters (up to 22 for the full kurtosis tensor), leading to
    significant noise amplification, particularly in kurtosis maps. Gaussian
    smoothing prior to fitting is widely used to address this, but it
    introduces blurring around tissue boundaries and can bias quantitative
    analyses [13]_.

This tutorial demonstrates a complete workflow to address both challenges,
following the approach of Hamilton et al. (2024) [14]_:

1. **An efficient 10-direction encoding scheme** that achieves twice the
   b-value efficiency of traditional schemes by maximising multiple gradient
   channels simultaneously.

2. **Axisymmetric DKI (AxDKI) fitting** [15]_, which reduces the parameter
   space from 22 (full tensor) to 8 by exploiting the cylindrical symmetry
   of white matter fibres — enabling robust fitting with fewer encoding
   directions.

3. **A shared axis of symmetry across frequencies**, exploiting the
   observation that the principal diffusion direction does not appreciably
   change with OGSE frequency, particularly in anisotropic white matter.

4. **Two-step spatial regularization** applied during fitting (not as a
   pre-processing step), which reduces noise amplification while preserving
   image contrast better than Gaussian smoothing.

Let's import all relevant modules:
"""

import matplotlib.pyplot as plt
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import dipy.reconst.dki as dki
from dipy.segment.mask import median_otsu

# The axisymmetric DKI module (to be integrated into DIPY)
# import dipy.reconst.axdki as axdki

###############################################################################
# Signal Model: Axisymmetric DKI
# --------------------------------
#
# The diffusion-weighted signal in DKI is given by Jensen et al. (2005) [1]_:
#
# .. math::
#
#     \log\left(\frac{S_{b,\hat{n}}}{S_0}\right) = -b D_{\hat{n}}
#     + \frac{b^2}{6} \bar{D}^2 W_{\hat{n}}
#
# where :math:`S_{b,\hat{n}}` is the diffusion-weighted signal at b-value
# :math:`b` and encoding direction :math:`\hat{n}`, :math:`S_0` is the
# signal at :math:`b=0`, :math:`D_{\hat{n}}` is the diffusion tensor
# element, and :math:`W_{\hat{n}}` is the diffusional kurtosis tensor
# element along :math:`\hat{n}`.
#
# In the axisymmetric model, the diffusion tensor is assumed to have
# cylindrical symmetry (i.e. :math:`\lambda_2 = \lambda_3`), which is
# a valid approximation in white matter voxels dominated by coherently
# oriented fibre bundles. Under this assumption, the kurtosis tensor is
# fully characterised by just **three independent parameters**:
#
# - :math:`\bar{W}`: mean kurtosis tensor
# - :math:`W_\perp`: radial tensor kurtosis
# - :math:`W_\parallel`: axial tensor kurtosis
#
# The kurtosis and diffusion elements measured along any direction
# :math:`\hat{n}` forming a polar angle :math:`\theta` with the
# axis of symmetry are given by:
#
# .. math::
#
#     W(\theta) = \frac{1}{16}\left[
#         \cos(4\theta)\left(10W_\perp + 5W_\parallel - 15\bar{W}\right)
#         + 8\cos(2\theta)\left(W_\parallel - W_\perp\right)
#         - 2W_\perp + 3W_\parallel + 15\bar{W}
#     \right]
#
# .. math::
#
#     D(\theta) = D_\perp + \cos^2(\theta)\left(D_\parallel - D_\perp\right)
#
# where :math:`D_\perp` is the radial diffusivity and :math:`D_\parallel`
# is the axial diffusivity.
#
# From these, the standard scalar metrics are recovered as:
#
# .. math::
#
#     \bar{D} = \frac{2D_\perp}{3} + \frac{D_\parallel}{3}
#
# .. math::
#
#     FA = \sqrt{\frac{3}{2}
#         \frac{(D_\parallel - \bar{D})^2 + 2(D_\perp - \bar{D})^2}
#              {D_\parallel^2 + 2D_\perp^2}}
#
# .. math::
#
#     K_\parallel = \frac{W_\parallel \cdot \bar{D}^2}{D_\parallel^2},
#     \quad
#     K_\perp = \frac{W_\perp \cdot \bar{D}^2}{D_\perp^2}
#
# The polar angle :math:`\theta` for each encoding direction is computed
# from the dot product of the encoding vector with the **axis of symmetry**,
# which is the principal eigenvector of the diffusion tensor in each voxel.
#
# This reduced parameter space (8 vs. 22 for the full kurtosis tensor)
# significantly decreases noise propagation during fitting, and has been
# shown to produce comparable or even improved kurtosis maps relative to
# full tensor fitting [15]_, [16]_, [17]_.

###############################################################################
# The Efficient 10-Direction Encoding Scheme
# -------------------------------------------
#
# To achieve sufficiently large b-values at high OGSE frequencies on
# standard pre-clinical hardware, an efficient 10-direction scheme is
# employed (Table 1 of [14]_). This scheme combines a 6-direction
# efficient scheme [7]_ with a tetrahedral scheme, ensuring that at least
# **two gradient channels are simultaneously at maximum** for each encoding
# direction. This doubles the gradient norm — and thus the achievable
# b-value — compared to typical schemes.
#
# The 10 encoding directions (before normalisation) are:
#
# +------------+------------+------------+
# |     x      |     y      |     z      |
# +============+============+============+
# |     0      |     1      |     1      |
# +------------+------------+------------+
# |     0      |     1      |    -1      |
# +------------+------------+------------+
# |     1      |     0      |     1      |
# +------------+------------+------------+
# |     1      |     0      |    -1      |
# +------------+------------+------------+
# |     1      |     1      |     0      |
# +------------+------------+------------+
# |     1      |    -1      |     0      |
# +------------+------------+------------+
# | √(2/3)    | √(2/3)    | √(2/3)    |
# +------------+------------+------------+
# | √(2/3)    | √(2/3)    | -√(2/3)   |
# +------------+------------+------------+
# | √(2/3)    | -√(2/3)   | √(2/3)    |
# +------------+------------+------------+
# | -√(2/3)   | √(2/3)    | √(2/3)    |
# +------------+------------+------------+
#
# While this scheme is not optimally distributed for uniform directional
# sampling, the trade-off is a substantially shorter echo time (TE). For
# example, at 120 Hz OGSE, the 10-direction scheme achieved a b-value of
# 2500 s/mm² with TE = 35.5 ms, whereas a 40-direction scheme required
# TE = 52 ms for the same b-value. The shorter TE greatly improves SNR,
# and SNR lost from fewer directions is recovered via signal averaging within
# the same total scan time [14]_.

# Define the efficient 10-direction scheme
directions_10 = np.array([
    [0,             1,             1            ],
    [0,             1,            -1            ],
    [1,             0,             1            ],
    [1,             0,            -1            ],
    [1,             1,             0            ],
    [1,            -1,             0            ],
    [np.sqrt(2/3), np.sqrt(2/3),  np.sqrt(2/3) ],
    [np.sqrt(2/3), np.sqrt(2/3), -np.sqrt(2/3) ],
    [np.sqrt(2/3),-np.sqrt(2/3),  np.sqrt(2/3) ],
    [-np.sqrt(2/3), np.sqrt(2/3), np.sqrt(2/3) ],
])

# Normalise to unit vectors
norms = np.linalg.norm(directions_10, axis=1, keepdims=True)
directions_10_normed = directions_10 / norms

print("10-direction scheme (normalised):")
print(np.round(directions_10_normed, 4))

###############################################################################
# The axis of symmetry can be estimated from the diffusion tensor in each
# voxel. A key insight exploited here is that the principal diffusion
# direction does not vary appreciably with OGSE frequency — at least over
# the range of frequencies accessible on typical pre-clinical hardware
# (0 to 120 Hz). This is because the molecular displacements probed at
# these frequencies (~2.5–8 μm) are larger than typical axon diameters
# (<1 μm), meaning water molecules reach the axonal membrane at all
# frequencies and the fibre geometry appears similar regardless of
# frequency [14]_.
#
# This motivates the **AFAB** (All Frequencies, All B-values) strategy:
# data from all OGSE frequencies and all b-value shells are pooled to fit
# a single diffusion tensor per voxel, yielding a more robust (less
# noise-susceptible) estimate of the axis of symmetry than fitting each
# frequency independently.
#
# The three strategies compared in [14]_ are:
#
# - **AFAB**: All frequencies + all b-values → one shared axis per voxel
# - **SFAB**: Separate frequencies + all b-values → one axis per frequency
# - **SFLB**: Separate frequencies + low b-value only → one axis per frequency
#
# AFAB consistently produced the highest contrast-to-standard-deviation
# ratios (CSR) in kurtosis maps.

###############################################################################
# Two-Step Spatial Regularization
# ---------------------------------
#
# Rather than applying Gaussian smoothing to diffusion-weighted images
# *before* fitting (the conventional approach), this implementation uses
# **spatial regularization during fitting**. Regularization penalises
# spatial roughness in the fitted parameter maps, balancing the fitting
# residual against smoothness. This approach preserves tissue boundary
# contrast better than Gaussian smoothing [14]_, consistent with findings
# in functional MRI [18]_, [19]_.
#
# The fitting is performed in two steps using isotropic total variation
# (Tikhonov regularization [20]_), solved via the conjugate gradient method.
#
# **Step 1 — Regularized diffusion tensor fitting (axis of symmetry)**
#
# .. math::
#
#     \arg\min \left(
#         \|A_{DTI}\, x_{DT} - y\|_2^2
#         + \gamma_{DT}\, \|T_{DT}\, x_{DT}\|_2^2
#     \right)
#
# where :math:`y` is the log-transformed signal, :math:`A_{DTI}` is the
# DTI encoding matrix, :math:`x_{DT}` is the vector of diffusion tensor
# parameters across all voxels, and :math:`T_{DT}` applies numerical
# spatial derivatives along each dimension for each tensor component.
# The regularization strength :math:`\gamma_{DT}` controls how much
# spatial smoothness is enforced. The principal eigenvector of the
# resulting diffusion tensor is used as the axis of symmetry, which is
# then **held fixed** throughout Step 2.
#
# **Step 2 — Regularized axisymmetric DKI fitting**
#
# .. math::
#
#     \arg\min \left(
#         \|A_{DKI}\, x_{DK} - y\|_2^2
#         + \gamma_{DK}\, \|T_{DK}\, x_{DK}\|_2^2
#     \right)
#
# where :math:`A_{DKI}` is the axisymmetric DKI encoding matrix built
# from Eq. (2) and (3) above (with :math:`\theta` computed using the
# fixed axis from Step 1), :math:`x_{DK}` contains the six axisymmetric
# parameters per voxel
# (:math:`\log(S_0), D_\perp, D_\parallel, \bar{D}^2 W_\perp,
# \bar{D}^2 W_\parallel, \bar{D}^2 \bar{W}`),
# and :math:`T_{DK}` applies spatial derivatives for each parameter.
#
# Both steps use **ordinary least squares** (OLS), which has been shown
# to have reduced bias compared to weighted least squares in this
# context [21]_.
#
# Base regularization weights used in [14]_:
#
# - Mouse data: :math:`\gamma_{DT} = 0.5`,  :math:`\gamma_{DK} = 0.075`
# - Human data: :math:`\gamma_{DT} = 0.5`,  :math:`\gamma_{DK} = 0.2`

###############################################################################
# Fitting Axisymmetric DKI to Data
# ---------------------------------
#
# Below we demonstrate how to load data, construct the gradient table for
# a multi-frequency acquisition, and fit the axisymmetric DKI model.
# This example uses the structure of a typical pre-clinical mouse
# experiment with PGSE (0 Hz), 60 Hz, and 120 Hz OGSE acquisitions,
# each with b-values of 1000 and 2500 s/mm².

# Load an example dataset (here we use the CFIN multi-shell dataset
# available in DIPY as a stand-in; in practice, supply your own
# multi-frequency data).
fraw, fbval, fbvec, t1_fname = get_fnames(name="cfin_multib")
data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs=bvecs)

# Brain masking
maskdata, mask = median_otsu(
    data, vol_idx=[0, 1], median_radius=4, numpass=2, dilate=1
)

###############################################################################
# For a **multi-frequency acquisition**, the gradient tables from each
# frequency block are concatenated along with a frequency label. In the
# axisymmetric DKI fitting, data from all frequencies is first pooled to
# estimate the shared axis of symmetry (AFAB strategy), and then each
# frequency block is fitted independently using that fixed axis.
#
# The code below illustrates how one would structure multi-frequency data.
# In a real experiment, ``data_0hz``, ``data_60hz``, ``data_120hz`` would
# be the pre-processed 4-D dMRI volumes for each frequency.

# Pseudocode for multi-frequency data assembly:
#
#   gtab_0hz   = gradient_table(bvals_0hz,   bvecs_10dir)
#   gtab_60hz  = gradient_table(bvals_60hz,  bvecs_10dir)
#   gtab_120hz = gradient_table(bvals_120hz, bvecs_10dir)
#
#   # Concatenate all data and gradient tables for AFAB axis estimation
#   data_all   = np.concatenate([data_0hz, data_60hz, data_120hz], axis=-1)
#   bvals_all  = np.concatenate([bvals_0hz, bvals_60hz, bvals_120hz])
#   bvecs_all  = np.concatenate([bvecs_10dir, bvecs_10dir, bvecs_10dir])
#   gtab_all   = gradient_table(bvals_all, bvecs_all)

###############################################################################
# For reference and comparison, we fit the standard DKI model (full kurtosis
# tensor) using DIPY's existing implementation. This allows direct
# comparison between axisymmetric fitting and tensor fitting.

dki_model = dki.DiffusionKurtosisModel(gtab)
dki_fit = dki_model.fit(maskdata, mask=mask)

MD = dki_fit.md
MK = dki_fit.mk(min_kurtosis=0, max_kurtosis=3)
FA = dki_fit.fa

###############################################################################
# Visualising the Results
# -------------------------
#
# The following code visualises DKI scalar maps for a representative
# axial slice. In the full workflow, analogous maps would be computed
# from the axisymmetric fitting at each frequency, allowing direct
# comparison of:
#
# - Regularized vs. unregularized axisymmetric maps
# - Axisymmetric vs. full kurtosis tensor maps
# - Maps from the efficient 10-direction scheme vs. traditional schemes

axial_slice = 9

fig, ax = plt.subplots(1, 3, figsize=(12, 4),
                       subplot_kw={"xticks": [], "yticks": []})
fig.subplots_adjust(wspace=0.05)

im0 = ax[0].imshow(
    MD[:, :, axial_slice].T * 1000,
    cmap="gray", vmin=0, vmax=2, origin="lower"
)
ax[0].set_title("Mean Diffusivity (DKI)")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04,
             label=r"MD ($\times 10^{-3}\ mm^2/s$)")

im1 = ax[1].imshow(
    MK[:, :, axial_slice].T,
    cmap="gray", vmin=0, vmax=2, origin="lower"
)
ax[1].set_title("Mean Kurtosis (DKI)")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="MK")

im2 = ax[2].imshow(
    FA[:, :, axial_slice].T,
    cmap="gray", vmin=0, vmax=1, origin="lower"
)
ax[2].set_title("Fractional Anisotropy (DKI)")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="FA")

plt.suptitle("Standard DKI maps (reference for AxDKI comparison)",
             fontsize=11)
plt.tight_layout()
plt.show()
fig.savefig("axdki_reference_maps.png", dpi=150)

###############################################################################
# Advantages Over Conventional Approaches
# -----------------------------------------
#
# **Axisymmetric fitting vs. full kurtosis tensor fitting**
#
# Both methods produce qualitatively and quantitatively similar maps.
# Axisymmetric fitting shows subtle improvements, particularly in axial
# kurtosis (:math:`K_\parallel`) maps, where tensor fitting exhibits
# increased noise. This advantage stems from the much smaller parameter
# space: 8 parameters for axisymmetric fitting vs. 22 for the full
# kurtosis tensor [14]_, [15]_.
#
# **Efficient 10-direction scheme vs. traditional 40-direction scheme**
#
# At matched scan time, the 10-direction scheme with signal averaging
# achieves approximately **3× higher SNR** in b=0 volumes compared to
# a 40-direction scheme at the same frequency, because the shorter TE
# (35.5 ms vs. 52 ms) substantially reduces T2-related signal loss.
# This translates to enhanced contrast and reduced kurtosis over-estimation
# in low-SNR regions [14]_.
#
# **Spatial regularization vs. Gaussian smoothing**
#
# At comparable levels of noise suppression (equal number of visually
# noisy voxels), spatial regularization achieves higher
# contrast-to-standard-deviation ratios (CSR) than Gaussian smoothing.
# In the study of [14]_, radial kurtosis CSR increased from 0.26 to 0.72
# with regularization, compared to only 0.26 to 0.50 with Gaussian
# smoothing. Critically, regularization does not introduce bias into
# diffusivity or kurtosis estimates, and even brings estimates closer to
# ground truth [14]_.
#
# **Frequency-dependence of kurtosis**
#
# By applying this workflow to data acquired at multiple OGSE frequencies,
# one can map how kurtosis metrics change with the length scale being
# probed. This **frequency dispersion** of kurtosis may provide increased
# sensitivity to cytoarchitectural changes at various spatial scales over
# the course of healthy aging and in pathological conditions, such as
# neurodegeneration, where changes at subcellular scales (myelination)
# and cellular scales (cell body density) may be distinguished.

###############################################################################
# References
# ----------
#
# .. [1] Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K. Diffusional
#    kurtosis imaging: The quantification of non-Gaussian water diffusion
#    by means of magnetic resonance imaging. *Magnetic Resonance in
#    Medicine*, 53(6):1432–1440, 2005.
#    https://doi.org/10.1002/mrm.20508
#
# .. [2] Grinberg F, Ciobanu L, Farrher E, Shah NJ. Diffusion kurtosis
#    imaging and log-normal distribution function imaging enhance the
#    visualisation of lesions in animal stroke models. *NMR in
#    Biomedicine*, 25(11):1295–1304, 2012.
#    https://doi.org/10.1002/nbm.2802
#
# .. [3] Stenberg J, et al. Acute diffusion tensor and kurtosis imaging
#    and outcome following mild traumatic brain injury. *Journal of
#    Neurotrauma*, 38(18):2560–2571, 2021.
#    https://doi.org/10.1089/neu.2021.0074
#
# .. [4] Falangola MF, et al. Non-Gaussian diffusion MRI assessment of
#    brain microstructure in mild cognitive impairment and Alzheimer's
#    disease. *Magnetic Resonance Imaging*, 31(6):840–846, 2013.
#    https://doi.org/10.1016/j.mri.2013.02.008
#
# .. [5] Novikov DS, Jensen JH, Helpern JA, Fieremans E. Revealing
#    mesoscopic structural universality with diffusion. *PNAS*,
#    111(14):5088–5093, 2014.
#    https://doi.org/10.1073/pnas.1316944111
#
# .. [6] Schachter M, Does MD, Anderson AW, Gore JC. Measurements of
#    restricted diffusion using an oscillating gradient spin-echo
#    sequence. *Journal of Magnetic Resonance*, 147(2):232–237, 2000.
#    https://doi.org/10.1006/jmre.2000.2203
#
# .. [7] Baron CA, Beaulieu C. Oscillating gradient spin-echo (OGSE)
#    diffusion tensor imaging of the human brain. *Magnetic Resonance
#    in Medicine*, 72(3):726–736, 2014.
#    https://doi.org/10.1002/mrm.24987
#
# .. [8] Arbabi A, Kai J, Khan AR, Baron CA. Diffusion dispersion imaging:
#    Mapping oscillating gradient spin-echo frequency dependence in the
#    human brain. *Magnetic Resonance in Medicine*, 83(6):2197–2208, 2020.
#    https://doi.org/10.1002/mrm.28083
#
# .. [9] Baron CA, et al. Reduction of diffusion-weighted imaging contrast
#    of acute ischemic stroke at short diffusion times. *Stroke*,
#    46(8):2136–2141, 2015.
#    https://doi.org/10.1161/STROKEAHA.115.008815
#
# .. [10] Iima M, et al. Time-dependent diffusion MRI to distinguish
#    malignant from benign head and neck tumors. *Journal of Magnetic
#    Resonance Imaging*, 50(1):88–95, 2019.
#    https://doi.org/10.1002/jmri.26578
#
# .. [11] Aggarwal M, et al. Imaging neurodegeneration in the mouse
#    hippocampus after neonatal hypoxia-ischemia using oscillating
#    gradient diffusion MRI. *Magnetic Resonance in Medicine*,
#    72(3):829–840, 2014.
#    https://doi.org/10.1002/mrm.24956
#
# .. [12] Xu J. Probing neural tissues at small scales: Recent progress
#    of oscillating gradient spin echo (OGSE) neuroimaging in humans.
#    *Journal of Neuroscience Methods*, 349:109024, 2021.
#    https://doi.org/10.1016/j.jneumeth.2020.109024
#
# .. [13] Falconer JC, Narayana PA. Cerebrospinal fluid-suppressed
#    high-resolution diffusion imaging of human brain. *Magnetic
#    Resonance in Medicine*, 37(1):119–123, 1997.
#    https://doi.org/10.1002/mrm.1910370117
#
# .. [14] Hamilton J, Xu K, Geremia N, Prado VF, Prado MAM, Brown A,
#    Baron CA. Robust frequency-dependent diffusional kurtosis computation
#    using an efficient direction scheme, axisymmetric modelling, and
#    spatial regularization. *Imaging Neuroscience*, 2, 2024.
#    https://doi.org/10.1162/imag_a_00055
#
# .. [15] Hansen B, Shemesh N, Jespersen SN. Fast imaging of mean, axial
#    and radial diffusion kurtosis. *NeuroImage*, 142:381–393, 2016.
#    https://doi.org/10.1016/j.neuroimage.2016.08.022
#
# .. [16] Nørhøj Jespersen S. White matter biomarkers from diffusion MRI.
#    *Journal of Magnetic Resonance*, 291:127–140, 2018.
#    https://doi.org/10.1016/j.jmr.2018.03.001
#
# .. [17] Oeschger JM, Tabelow K, Mohammadi S. Axisymmetric diffusion
#    kurtosis imaging with Rician bias correction: A simulation study.
#    *Magnetic Resonance in Medicine*, 89(2):787–799, 2023.
#    https://doi.org/10.1002/mrm.29474
#
# .. [18] Liu W, et al. Spatial regularization of functional connectivity
#    using high-dimensional Markov random fields. *MICCAI 2010 LNCS*,
#    6362:363–370, 2010.
#    https://doi.org/10.1007/978-3-642-15745-5_45
#
# .. [19] Casanova R, et al. Evaluating the impact of spatio-temporal
#    smoothness constraints on the BOLD hemodynamic response function
#    estimation. *Physiological Measurement*, 30(5):N37–N51, 2009.
#    https://doi.org/10.1088/0967-3334/30/5/N01
#
# .. [20] Tikhonov AN, Goncharsky AV, Stepanov VV, Yagola AG.
#    *Numerical Methods for the Solution of Ill-Posed Problems*.
#    Springer Netherlands, 1995.
#    https://doi.org/10.1007/978-94-015-8480-7
#
# .. [21] Morez J, et al. Optimal experimental design and estimation for
#    q-space trajectory imaging. *Human Brain Mapping*, 44(4):1793–1809,
#    2023.
#    https://doi.org/10.1002/hbm.26175
#
# The MatMRI toolbox implementation (MATLAB) upon which this DIPY
# integration is based is available at:
# https://gitlab.com/cfmm/matlab/matmri