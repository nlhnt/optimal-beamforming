import numpy as np
import cvxpy
import logging

logger = logging.getLogger("pyforming")


def zfbf(h: np.ndarray, d: np.ndarray = None):
    """
    Calculates the zero-forcing beamforming (ZFBF) vectors for a
    scenario where all or a subset of antennas transmit to each user.

    The references to definitions and equations refer to the following book:

    Emil Bjornson, Eduard Jorswieck, Optimal Resource Allocation in
    Coordinated Multi-Cell Systems, Foundations and Trends in Communications
    and Information Theory, vol. 9, no. 2-3, pp. 113-381, 2013.

    This is version 1.1. (Last edited: 2014-03-26)

    License: This code is licensed under the GPLv2 license. If you in any way
    use this code for research that results in publications, please cite our
    original article listed above.

    INPUT:
    H  =    Kr x Kt*Nt matrix with row index for users and column index
            transmit antennas
    D  =    Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if j:th
            transmit antenna can transmit to user k and zero otherwise

    OUTPUT:
    wZFBF = Kt*Nt x Kr matrix with normalized ZFBF
    """
    # number of users
    kr = h.shape[0]

    # total number of antennas
    n = h.shape[-1]

    # If d matrix is not provided, all antennas can transmit to everyone
    if d is None:
        d = np.tile(np.eye(n), [kr, 1, 1])

    # Pre-allocate an array for SLNR-MAX beamforming
    wZFBF = np.zeros(np.transpose(h.shape), dtype=np.complex128)

    # Compute SLNR-MAX, based on Definition 3.5 in optimal resource allocation
    for k in range(0, kr):
        effective_channel = np.conj(np.transpose(np.matmul(h, d[k, :, :])))
        # np.linalg.lstsq is the numpy solution to right division in MATLAB
        # Compute zero-forcing based on channel inversion
        # Normalization of zero-forcing direction
        # b/A -> solve a.T x.T = b.T instead
        channel_inversion, _, _, _ = np.linalg.lstsq(
            np.matmul(
                np.conj(
                    np.transpose(effective_channel),
                ),
                effective_channel,
            ).transpose(),
            effective_channel.transpose(),
            rcond=-1,
        )
        channel_inversion = channel_inversion.transpose()
        # Normalization of useful channel
        # use the 2-norm, like in MATLAB
        # print(
        #     f"wZFBF shape: {wZFBF.shape}"
        #     f"\nresult shape: {(channel_inversion / np.linalg.norm(channel_inversion, ord=2)).shape}"
        # )
        wZFBF[:, k] = channel_inversion[:, k] / np.linalg.norm(channel_inversion[:, k], ord=2)

    return wZFBF


def test_zfbf_result():
    H = (
        np.asarray(
            [
                [0.013860 + 0.031335j, 1.073221 - 0.940552j, 0.920571 - 1.373000j, 0.442014 - 0.353275j],
                [-0.067678 - 0.514558j, 0.785435 + 0.629878j, 0.230476 + 0.989237j, -1.701096 + 1.125456j],
                [0.868633 + 0.569620j, 0.191321 - 0.177566j, -0.151462 + 0.232256j, 1.203675 + 0.364566j],
                [2.037872 - 0.802488j, -2.043176 - 0.129150j, 0.487697 + 0.379195j, 0.042107 - 0.400414j],
            ]
        )
        .transpose()
        .conj()
    )
    vector = np.asarray(
        [
            [
                0.3879614290224073 + 0.09872028990717077j,
                0.3406564576226037 + 0.05635336231117433j,
                0.2204253968379605 - 0.5742745067676878j,
                -0.3278458767505699 - 0.05156352160548806j,
            ],
            [
                0.3411405788153983 - 0.3162812665800751j,
                0.4526354346504607 - 0.1702665822637207j,
                -0.2110303708731221 + 0.1036004519893354j,
                -0.4495672529458771 + 0.3576682857477722j,
            ],
            [
                0.7236994467439906 - 0.150669495088244j,
                0.6066910616540805 + 0.3688901068039437j,
                -0.6464264434179551 - 0.193962534531747j,
                0.5495144699416846 + 0.4298114980905241j,
            ],
            [
                0.2684473029079276 + 0.06945040994327685j,
                -0.3751410719745281 + 0.04497945784785218j,
                0.2968207211063859 - 0.1508754857232733j,
                -0.2078596978975164 + 0.1729486519464096j,
            ],
        ]
    )
    return zfbf(H), vector, H