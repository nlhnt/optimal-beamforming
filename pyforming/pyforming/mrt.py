import numpy as np
import cvxpy
import logging

logger = logging.getLogger("pyforming")


def mrt(h: np.ndarray, d: np.ndarray = None):
    """Calculates the maximum ratio transmission (MRT) beamforming vectors for a
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
    H  = Kr x Kt*Nt matrix with row index for users and column index
        transmit antennas
    D  = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if j:th
        transmit antenna can transmit to user k and zero otherwise

    OUTPUT:
    wMRT = Kt*Nt x Kr matrix with normalized MRT beamforming
    """
    # number of users
    kr = h.shape[0]

    # total number of antennas
    n = h.shape[-1]

    # if d matrix is not provided, all antennas can transmit to everyone
    if d is None:
        d = np.tile(np.eye(n), [kr, 1, 1])

    # Pre-allocate an array for MRT beamforming
    wMRT = np.zeros(np.transpose(h.shape), dtype=np.complex128)

    # Compute MRT, based on Definition 3.2 in optimal resource allocation
    for k in range(0, kr):
        # Useful channel:
        #   1) take all elements from h in row k, H(k,:)
        #   2) multiply the result matrix with D(:,:,k)'
        channel_vector = np.conj(np.transpose(np.matmul(h[k, :], d[k, :, :])))
        # Normalization of useful channel
        # use the 2-norm, like in MATLAB
        wMRT[:, k] = channel_vector / np.linalg.norm(channel_vector, ord=2)

    return wMRT


def test_mrt():
    """Used to debug only, a copy is available in pytest module"""
    H = np.transpose(
        np.conj(
            np.asarray(
                [
                    [0.013860 + 0.031335j, 1.073221 - 0.940552j, 0.920571 - 1.373000j, 0.442014 - 0.353275j],
                    [-0.067678 - 0.514558j, 0.785435 + 0.629878j, 0.230476 + 0.989237j, -1.701096 + 1.125456j],
                    [0.868633 + 0.569620j, 0.191321 - 0.177566j, -0.151462 + 0.232256j, 1.203675 + 0.364566j],
                    [2.037872 - 0.802488j, -2.043176 - 0.129150j, 0.487697 + 0.379195j, 0.042107 - 0.400414j],
                ]
            )
        )
    )  # conjugate (hermitian) transpose in numpy

    vector = np.asarray(
        [
            [
                0.005590517116029192 + 0.01263916694305734j,
                0.3969593234537447 - 0.3478881661773917j,
                0.4479778380338826 - 0.6681435452784422j,
                0.1771687477836263 - 0.1416002420132859j,
            ],
            [
                -0.0272983418022095 - 0.2075501663917567j,
                0.2905140192158856 + 0.23297712655492j,
                0.112156629090746 + 0.4813928013842755j,
                -0.6818359784525727 + 0.4511070468482195j,
            ],
            [
                0.3503685176080653 + 0.2297597662072546j,
                0.07076515901430729 - 0.06567750652324883j,
                -0.07370601431534116 + 0.1130228312106263j,
                0.4824589096464283 + 0.146125918419972j,
            ],
            [
                0.8219883330646928 - 0.3236885208808105j,
                -0.7557229709975188 - 0.04776956155726651j,
                0.2373281883478954 + 0.1845278162067436j,
                0.01687739407105918 - 0.160494570251243j,
            ],
        ]
    )
    return mrt(H), vector, H