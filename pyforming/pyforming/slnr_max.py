import numpy as np
import cvxpy
import logging

logger = logging.getLogger("pyforming")


def slnr_max(h: np.ndarray, eta: np.ndarray = None, d: np.ndarray = None):
    """
    Calculates the Signal-to-leakage-and-noise ratio maximizing (SLNR-MAX)
    beamforming for a scenario where all or a subset of antennas transmit
    to each user. Note that SLNR-MAX is also known as regularized zero-forcing
    beamforming and transmit MMSE beamforming

    The references to definitions and equations refer to the following book:

    Emil Bjornson, Eduard Jorswieck, Optimal Resource Allocation in
    Coordinated Multi-Cell Systems, Foundations and Trends in Communications
    and Information Theory, vol. 9, no. 2-3, pp. 113-381, 2013.

    This is version 1.1. (Last edited: 2014-03-26)

    License: This code is licensed under the GPLv2 license. If you in any way
    use this code for research that results in publications, please cite our
    original article listed above.

    INPUT:
    H       = Kr x Kt*Nt matrix with row index for users and column index
            transmit antennas
    eta     = Kr x 1 vector with SNR^(-1) like parameter of this user
    D       = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if j:th
            transmit antenna can transmit to user k and zero otherwise

    OUTPUT:
    wSLNRMAX = Kt*Nt x Kr matrix with normalized SLNR-MAX beamforming
    """
    # number of users
    kr = h.shape[0]

    # total number of antennas
    n = h.shape[-1]

    # If eta vector is not provided, all values are set to unity
    if eta is None:
        eta = np.ones((kr, 1))

    # If d matrix is not provided, all antennas can transmit to everyone
    if d is None:
        d = np.tile(np.eye(n), [kr, 1, 1])

    # Pre-allocate an array for SLNR-MAX beamforming
    w_slnr_max = np.zeros(np.transpose(h.shape), dtype=np.complex128)

    # Compute SLNR-MAX, based on Definition 3.5 in optimal resource allocation
    for k in range(0, kr):
        effective_channel = np.conj(np.transpose(np.matmul(h, d[k, :, :])))
        # np.linalg.lstsq is the numpy solution to right division in MATLAB
        # Compute zero-forcing based on channel inversion
        # Normalization of zero-forcing direction
        projected_vector, _, _, _ = np.linalg.lstsq(
            (
                np.eye(n) / eta[k]
                + np.matmul(
                    effective_channel,
                    np.conj(
                        np.transpose(effective_channel),
                    ),
                )
            ),
            effective_channel[:, k],
            rcond=-1,
        )
        # Normalization of useful channel
        # use the 2-norm, like in MATLAB
        w_slnr_max[:, k] = projected_vector / np.linalg.norm(projected_vector, ord=2)

    return w_slnr_max


def test_slnr_result():
    vector = np.asanyarray(
        [
            [
                0.3476628662852445 - 0.01783588486260779j,
                0.4494379818391639 - 0.1763682075947649j,
                0.3899540484325554 - 0.6482741506826423j,
                -0.2285278261440254 - 0.1696504694549773j,
            ],
            [
                0.3463798865507545 - 0.2959436304501834j,
                0.428717593984793 - 0.1745054033561354j,
                -0.06451886696333453 + 0.21950816809492j,
                -0.6133871186652898 + 0.3584529417431375j,
            ],
            [
                0.6496666894842142 - 0.006050820533212617j,
                0.4161470856235192 + 0.3345735754410368j,
                -0.4900364211614296 - 0.03295049052757694j,
                0.5475093195161629 + 0.2651127301627587j,
            ],
            [
                0.4889848952633911 - 0.1002053229938145j,
                -0.5148073666334743 - 0.05004343754291576j,
                0.3587762455059538 - 0.07340383084744746j,
                -0.1283085684196549 + 0.1665748442036544j,
            ],
        ]
    )
    H = np.conj(
        np.transpose(
            (
                [
                    [
                        0.013860 + 0.031335j,
                        1.073221 - 0.940552j,
                        0.920571 - 1.373000j,
                        0.442014 - 0.353275j,
                    ],
                    [
                        -0.067678 - 0.514558j,
                        0.785435 + 0.629878j,
                        0.230476 + 0.989237j,
                        -1.701096 + 1.125456j,
                    ],
                    [
                        0.868633 + 0.569620j,
                        0.191321 - 0.177566j,
                        -0.151462 + 0.232256j,
                        1.203675 + 0.364566j,
                    ],
                    [
                        2.037872 - 0.802488j,
                        -2.043176 - 0.129150j,
                        0.487697 + 0.379195j,
                        0.042107 - 0.400414j,
                    ],
                ]
            )
        )
    )
    return slnr_max(H), vector, H