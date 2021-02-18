import numpy as np
import cvxpy


def brb_algorithm_cvx(
    H,
    D,
    Qsrt,
    q,
    boxes_lower_corners,
    boxer_upper_corners,
    wights,
    delta,
    epsilon,
    max_interations,
    max_func_evaluations,
    local_feasible,
):
    """Maximizes the weighted sum rate or weighted proportional fairness using
    the Branch-Reduce-and-Bound (BRB) algorithm in Algorithm 3. Both problems
    are non-convex and NP-hard in general, thus the computational complexity
    scales exponentially with the number of users, Kr. This implementation is
    not recommend for more than Kr=6 users.

    The references to theorems and equations refer to the following book:

    Emil Bjornson, Eduard Jorswieck, Optimal Resource Allocation in
    Coordinated Multi-Cell Systems, Foundations and Trends in Communications
    and Information Theory, vol. 9, no. 2-3, pp. 113-381, 2013.

    This is version 1.2. (Last edited: 2014-03-26)

    License: This code is licensed under the GPLv2 license. If you in any way
    use this code for research that results in publications, please cite our
    original article listed above.

    The implementation utilizes and requires cvxpy: https://www.cvxpy.org/


    INPUT:
    H                       = Kr x Kt*Nt matrix with row index for receiver and column
                            index transmit antennas
    D                       = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one
                            if j:th antenna can transmit to user k and zero otherwise
    Qsqrt                   = N x N x L matrix with matrix-square roots of the L
                            weighting matrices for the L power constraints
    q                       = Limits of the L power constraints
    boxesLowerCorners       = Kr x 1 vector with lower corner of an initial box that
                            covers the rate region
                            boxesUpperCorners = Kr x 1 vector with upper corner of an initial box that
                            covers the rate region
    weights                 = Kr x 1 vector with positive weights for each user
    delta                   = Accuracy of the line-search in FPO subproblems
                            (see functionFairnessProfile() for details
    epsilon                 = Accuracy of the final value of the utility
    maxIterations           = Maximal number of outer iterations of the algorithm
    maxFuncEvaluations      = Maximal number of convex feasibility subproblems to
                            be solved
    localFeasible           = (Optional) Kr x 1 vector with any feasible solution
    problemMode             = (Optional) Weighted sum rate is given by mode==1 (default)
                            Weighted proportional fairness is given by mode==2
    saveBoxes               = (Optional) Saves and return the set of boxes from each
                            iteration of the algorithm if saveBoxes==1

    OUTPUT:
    bestFeasible            = The best feasible solution found by the algorithm
    Woptimal                = Kt*Nt x Kr matrix with beamforming that achieves bestFeasible
    totalNbrOfEvaluations   = Vector with number of times that the convex
                            subproblem was solved at each iteration of the
                            algorithm
    bounds                  = Matrix where first/second column gives the global
                            lower/upper bound at each iteration of the algorithm
    boxes                   = Cell array where boxes{k}.lowerCorners and
                            boxes{k}.upperCorners contain the corners of the
                            boxes at the end of iteration k.
    """
    pass
