function wMRT = functionMRT(H,D)
%Calculates the maximum ratio transmission (MRT) beamforming vectors for a
%scenario where all or a subset of antennas transmit to each user.
%
%The references to definitions and equations refer to the following book:
%
%Emil Björnson, Eduard Jorswieck, “Optimal Resource Allocation in
%Coordinated Multi-Cell Systems,” Foundations and Trends in Communications
%and Information Theory, vol. 9, no. 2-3, pp. 113-381, 2013.
%
%This is version 1.1. (Last edited: 2014-03-26)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%
%INPUT:
%H  = Kr x Kt*Nt matrix with row index for users and column index
%     transmit antennas
%D  = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if j:th
%     transmit antenna can transmit to user k and zero otherwise
%
%OUTPUT:
%wMRT = Kt*Nt x Kr matrix with normalized MRT beamforming



%Number of users
Kr = size(H,1);

%Total number of antennas
N = size(H,2);

%If D matrix is not provided, all antennas can transmit to everyone
if nargin<2
    D = repmat( eye(N), [1 1 Kr]);
end

%Pre-allocation of MRT beamforming
wMRT = zeros(size(H'));

%Computation of MRT, based on Definition 3.2
for k = 1:Kr
    channelvector = (H(k,:)*D(:,:,k))'; %Useful channel
    wMRT(:,k) = channelvector/norm(channelvector); %Normalization of useful channel
end

%!test
%H = [
%    0.013860 + 0.031335i,   1.073221 - 0.940552i,   0.920571 - 1.373000i,   0.442014 - 0.353275i;
%    -0.067678 - 0.514558i,   0.785435 + 0.629878i,   0.230476 + 0.989237i,  -1.701096 + 1.125456i;
%    0.868633 + 0.569620i,   0.191321 - 0.177566i,  -0.151462 + 0.232256i,   1.203675 + 0.364566i;
%    2.037872 - 0.802488i,  -2.043176 - 0.129150i,   0.487697 + 0.379195i,   0.042107 - 0.400414i;
%]'
%vector = [
%   0.005590 - 0.012639i,  -0.027298 + 0.207550i,   0.350369 - 0.229760i,   0.821988 + 0.323688i;
%   0.396959 + 0.347888i,   0.290514 - 0.232977i,   0.070765 + 0.065678i,  -0.755723 + 0.047770i;
%   0.447978 + 0.668144i,   0.112156 - 0.481393i,  -0.073706 - 0.113023i,   0.237328 - 0.184528i;
%   0.177169 + 0.141600i,  -0.681836 - 0.451107i,   0.482459 - 0.146126i,   0.016877 + 0.160495i;
%]'
%assert (functionMRT(H), vector)
