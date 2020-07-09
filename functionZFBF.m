function wZFBF = functionZFBF(H,D)
%Calculates the zero-forcing beamforming (ZFBF) vectors for a
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
%wZFBF = Kt*Nt x Kr matrix with normalized ZFBF




%Number of users
Kr = size(H,1);

%Total number of antennas
N = size(H,2);

%If D matrix is not provided, all antennas can transmit to everyone
if nargin<2
    D = repmat( eye(N), [1 1 Kr]);
end

%Pre-allocation of MRT beamforming
wZFBF = zeros(size(H'));

%Computation of ZFBF, based on Definition 3.4
for k = 1:Kr
    effectivechannel = (H*D(:,:,k))'; %Effective channels
    channelinversion = effectivechannel/(effectivechannel'*effectivechannel); %Compute zero-forcing based on channel inversion
    wZFBF(:,k) = channelinversion(:,k)/norm(channelinversion(:,k));  %Normalization of zero-forcing direction
end

%!test
%! H = ctranspose([
%!    0.013860 + 0.031335i,   1.073221 - 0.940552i,   0.920571 - 1.373000i,   0.442014 - 0.353275i;
%!    -0.067678 - 0.514558i,   0.785435 + 0.629878i,   0.230476 + 0.989237i,  -1.701096 + 1.125456i;
%!    0.868633 + 0.569620i,   0.191321 - 0.177566i,  -0.151462 + 0.232256i,   1.203675 + 0.364566i;
%!    2.037872 - 0.802488i,  -2.043176 - 0.129150i,   0.487697 + 0.379195i,   0.042107 - 0.400414i;
%! ]);
%! vector = [
%!    0.38796 + 0.09872i   0.34066 + 0.05635i   0.22043 - 0.57427i  -0.32785 - 0.05156i;
%!    0.34114 - 0.31628i   0.45264 - 0.17027i  -0.21103 + 0.10360i  -0.44957 + 0.35767i;
%!    0.72370 - 0.15067i   0.60669 + 0.36889i  -0.64643 - 0.19396i   0.54951 + 0.42981i;
%!    0.26845 + 0.06945i  -0.37514 + 0.04498i   0.29682 - 0.15088i  -0.20786 + 0.17295i;
%! ];
%! assert (functionZFBF(H), vector)