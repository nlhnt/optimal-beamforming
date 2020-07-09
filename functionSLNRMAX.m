function wSLNRMAX = functionSLNRMAX(H,eta,D)
%Calculates the Signal-to-leakage-and-noise ratio maximizing (SLNR-MAX)
%beamforming for a scenario where all or a subset of antennas transmit 
%to each user. Note that SLNR-MAX is also known as regularized zero-forcing
%beamforming and transmit MMSE beamforming
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
%H   = Kr x Kt*Nt matrix with row index for users and column index
%      transmit antennas
%eta = Kr x 1 vector with SNR^(-1) like parameter of this user 
%D   = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if j:th
%      transmit antenna can transmit to user k and zero otherwise
%
%OUTPUT:
%wSLNRMAX = Kt*Nt x Kr matrix with normalized SLNR-MAX beamforming



%Number of users
Kr = size(H,1);

%Total number of antennas
N = size(H,2);

%If eta vector is not provided, all values are set to unity
if nargin<2
    eta = ones(Kr,1);
end

%If D matrix is not provided, all antennas can transmit to everyone
if nargin<3
    D = repmat( eye(N), [1 1 Kr]);
end

%Pre-allocation of MRT beamforming
wSLNRMAX = zeros(size(H'));

%Computation of SLNR-MAX, based on Definition 3.5
for k = 1:Kr
    effectivechannel = (H*D(:,:,k))'; %Effective channels
    projectedchannel = (eye(N)/eta(k)+effectivechannel*effectivechannel')\effectivechannel(:,k); %Compute zero-forcing based on channel inversion
    wSLNRMAX(:,k) = projectedchannel/norm(projectedchannel);  %Normalization of zero-forcing direction
end

%!test
%! vector = [
%!    0.3476628662852445 - 0.01783588486260779i	0.4494379818391639 - 0.1763682075947649i	0.3899540484325554 - 0.6482741506826423i	-0.2285278261440254 - 0.1696504694549773i;
%!    0.3463798865507545 - 0.2959436304501834i	0.428717593984793 - 0.1745054033561354i	-0.06451886696333453 + 0.21950816809492i	-0.6133871186652898 + 0.3584529417431375i;
%!    0.6496666894842142 - 0.006050820533212617i	0.4161470856235192 + 0.3345735754410368i	-0.4900364211614296 - 0.03295049052757694i	0.5475093195161629 + 0.2651127301627587i;
%!    0.4889848952633911 - 0.1002053229938145i	-0.5148073666334743 - 0.05004343754291576i	0.3587762455059538 - 0.07340383084744746i	-0.1283085684196549 + 0.1665748442036544i;
%! ]
%! H = ctranspose([
%!    0.013860 + 0.031335i,   1.073221 - 0.940552i,   0.920571 - 1.373000i,   0.442014 - 0.353275i;
%!    -0.067678 - 0.514558i,   0.785435 + 0.629878i,   0.230476 + 0.989237i,  -1.701096 + 1.125456i;
%!    0.868633 + 0.569620i,   0.191321 - 0.177566i,  -0.151462 + 0.232256i,   1.203675 + 0.364566i;
%!    2.037872 - 0.802488i,  -2.043176 - 0.129150i,   0.487697 + 0.379195i,   0.042107 - 0.400414i;
%! ]);
%! assert (functionSLNRMAX(H), vector)