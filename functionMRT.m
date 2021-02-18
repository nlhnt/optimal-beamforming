function wMRT = functionMRT(H,D)
%Calculates the maximum ratio transmission (MRT) beamforming vectors for a
%scenario where all or a subset of antennas transmit to each user.
%
%The references to definitions and equations refer to the following book:
%
%Emil Bj�rnson, Eduard Jorswieck, �Optimal Resource Allocation in
%Coordinated Multi-Cell Systems,� Foundations and Trends in Communications
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
%! H = ctranspose([
%!    0.013860 + 0.031335i,   1.073221 - 0.940552i,   0.920571 - 1.373000i,   0.442014 - 0.353275i;
%!    -0.067678 - 0.514558i,   0.785435 + 0.629878i,   0.230476 + 0.989237i,  -1.701096 + 1.125456i;
%!    0.868633 + 0.569620i,   0.191321 - 0.177566i,  -0.151462 + 0.232256i,   1.203675 + 0.364566i;
%!    2.037872 - 0.802488i,  -2.043176 - 0.129150i,   0.487697 + 0.379195i,   0.042107 - 0.400414i;
%! ]);
%! vector = [
%!    0.005590517116029192 + 0.01263916694305734i	0.3969593234537447 - 0.3478881661773917i	0.4479778380338826 - 0.6681435452784422i	0.1771687477836263 - 0.1416002420132859i;
%!    -0.0272983418022095 - 0.2075501663917567i	0.2905140192158856 + 0.23297712655492i	0.112156629090746 + 0.4813928013842755i	-0.6818359784525727 + 0.4511070468482195i;
%!    0.3503685176080653 + 0.2297597662072546i	0.07076515901430729 - 0.06567750652324883i	-0.07370601431534116 + 0.1130228312106263i	0.4824589096464283 + 0.146125918419972i;
%!    0.8219883330646928 - 0.3236885208808105i	-0.7557229709975188 - 0.04776956155726651i	0.2373281883478954 + 0.1845278162067436i	0.01687739407105918 - 0.160494570251243i;
%! ];
%! assert (functionMRT(H), vector, eps)
