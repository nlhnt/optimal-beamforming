function wZFBF = functionZFBF(H,D)
%Calculates the zero-forcing beamforming (ZFBF) vectors for a
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
    % b/A
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
%!   0.3879614290224073 + 0.09872028990717077i	0.3406564576226037 + 0.05635336231117433i	0.2204253968379605 - 0.5742745067676878i	-0.3278458767505699 - 0.05156352160548806i;
%!   0.3411405788153983 - 0.3162812665800751i	0.4526354346504607 - 0.1702665822637207i	-0.2110303708731221 + 0.1036004519893354i	-0.4495672529458771 + 0.3576682857477722i;
%!   0.7236994467439906 - 0.150669495088244i	0.6066910616540805 + 0.3688901068039437i	-0.6464264434179551 - 0.193962534531747i	0.5495144699416846 + 0.4298114980905241i;
%!   0.2684473029079276 + 0.06945040994327685i	-0.3751410719745281 + 0.04497945784785218i	0.2968207211063859 - 0.1508754857232733i	-0.2078596978975164 + 0.1729486519464096i;
%! ];
%! assert (functionZFBF(H), vector, eps)