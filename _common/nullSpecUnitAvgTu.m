function [ root_locs, c1 ] = nullSpecUnitAvgTu( r, K, u, option )
% GENERALIZED VANDERMONDE DECOMPOSITION: FINDS DOAS FROM COVARIANCE MATRIX FORMED FROM
% UNEVENLY SAMPLED SUM OF SIGNALS IN NOISE
% INPUTS:
% r             SENSOR LOCATIONS
% K             NUMBER OF DOAS PRESENT
% u             Toeplitz matrix elements [# SENSORS, # INCOHERENT DATA]
% OUTPUT:
% root_locs     DOAS GIVEN BETWEEN [-.5, .5)
% c1            EACH DOA CONTRIBUTIONS

% Version 1.0: (12/31/2019)
% wagner_decomp.m written by M. Wagner
% Version 2.0: (01/10/2023)
% written by Y. Park
% Null spectrum on unit circle for each Toeplitz matrix (measurement) separately
% and averaged across multiple measurements

%% # sensors / # coherent processing data   / 
%%           / # incoherent processing data / Array configuration
M     = size(u,1);
if strcmpi(option,'Toep(u)') % Number of averaging (# of Toeplitz matrices (measurements))
    NAvg  = size(u,3);
else
    NAvg  = size(u,2);
end
if size(r,2) ~= M
    r = real(r(:)).';
end

%% Find noise subspace for each incoherent processing data
G     = zeros(M,M,NAvg);
for nAvg = 1:NAvg
%     [U,~,~]     = svd( u(:,:,nAvg)*u(:,:,nAvg)'/L );
    if strcmpi(option,'Toep(u)')
        [U,~,~] = svd( u(:,:,nAvg) );
    else
        [U,~,~] = svd( toeplitz(u(:,nAvg)) );
    end
    Un          = U(:,K+1:end);
    G(:,:,nAvg) = Un*Un';
end

%% Evaluate MUSIC spectrum
samples                 = 100*(M);%15*(max(r)-min(r));%10*(M);
spacing                 = 180/samples;
f                       = -90:spacing:90;
MUSIC_spectrum          = zeros(length(f),1);
for ii = 1:NAvg
    MUSIC_spectrumTmp        = zeros(length(f),1);
    for i = 1:length(f)
        a                    = exp(1i*pi*sind(f(i)).*r');    %steering vector
        Gsingle              = G(:,:,ii);
        MUSIC_spectrumTmp(i) = -abs((a'*Gsingle*a));               %negative null spectrum
    end
    MUSIC_spectrum           = MUSIC_spectrum + MUSIC_spectrumTmp;
end
% Averaged over measurements
MUSIC_spectrum = MUSIC_spectrum / NAvg;

%% find peaks
[pks,inds]              = findpeaks(MUSIC_spectrum,(1:length(MUSIC_spectrum))); %find the peaks
[~,id]                  = sort(pks,'descend');

%% refine root estimates to high precision
root_locs               = f(inds(id(1:min([length(id),K]))));
root_locs_refined       = zeros(K,1);

options                 = optimset('Display','none','TolX',1e-25);
for i = 1:length(root_locs)
    root_locs_refined(i)= fminbnd(@(f)collectNull(G,f,r),root_locs(i)-(abs(f(1)-f(2))/2),root_locs(i)+(abs(f(1)-f(2))/2),options);      %find local minima about generalized null spectrum
end      
rl_rad                  = root_locs_refined/180*pi;
root_locs               = sort(sin(rl_rad)/2);

%% Reconstruct
W_est       = exp(1i*2*pi*root_locs(:).'.*r(:)); %regenerate irregular Vandermonde Matrix
W_inv       = pinv(W_est);
c1          = zeros(K,1);
for ii = 1:NAvg
    Gsingle = G(:,:,ii);
    c1      = c1 + real(diag(W_inv*Gsingle*W_inv'));
end
c1 = c1 / NAvg;
end

function fun = collectNull(G,f,r)
% Collect multiple null spectra for each incoherent processing data
fun = zeros(length(f),1);
for ii = 1:size(G,3)
    Gsingle = G(:,:,ii);
    fun     = fun + abs(diag((exp(1i*pi*sind(f).*r')')*Gsingle*exp(1i*pi*sind(f).*r')));
end
fun = fun / size(G,3);
end