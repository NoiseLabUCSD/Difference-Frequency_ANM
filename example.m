clear; clc;
close all;

dbstop if error;
% rng(7)

addpath([cd,'/_common'])

%% Environment parameters
c       = 1500;

% Linear array configuration
d = 1.50;                  % intersensor spacing [m]
% f = 500; % Hz -> 0.5 lambda = 0.5 * 1500 / 500 = 1.50 [m]

Nsensor = 20;               % number of sensors

%% Frequencies
dfreq   = 500;                  % difference frequency [Hz]

ftmp    = [4900,4950,5000];
f       = [ftmp,ftmp+dfreq];    % frequency [Hz]
Nfreq   = numel(f);

%%
lambda = c./f;   % wavelength

% true DOAs
doa     = [-49 45];

% number of snapshots
Nsnapshot   = 25;

% range of angle space
thetalim = [-90 90];

% Angular search grid for gridded methods
theta_separation    = .1;
theta               = (thetalim(1):theta_separation:thetalim(2))';
Ntheta              = length(theta);

% Design/steering matrix (Sensing matrix)
sin_theta = sind(theta);

%% Simulation parameters
NmonteCarlo = 1;
SNR         = 20;
anglesTrue  = doa.';
fprintf(['True DOAs                :',...
    repmat([' %.4f '],1,numel(anglesTrue)),'\n'],anglesTrue.')

anglesTracks = repmat(anglesTrue,[1,Nsnapshot]);
sinAnglesTracks = sind(anglesTracks);

Ndoas     = numel(anglesTrue);

LSnapshot = Nsnapshot * NmonteCarlo; % Number of array data vector observations "Large"
LrPhase   = (exp(1i*2*pi*rand(Ndoas,LSnapshot,Nfreq)));
Lnwhite   = complex(randn(Nsensor,LSnapshot,Nfreq),randn(Nsensor,LSnapshot,Nfreq))/sqrt(2);

n_monteCarlo = 1;
q     = (0:1:(Nsensor-1))';     % sensor numbering
xq    = (q-(Nsensor-1)/2)*d;   % sensor locations

% Sensing matrix for DF
sensingMatrixDF = zeros(Nsensor,Ntheta,Nfreq/2);
for nfreq = 1:Nfreq/2
    sensingMatrixDF(:,:,nfreq) = exp(-1i*2*pi/ (c/dfreq) *xq*sin_theta.')/sqrt(Nsensor);
end

receivedSignalMultiFreq = zeros(Nsensor,Nsnapshot,Nfreq);
rPhase  = LrPhase(:,(n_monteCarlo-1)*Nsnapshot+(1:Nsnapshot),:);
rnwhite = Lnwhite(:,(n_monteCarlo-1)*Nsnapshot+(1:Nsnapshot),:);

for nfreq = 1:Nfreq
    receivedSignal = zeros(Nsensor,Nsnapshot);
    e = zeros(Nsensor,Nsnapshot);
    for snapshot = 1:Nsnapshot
        source_amp(:,snapshot) = 10*ones(size(anglesTrue));
        Xsource = source_amp(:,snapshot).*rPhase(:,snapshot,nfreq);    % random phase

        % Represenation matrix (steering matrix)
        transmitMatrix = exp( -1i*2*pi/lambda(nfreq)*xq*sinAnglesTracks(:,snapshot).' );

        % Received signal without noise
        receivedSignal(:,snapshot) = sum(transmitMatrix*diag(Xsource),2);

        % add noise to the signals
        nwhite        = complex(randn(Nsensor,1),randn(Nsensor,1))/sqrt(2);
        rnl           = 10^(-SNR/20)*norm(receivedSignal(:,snapshot))/norm(nwhite);
        e(:,snapshot) = nwhite * rnl;	% error vector
        %           20*log10(norm(receivedSignal(:,snapshot))/norm(e(:,snapshot)))

        receivedSignal(:,snapshot) = receivedSignal(:,snapshot) + e(:,snapshot);
    end
    receivedSignalMultiFreq(:,:,nfreq) = receivedSignal;
end

%% Hadamard-product
HPset = zeros(Nsensor,Nsnapshot,Nfreq/2);
for nfreq = 1:Nfreq/2
    receivedSignalTmp(:,:)  = receivedSignalMultiFreq(:,:,nfreq);
    receivedSignalTmp2(:,:) = receivedSignalMultiFreq(:,:,nfreq+Nfreq/2);
    HPTmp = receivedSignalTmp2 .* conj(receivedSignalTmp);

    HPset(:,:,nfreq) = HPTmp;
end

%% DF-CBF
for fIndn = 1:Nfreq/2
    for iGrid=1:size(sensingMatrixDF,2)
        PbartlettTmp(iGrid) = (sensingMatrixDF(:,iGrid,fIndn))' ...
            * HPset(:,:,fIndn) * HPset(:,:,fIndn)' ...
            * (sensingMatrixDF(:,iGrid,fIndn));
    end
    PbartlettDF(:,fIndn) = abs(PbartlettTmp);
end
PbartlettDF = mean(PbartlettDF,2);

figure, set(gcf,'position',[100,45,450,800]); clf;
ax2 = subplot(412);
plot(theta,10*log10( PbartlettDF/max(PbartlettDF) ),'k','Linewidth',2)
%     hold on; plot([-90 90],[.500 .500], 'r-.', 'linewidth',2); hold off;
axis([-90 90 -27 2])
set(gca,'fontsize',18,'TickLabelInterpreter','latex','XTick',-80:40:80)
set(gca,'Position',[.1833,.5445,.7750,.2075])

xlabel('DOA~[$^\circ$]','interpreter','latex');
ylabel('$P$ [dB re max]','interpreter','latex');

%% DF-MUSIC
Neig = Ndoas^2;
for fIndn = 1:Nfreq/2
    HPsetTmp = reshape(HPset(:,:,fIndn),Nsensor,Nsnapshot);
    RzzTmp = HPsetTmp*HPsetTmp'/Nsnapshot;
    [Rv,Rd] = eig(RzzTmp);
    Rvn = Rv(:, 1:end-Neig);
    PmusicTmp = zeros(size(sensingMatrixDF,2),1);
    for iGrid=1:size(sensingMatrixDF,2)
        PmusicTmp(iGrid) = 1./( (sensingMatrixDF(:,iGrid))' ...
            * (Rvn * Rvn') ...
            * (sensingMatrixDF(:,iGrid)) );
    end
    PmusicDF(:,fIndn) = abs(PmusicTmp);
end
PmusicDF = mean(PmusicDF,2);

hold on;
pColor = lines; pColor = pColor(5,:);
plot(theta,10*log10( PmusicDF/max(PmusicDF) ),'--','Linewidth',2,'Color',pColor)
hold off;

%% Multi-snapshot-DF-SA ANM
for fIndn = 1:Nfreq/2
    HPsetTmp = reshape(HPset(:,:,fIndn),Nsensor,Nsnapshot);

    tau = .1;
    cvx_solver sdpt3
    cvx_begin sdp quiet
        variable y_AT(Nsensor,Nsnapshot)   complex
        variable W_AT(Nsnapshot,Nsnapshot) complex hermitian
        variable u_AT(Nsensor,Nsensor)     hermitian toeplitz
            minimize (1/2*pow_pos(norm(HPsetTmp - y_AT,'fro'),2) + tau*(trace(u_AT)/Nsensor + trace(W_AT))/2)
            subject to
                [u_AT,y_AT;y_AT',W_AT] >= 0;
    cvx_end
    TuT3(:,:,fIndn) = y_AT;
end

%% DOA extraction
Nest = Ndoas^2;
[ root_locs, c1 ] = nullSteerIncAvg( q, Nest, TuT3 );
DoA_est_deg       = asin( -root_locs*(c/dfreq)/d ) /pi *180;

% Source power
sensingMatrixDFtemp = zeros(Nsensor,Nest);
for nfreq = 1:Nfreq/2
    sensingMatrixDFtemp = exp(-1i*2*pi/ (c/dfreq) *xq*sind(DoA_est_deg).')/sqrt(Nsensor);

    HPsetTmp = reshape(HPset(:,:,nfreq),Nsensor,Nsnapshot);
    P_ANMTmp = pinv(sensingMatrixDFtemp) * HPsetTmp; P_ANMTmp = power(abs(P_ANMTmp),2);
    P_ANM(:,nfreq) = mean(P_ANMTmp,2);
end
P_ANM = mean(P_ANM,2);

[~,sortPOW] = sort(P_ANM,'descend');

P_ANM       = P_ANM(sortPOW(1:Ndoas));
DoA_est_deg = DoA_est_deg(sortPOW(1:Ndoas));
% DoA_error   = errorDOAcutoff(DoA_est_deg,anglesTrue,10);
% disp(['RMSE multi-snap      ANM       : ',num2str(sqrt(mean(power(DoA_error,2))))])

hold on;
pColor = lines;
stem(DoA_est_deg,10*log10( P_ANM/max(P_ANM) ),'.','basevalue',-200, ...
    'Markersize',8,'Linewidth',6,'Color',[1,0,0]);
hold off;
hold on; stem(anglesTrue,100*ones(size(anglesTrue)),'k-.','basevalue',-200,'Marker','none','Linewidth',1); hold off;

legend('CBF','MUSIC','ANM','Interpreter','latex','fontsize',18,'Position',[0.4662,0.6674,0.1939,0.0692])

%% Multi-snapshot-DF-SA Null spectrum
utot = TuT3;

[U,~,~] = svd(utot(:,:,1));    En = U(:,Ndoas^2+1:end);    G1 = En*En';
[U,~,~] = svd(utot(:,:,2));    En = U(:,Ndoas^2+1:end);    G2 = En*En';
[U,~,~] = svd(utot(:,:,3));    En = U(:,Ndoas^2+1:end);    G3 = En*En';
%
fun11        = @(theta) abs(null_spec_polynomial( G1, q, exp(-1i*pi*sind(theta))) );
fun12        = @(theta) abs(null_spec_polynomial( G2, q, exp(-1i*pi*sind(theta))) );
fun13        = @(theta) abs(null_spec_polynomial( G3, q, exp(-1i*pi*sind(theta))) );
max_theta   = 90;
thetas      = -max_theta:.1:max_theta;
Null_spec   = zeros(length(thetas),1);
for i = 1:length(thetas)
    Null_spec(i)   = ( fun11(thetas(i)) + fun12(thetas(i)) + fun13(thetas(i)) ) / 3;
end

ax1 = subplot(411);
plot(thetas,Null_spec,'LineWidth',4,'Color',[1,.5,0])
set(gca,'fontsize',18,'TickLabelInterpreter','latex','XTick',-80:40:80)
axis([-90 90 0 22])
hold on; stem(anglesTrue,100*ones(size(anglesTrue)),'k-.','basevalue',-200,'Marker','none','Linewidth',1); hold off;

%     xlabel('DOA~[$^\circ$]','interpreter','latex');
ylabel('Magnitude','interpreter','latex');
set(gca,'XTickLabel',' ')
set(gca,'Position',[.1833,.7675,.7750,.1575])

title('Multi-snapshot-DF-SA')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% Multi-snapshot-DF
figure, set(gcf,'position',[600,45,450,800]);
%% joint DFs
HPsetTmp = [];
for fIndn = 1:Nfreq/2
    HPsetTmpTmp = reshape(HPset(:,:,fIndn),Nsensor,Nsnapshot);
    HPsetTmp = [HPsetTmp,HPsetTmpTmp];
end

%% DF-CBF
clear PbartlettDF
for iGrid=1:size(sensingMatrixDF,2)
    PbartlettTmp(iGrid) = (sensingMatrixDF(:,iGrid,fIndn))' ...
        * (HPsetTmp * HPsetTmp' / size(HPsetTmp,2)) ...
        * (sensingMatrixDF(:,iGrid,fIndn));
end
PbartlettDF(:,fIndn) = abs(PbartlettTmp);

ax2 = subplot(412);
plot(theta,10*log10( PbartlettDF/max(PbartlettDF) ),'k','Linewidth',2)
axis([-90 90 -27 2])
set(gca,'fontsize',18,'TickLabelInterpreter','latex','XTick',-80:40:80)
set(gca,'Position',[.1833,.5445,.7750,.2075])

xlabel('DOA~[$^\circ$]','interpreter','latex');
ylabel('$P$ [dB re max]','interpreter','latex');
% set(gca,'YTickLabel',' ')

%% DF-MUSIC
clear PmusicDF
Neig = Ndoas^2;
RzzTmp = HPsetTmp*HPsetTmp'/size(HPsetTmp,2);
[Rv,Rd] = eig(RzzTmp);
Rvn = Rv(:, 1:end-Neig);
PmusicTmp = zeros(size(sensingMatrixDF,2),1);
for iGrid=1:size(sensingMatrixDF,2)
    PmusicTmp(iGrid) = 1./( (sensingMatrixDF(:,iGrid))' ...
        * (Rvn * Rvn') ...
        * (sensingMatrixDF(:,iGrid)) );
end
PmusicDF(:,fIndn) = abs(PmusicTmp);

hold on;
pColor = lines; pColor = pColor(5,:);
plot(theta,10*log10( PmusicDF/max(PmusicDF) ),'--','Linewidth',2,'Color',pColor)
hold off;

%% Multi-snapshot-DF ANM
tau = .1;
cvx_solver sdpt3
cvx_begin sdp quiet
variable y_AT(Nsensor          ,Nsnapshot*Nfreq/2)  complex
variable W_AT(Nsnapshot*Nfreq/2,Nsnapshot*Nfreq/2)  complex hermitian
variable u_AT(Nsensor,Nsensor)                      hermitian toeplitz
minimize (1/2*pow_pos(norm(HPsetTmp - y_AT,'fro'),2) + tau*(trace(u_AT)/Nsensor + trace(W_AT))/2)
subject to
[u_AT,y_AT;y_AT',W_AT] >= 0;
cvx_end
TuT3 = y_AT;

%% DOA extraction
Nest = Ndoas^2;
% [ root_locs, c1 ] = nullSteerIncAvg( q, Nest, TuT3 );
[ root_locs, c1 ] = nullSpecUnitAvgTu( q, Nest, u_AT, 'Toep(u)' );
DoA_est_deg       = asin( -root_locs*(c/dfreq)/d ) /pi *180;

% Source power
sensingMatrixDFtemp = exp(-1i*2*pi/ (c/dfreq) *xq*sind(DoA_est_deg).')/sqrt(Nsensor);
P_ANM = pinv(sensingMatrixDFtemp) * HPsetTmp; P_ANM = power(abs(P_ANM),2);
P_ANM = mean(P_ANM,2);

[~,sortPOW] = sort(P_ANM,'descend');

P_ANM       = P_ANM(sortPOW(1:Ndoas));
DoA_est_deg = DoA_est_deg(sortPOW(1:Ndoas));
% DoA_error   = errorDOAcutoff(DoA_est_deg,anglesTrue,10);
% disp(['RMSE multi-snap      ANM       : ',num2str(sqrt(mean(power(DoA_error,2))))])

hold on;
pColor = lines;
stem(DoA_est_deg,10*log10( P_ANM/max(P_ANM) ),'.','basevalue',-200, ...
    'Markersize',8,'Linewidth',6,'Color',[1,0,0]);
hold off;
hold on; stem(anglesTrue,100*ones(size(anglesTrue)),'k-.','basevalue',-200,'Marker','none','Linewidth',1); hold off;

legend('CBF','MUSIC','ANM','Interpreter','latex','fontsize',18,'Position',[0.4662,0.6674,0.1939,0.0692])

%% Multi-snapshot-DF Null spectrum

[U,~,~] = svd(u_AT);    En = U(:,Ndoas+1:end);    G1 = En*En';
%
fun11        = @(theta) abs(null_spec_polynomial( G1, q, exp(-1i*pi*sind(theta))) );
max_theta   = 90;
thetas      = -max_theta:.1:max_theta;
Null_spec   = zeros(length(thetas),1);
for i = 1:length(thetas)
    Null_spec(i)   = fun11(thetas(i));
end

ax1 = subplot(411); pColor = lines; pColor = pColor(1,:);
plot(thetas,Null_spec,'LineWidth',4,'Color',pColor)
set(gca,'fontsize',18,'TickLabelInterpreter','latex','XTick',-80:40:80)
axis([-90 90 0 22])
hold on; stem(anglesTrue,100*ones(size(anglesTrue)),'k-.','basevalue',-200,'Marker','none','Linewidth',1); hold off;

%     xlabel('DOA~[$^\circ$]','interpreter','latex');
ylabel('Magnitude','interpreter','latex');
set(gca,'XTickLabel',' ')
set(gca,'Position',[.1833,.7675,.7750,.1575])

title('Multi-snapshot-DF')

%%
rmpath([cd,'/_common'])