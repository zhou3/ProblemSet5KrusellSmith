close all
beta = .99; %discount factor 
alpha = 0.36;
delta = 0.025;
% original method to compute the value function:
% get the state variable Z 
[PI, PI_star] = transmat;
%PI is 4*4 matrix, we dine[G1=1 G0=2 B1=3 B0=4]
%PI_A = PI_star;
PI_zstationary=[PI_star(1)+PI_star(2), 1-(PI_star(1)+PI_star(2))
    PI_star(1)+PI_star(2), 1-(PI_star(1)+PI_star(2))];
%PI_z=PI_zstationary^(1/1000);
%stationary distribution
% capital VECTOR
k_lo =0; % lower bound of grid points
k_hi =80; % upper bound of grid points
% num_k =101;
num_k=700;
kk = linspace(k_lo, k_hi, num_k); % asset (row) vector
% aggregate labor in each state
agg_L = [0.96,0.9];
% grid for aggregate capital
K_min = 30;
K_max = 40;
num_K = 11;
K = linspace(K_min, K_max, num_K);

% exogenous states:
z_grid = [1, 0]; %state of employment status
A_grid = [1.01, 0.99];%state of Aggregate TFP
Az_grid = [A_grid(1), z_grid(1)
           A_grid(2), z_grid(1)
           A_grid(1), z_grid(2)
           A_grid(2), z_grid(2)];
zz_grid = [ z_grid(1)
            z_grid(1)
            z_grid(2)
            z_grid(2)];


% initial guess for coefficients
coeff_tol=1;
coeffs = [0 1
          0 1];
while(coeff_tol>.1)



%factor prices for each of the states
r=A_grid'*alpha.*(agg_L'.^(1-alpha))*(K.^(alpha-1))+(1-delta);
r=repmat(r,[2 1]);
%r=repmat(r,[1 1 num_k num_k]);
r=permute(r,[3 4 2 1]);
r=repmat(r,[num_k num_k]);
% r=alpha*(N^(1-alpha))*(K^(alpha-1))+(1-delta);
w=A_grid'*(1-alpha).*(agg_L'.^(-alpha))*(K.^alpha);
w=repmat(w,[2 1]);
%r=repmat(r,[1 1 num_k num_k]);
w=permute(w,[3 4 2 1]);
w=repmat(w,[num_k num_k]);

z(:,:,:,1)=ones(num_k,num_k,num_K);
z(:,:,:,2)=zeros(num_k,num_k,num_K);
z(:,:,:,3)=ones(num_k,num_k,num_K);
z(:,:,:,4)=zeros(num_k,num_k,num_K);

k=repmat(kk',[1 num_k num_K 4]);
k_pri=repmat(kk,[num_k 1 num_K 4]);

cons=z.*w+r.*k+(1-delta)*k-k_pri;
%get the consumption matrix

% ret_fn=(1/(1 - sigma))*(cons .^ (1-sigma));
ret_fn = log(cons);
ret_fn(cons<=0) = -Inf;

% for each aggregate state figure out approximated K_prime according to the
% G~K function using current parameter guesses
K_prime = [coeffs(1,1)+coeffs(1,2)*K',coeffs(2,1)+coeffs(2,2)*K',...
    coeffs(1,1)+coeffs(1,2)*K',coeffs(2,1)+coeffs(2,2)*K'];

%K_prime=permute(K_prime,[3 4 1 2]);
%K_prime=repmat(K_prime,[num_k num_k]);
%%%%jintian gaodao zhe 

v_guess = zeros(num_K,num_k,4);
%v_guessmat=repmat(v_guess,[1 num_k 1 1]);
v_mat=zeros(num_k,num_k,num_K,4);
v_tol = 1;
% initialize value_fun:(guess)
%v_tol = 1;
%v_guess=zeros(num_a,5);
tic 
while v_tol>.0001;
% vfn=zeros(500,5);
% vfn_mat=zeros(500,500,5);
  V_exp = zeros(num_K, num_k, 4);
    for iAz = 1:4
        for iK = 1:num_K
            Exp_vmat=permute(interp1(K',v_guess,K_prime(iK,iAz)),[2 3 1])*PI(iAz,:)';
            Exp_vvv=repmat(Exp_vmat',[num_k 1 1]);
               v_mat(:,:,iK,iAz)=ret_fn(:,:,iK,iAz)+beta*Exp_vvv;
               [vfn, pol_indx] = max(v_mat, [], 2);
               vfn=permute(vfn,[3 1 4 2]);
               v_tol=abs(max(v_guess(:)-vfn(:)));
               v_guess=vfn;
        end 
    end 
end 


pol_indx = permute(pol_indx, [3 1 4 2]);
%pol_fn = k(pol_indx);
% step 3 simulate the economy, we pick initial condition that is G1
% draw random numbers for simulation
getA1

% define PI_z matrix
%From B to B
PI_zBB=[PI(2,2)/[PI(2,2)+PI(2,4)],PI(2,4)/[PI(2,2)+PI(2,4)]
        PI(4,2)/[PI(4,2)+PI(4,4)],PI(4,4)/[PI(4,2)+PI(4,4)]];
%From B to A
PI_zBA=[PI(2,1)/[PI(2,1)+PI(2,3)],PI(2,3)/[PI(2,1)+PI(2,3)]
        PI(4,1)/[PI(4,1)+PI(4,3)],PI(4,3)/[PI(4,1)+PI(4,3)]];
%From A to A
PI_zAA=[PI(1,1)/[PI(1,1)+PI(1,3)],PI(1,3)/[PI(1,1)+PI(1,3)]
        PI(3,1)/[PI(3,1)+PI(3,3)],PI(3,3)/[PI(3,1)+PI(3,3)]];
%From A to B
PI_zAB=[PI(1,2)/[PI(1,2)+PI(1,4)],PI(1,4)/[PI(1,2)+PI(1,4)]
        PI(3,2)/[PI(3,2)+PI(3,4)],PI(3,4)/[PI(3,2)+PI(3,4)]];
 
%we get A1(is state btw Good and Bad){
%we get AA is state btw G1 G0 B1 B0

%define PI_A matrix
AtoA=((PI(1,1)+PI(1,3))*.48+(PI(3,1)+PI(3,3))*.02)/(.5);
BtoA=((PI(2,1)+PI(2,3))*.45+(PI(4,1)+PI(4,3))*.05)/(.5);
PI_A=[AtoA,1-AtoA;BtoA,1-BtoA];

%AAA=zeros(5500,1);
%AAA(1)=1;
%for iii=1:5500
 %   if A(iii)==1
  %  AAA(iii+1)=binornd(1,0.875);
   % else
    %  AAA(iii+1)=binornd(1,0.125);
   % end 
%end 
%A1A=AAA(501:5500,1);
% can use that loop to generate A, but I already do it in another way as I
% said in the class
      
%set up the initial distribution
%Mu = zeros(2,num_k);
Mu_guess = ones(2, num_k); 
 Mu = Mu_guess / sum(Mu_guess(:)); 
%initial guess of Mu,same mass in all states
% normalize total mass to 1
% ITERATE OVER DISTRIBUTIONS
%mu_tol = 1;
%from the sequence of A1, we can know every PI_z we use:
PI_z=zeros(2,2,4999);
for tt=1:4999
    if A1(tt)==1 && A1(tt+1)==2 %G to B
    PI_z(:,:,tt)=PI_zAB;

    elseif A1(tt)==1 && A1(tt+1)==1 %G to G
    PI_z(:,:,tt)=PI_zAA;
    
    elseif A1(tt)==2 && A1(tt+1)==2 %B to B
    PI_z(:,:,tt)=PI_zBB;
    else 
     PI_z(:,:,tt)= PI_zBA; 
    
    end 
end 

T_sim=5000;
K_series = zeros(T_sim, 1);
for t = 1:T_sim-1
    K_agg = sum(Mu,1) * kk';
    %compute the K_agg in this state 
    K_series(t) = K_agg;
    % we store the K sequences
    
    
    pol_ind_interp = interp1(K',pol_indx,K_agg,'linear','extrap');
    pol_ind_interp=permute(pol_ind_interp,[3 2 1]);
    pol_ind_interp=pol_ind_interp([A1(t) A1(t)+2],:);
    pol_ind_interp=floor(pol_ind_interp);
    % interpolate the policy function according to the current K_agg
    %we get the pol_indx for this period t, this pol_indx maybe in the
    %decimal form,I am a little bit slappy here that I do not give weight
    %to the MuPrime
    
    %from the sequence of A1, we can know every PI_z we use:
    
    PI_zz=PI_z(:,:,t);
    
    % update distribution according to interpolated policy function
    
    % how to deal with the odd indices? For example, what if the
    % interpolated policy index in state (4,10,2) is 35.4? 
    % In that case we would distribute the mass in the distribution that is
    % in Mu(4,10,2) between cells 35 and 36 in MuPrime; in particular we
    % would put 40% into MuPrime(:,35,:) and 60% into MuPrime(:,36,:) (note
    % that the ':' states are K, which is determined by G~K, and the (A,z)
    % exogenous state which is determined by the simulated A shocks and the
    % PI matrix for transitions between z)
    %MuNew = 
    %...
    mu_tol = 1;
while mu_tol > 1e-08
    [z_ind, a_ind] = find(Mu > 0); % find non-zero indices
    
    MuNew = zeros(size(Mu));
    for ii = 1:length(z_ind)
        apr_ind = pol_ind_interp(z_ind(ii), a_ind(ii)); 
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
            (PI_zz(z_ind(ii), :) * Mu(z_ind(ii), a_ind(ii)) )';
    end

    mu_tol = max(abs(MuNew(:) - Mu(:)));
    
    Mu = MuNew ;
end
    
    
    Mu = MuNew ;
    
end

[AB_indx]=find(A1> 1);
[AG_indx]=find(A1==1);
%1 for G state
K1=K_series(AG_indx(1:size(AG_indx)-1));
K1_pri=K_series(AG_indx(1:size(AG_indx)-1)+1);
K2=K_series(AB_indx(1:size(AB_indx)-1));
K2_pri=K_series(AB_indx(1:size(AB_indx)-1)+1);
%for ttt=1:4999
    %if A1(ttt)==1
     %   K1(ttt)=K_series(ttt);
     %   K1_pri(ttt)=K_series(ttt+1);
    %else 
   %     K2(ttt)=K_series(ttt);
  %      K2_pri(ttt)=K_series(ttt+1);
 %   end 
%end 
%K1=K1(K1~=0);
%K1_pri=K1_pri(K1_pri~=0);
%K2=K2(K2~=0);

X1=[ones(size(K1,1),1) K1];
Y1=K1_pri;
b1=inv(X1'*X1)*X1'*Y1;

 X2=[ones(size(K2,1),1) K2];
Y2=K2_pri;
b2=inv(X2'*X2)*X2'*Y2;
coeffs_new=[b1';b2'];
coeff_tol=max(coeffs(:)-coeffs_new(:));
coeffs=.9*coeffs+.1*coeffs_new;

end 