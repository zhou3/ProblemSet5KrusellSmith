% draw random numbers for simulation
T_sim = 5500;
%rng(1);
A=zeros(T_sim,1);
A(1)=1;% initialize the first state to be G,1
i=1;
%jj=rand(5500,1);
for j=1:5500
    jj=rand;
if jj<=PI(i,1) %sum( jj(:)>0 & jj(:)<0.1 )  
    ii=1;
elseif PI(i,1)<=jj && jj<=PI(i,1)+PI(i,2)
    ii=2;
elseif PI(i,1)+PI(i,2)<=jj && jj<=PI(i,1)+PI(i,2)+PI(i,3)
    ii=3;
else
    ii=4;

end 
%i=i;
i=ii;
A(j+1)=ii;
end 
%sum( jj(:)>0 & jj(:)<0.1 )  
%simulate various A state.

AA=A(501:5500,1);
A1=AA;
for ii=1:5000
    if A1(ii)==3;
        A1(ii)=1;
  elseif A1(ii)==4;
        A1(ii)=2;
    end 
end 



% then A1 is the pure state of good or bad.
 % then AA is the combined state(combined with z)