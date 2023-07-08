function [Ma,U,S,V]=BeckersRixen(M0,nmin,nmax,perc,tol)
%This is a Beckers and Rixen 2003 implementation of the iterated
%EOF gap-filling technique.  The method takes a gappy dataset in M0
%M0 and produces an analysis dataset Ma of the same size.
%M0 should have rows as time-series and columns as datagrid patterns.
%
%Each iteration proceeds by SVD of the previous Ma,
% [Un,Sn,Vn]=svds(Ma(old),num)
% Then, by truncation to fewer modes n<num, an update is made
% Ma(new)=Un*Sn*Vn'
% This continues until Sn converges.
%
% The algorithm begins with n=1 and num=n+1, runs to convergence
% based on reducing error to tol*(initial error), and estimates final error.
% Next, the iteration is repeated with higher n and error is estimated
% When the estimated error increases, the previous n result is saved.
% nmax sets an upper limit of the number of modes to include.
%
% The error is estimated by repeating perc percent of the data with
% zeros introduced.  The synthetic zeros occur in the same proportion
% by row of M0 as they do in the original.

nnow=1;
sm=size(M0);
if ~exist('nmin')  % This is the default % of modes to use
    nmin=1;
end
if ~exist('nmax')  % This is the default % of modes to use
    nmax=round(min(sm)./2);
end
if ~exist('perc')  % This is the default % of matrix to use as error est.
    perc=10;
end
if ~exist('tol')  % This is the default error tolerance
    tol=1e-4;
end
nmax=min(min(sm)-1,nmax);

%Count the zeros by row
zcount=sum(M0'==0)./sm(2);

% Add on the error estimate rows
added=ceil(sm(2)*perc/100);   % how many to add?
Mp=zeros(sm(1),sm(2)+added);  % a bigger matrix with the added rows
Mp(:,1:sm(2))=M0;
Md=Mp;
ndx=randi(sm(2),added,1);   % make a random guess of columns to include

for ii=1:added
  while prod(M0(:,ndx(ii)))==0
      ndx(ii)=randi(sm(2),1,1);  %try again until you find a column without gaps
  end
  Mp(:,sm(2)+ii)=M0(:,ndx(ii));  % Duplicate the original nonzero data
  Md(:,sm(2)+ii)=Mp(:,sm(2)+ii).*(rand(1,sm(1))>zcount)'; %zeroing some
end

zdx=find(Md==0);
adx=(sm(2)+1):(sm(2)+added);

initerror=sum(sum((Mp(:,adx)-Md(:,adx)).^2));  % The initial error, now lets reduce it!
itererror=initerror;
nerror(nnow)=0;
decerror=true;

% Heres the loop for the number of modes used
for nnow=1:nmax
 itererror=initerror;
 errorup=2*initerror;
 olderrorup=3*initerror;
 Mdnew=Md;
 
% pause
 while abs(errorup./olderrorup-1)>tol%&&errorup>tol*initerror
   
   Md(zdx)=Mdnew(zdx);
     
   [U,S,V]=svds(Md,nnow+1);
   
   for jj=(nnow+1):length(S)
    S(jj,jj)=0;   % Truncate the modes
    U(:,jj)=0;   % Truncate the modes
    V(:,jj)=0;   % Truncate the modes
   end
   Mdnew=U*S*V';  % Form the new guess
   
   itererror=sum(sum((Mp(:,adx)-Mdnew(:,adx)).^2)); %Find new error
   olderror=sum(sum((Mp(:,adx)-Md(:,adx)).^2)); %Find error update
   
   olderrorup=errorup;
   errorup=sum(sum((Mp(:,adx)-Md(:,adx)).^2))-itererror; %Find error update
   [abs(errorup./olderrorup-1),tol];
   
   %figure(100)
   %clf
   %plot(Mp(zdx),Mp(zdx),'-',Md(zdx),Mp(zdx),'k.',Mdnew(zdx),Mp(zdx),'r.')
   %legend('line','Md','Mdnew','Location','northwest')
   %axis equal
   %drawnow
   
 end
 nerror(nnow)=itererror; % Check to ensure error is decreasing with increasing n
 if nerror(nnow)>nerror(max(1,nnow-1))
  break
 end
 nnow=nnow+1;  % Take more nodes
end

Ma=Md(:,1:sm(2));
