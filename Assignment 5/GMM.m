function [mean1,Var,piGMM,J1]=GMM(k,trainx1)
N1 = size(trainx1,1);
ep = 10^(-4);
%Threshold on No. of iterations
n = 100;

%Treshold on the cost J
n1= 0.1;

%Initialise the GMM
minx11 = min(trainx1(:,1));
maxx11 = max(trainx1(:,1));
minx12 = min(trainx1(:,2));
maxx12 = max(trainx1(:,2));
Ix1 = linspace(minx11,maxx11,k+1);
Ix2 = linspace(minx12,maxx12,k+1);

% Ix1 = abs(minx11-maxx11)*rand(1,k+1)+abs(minx11-maxx11)/2;
% Ix2 = abs(minx12-maxx12)*rand(1,k+1)+abs(minx12-maxx12)/2;

meanx1 = zeros(1,k);
meanx2 = zeros(1,k);
mean1 = zeros(k,2);
for i=1:k
    meanx1(i)=(Ix1(i)+Ix1(i+1))/2;
    meanx2(i)=(Ix2(i)+Ix2(i+1))/2;
end
for i= 1:k
    mean1(i,:) = [meanx1(i),meanx2(i)];
end
Var1 = abs(minx11-maxx11)/k*eye(2);
Var2 = abs(minx11-maxx11)/k*eye(2);
Var3 = abs(minx11-maxx11)/k*eye(2);
piGMM = 1/k*ones(1,k);
for i= 1:k
    Var(:,:,i) = Var1;
end

i1=2;
J1 = [0 0];

while i1<n

Rnk = zeros(N1,k);
d = zeros(N1,k);

%Calculating the distances of every data point form the k-means and finding
%which data point is alooted which mean
for i=1:N1
    for j=1:k
        d(i,j) = gs(trainx1(i,:),mean1(j,:),Var(:,:,j));
        Rnk(i,j) = piGMM(j)*d(i,j);
    end
    denRnk(i) = sum(Rnk(i,:));
    
end
for i = 1:N1
    Rnk(i,:) = Rnk(i,:)./denRnk(i);
end
    
% max(d)
%defining the cost
J1(i1) = 0;
sumRnk = zeros(1,k);
for i=1:N1
    for j = 1:k
        J1(i1) = J1(i1) + log(piGMM(j)*d(i,j));
        
    end
end
for j = 1:k
    sumRnk(j) = sum(Rnk(:,j));
end

%Updating the means 
mean1 = zeros(k,2);
for i=1:N1
    for j = 1:k
        temp = Rnk(i,j)*trainx1(i,:);
        mean1(j,:) = mean1(j,:)+temp;
    end
end
for i =1:k
    if sumRnk(i)~=0
        mean1(i,:) = mean1(i,:)/sumRnk(i);
    else
        [c v] = min(d);
        mean1(i,:) = trainx1(v(i),:);
    end
end

%Updating the variances
Var = zeros(2,2,k);
for i=1:N1
    for j = 1:k
        Var(:,:,j) =Var(:,:,j)+ Rnk(i,j)*(trainx1(i,:)-mean1(j,:))'*(trainx1(i,:)-mean1(j,:));
    end
end
for i = 1:k
    if sumRnk(i)~=0
        Var(:,:,i) = Var(:,:,i)/sumRnk(i);
    else
        Var(:,:,i) = 5*eye(2);
    end
end

%Updating piGMM
for i = 1:k
    piGMM(i) = sumRnk(i)/N1;
end
% figure;
% scatter(mean1(:,1),mean1(:,2));

%Checking if previous cost is near to the present cost end the itereation
if (abs(J1(i1)-J1(i1-1))<n1)
    break;
end
i1 = i1+1;
J1 = [J1 0];
% J1
end
return