clc
clear
close all
f = 'Nonlinear3';

%Threshold on No. of iterations
n = 100;

%Treshold on the cost J
n1= 0.1;

% Import Data
X1 = importdata('Clas1_b7.txt');   
X2 = importdata('Clas2_b7.txt');
X3 = importdata('Clas3_b7.txt');

% X1 = 10*X1;
% X2 = 10*X2;
% X3 = 10*X3;


%Determine the maximum and minimum value for feature 1 (Will be used for determining the region)
maxdatax1 = max([max(X1(:,1)) , max(X2(:,1)) , max(X3(:,1))]);
mindatax1 = min([min(X1(:,1)) , min(X2(:,1)) , min(X3(:,1))]);

%Determine the maximum and minimum value for feature 2 (Will be used for determining the region)
maxdatax2 = max([max(X1(:,2)) , max(X2(:,2)) , max(X3(:,2))]);
mindatax2 = min([min(X1(:,2)) , min(X2(:,2)) , min(X3(:,2))]);

size1 = size(X1);                                                           %size of Class1
size2 = size(X2);                                                           %size of Class2
size3 = size(X3);                                                           %size of Class3

%Defining 80 percent training data
N1 = round(0.8*size1(1));                                                   
N2 = round(0.8*size2(1));
N3 = round(0.8*size3(1));

%Rest 20 Percent test data
testsize1 = size1(1)-N1;
testsize2 = size2(1)-N2;
testsize3 = size3(1)-N3;

%Preallocation of memory and Making the Train matrices
trainx1 = zeros(N1,2);
trainx2 = zeros(N2,2);
trainx3 = zeros(N3,2);

for i = 1:N1
    trainx1(i,:) = X1(i,:);
end
for i = 1:N2
    trainx2(i,:) = X2(i,:);
end
for i = 1:N3
    trainx3(i,:) = X3(i,:);
end

%Preallocation of memory and Making the Test matrices

testx1 = zeros(testsize1,2);
testx2 = zeros(testsize2,2);
testx3 = zeros(testsize3,2);

for i = N1+1:testsize1+N1
    testx1(i-N1,:) = X1(i,:);
end
for i = N2+1:testsize2+N2
    testx2(i-N2,:) = X2(i,:);
end
for i = N3+1:testsize3+N3
    testx3(i-N3,:) = X3(i,:);
end


k=10;

[mean1,Var1,piGMM1,J1] = GMM(k,trainx1);
[mean2,Var2,piGMM2,J2] = GMM(k,trainx2);
[mean3,Var3,piGMM3,J3] = GMM(k,trainx3);


% Training 
scatter(trainx1(:,1),trainx1(:,2),'r');
hold on
scatter(trainx2(:,1),trainx2(:,2),'g');
hold on
scatter(trainx3(:,1),trainx3(:,2),'b');
hold on
for i=1:k
    plotgauss(mean1(i,:),Var1(:,:,i),'r')
    hold on
end

for i=1:k
    plotgauss(mean2(i,:),Var2(:,:,i),'g')
    hold on
end

for i=1:k
    plotgauss(mean3(i,:),Var3(:,:,i),'b')
    hold on
end

%Test Phase

prob1 = zeros(1,testsize1);
prob2 = zeros(1,testsize2);
prob3 = zeros(1,testsize3);

%for class 1
for i=1:testsize1
    d1 = 0;
    d2 = 0;
    d3 = 0;
    for j = 1:k
        d1 = d1+piGMM1(j)*gs((testx1(i,:)),mean1(j,:),Var1(:,:,j));
        d2 = d2+piGMM2(j)*gs((testx1(i,:)),mean2(j,:),Var2(:,:,j));
        d3 = d3+piGMM3(j)*gs((testx1(i,:)),mean3(j,:),Var3(:,:,j));
    end
    
    mink(1) = d1;
    mink(2) = d2;
    mink(3) = d3;
    
    [mind1,I1] = max(mink);
    prob1(i) = I1;
    
end

%for class 2
for i=1:testsize2
    d1 = 0;
    d2 = 0;
    d3 = 0;
    for j = 1:k
        d1 = d1+piGMM1(j)*gs((testx2(i,:)),mean1(j,:),Var1(:,:,j));
        d2 = d2+piGMM2(j)*gs((testx2(i,:)),mean2(j,:),Var2(:,:,j));
        d3 = d3+piGMM3(j)*gs((testx2(i,:)),mean3(j,:),Var3(:,:,j));
    end
    
    mink(1) = d1;
    mink(2) = d2;
    mink(3) = d3;
    
    [mind2,I2] = max(mink);
    prob2(i) = I2;
    
end

%for class 3
for i=1:testsize3
    d1 = 0;
    d2 = 0;
    d3 = 0;
    for j = 1:k
        d1 = d1+piGMM1(j)*gs((testx3(i,:)),mean1(j,:),Var1(:,:,j));
        d2 = d2+piGMM2(j)*gs((testx3(i,:)),mean2(j,:),Var2(:,:,j));
        d3 = d3+piGMM3(j)*gs((testx3(i,:)),mean3(j,:),Var3(:,:,j));
    end
    
    mink(1) = d1;
    mink(2) = d2;
    mink(3) = d3;
    
    [mind3,I3] = max(mink);
    prob3(i) = I3;
    
end

%Making a vector having 100 values from min to max  for each feature (Will %be used for determining the Region)
x1 = linspace(mindatax1,maxdatax1,100);
x2 = linspace(mindatax2,maxdatax2,100);
x = [x1' x2'];

% Preallocating Memory for determining which grid point lies in which class
class1 =[];
class2 =[];
class3 =[];
% Loop for determining which grid point lies in which class
for i = 1:100
    for j = 1:100
        d1 = 0;
        d2 = 0;
        d3 = 0;
        for l = 1:k
            
            d1 = d1+piGMM1(l)*gs([x1(i),x2(j)],mean1(l,:),Var1(:,:,l));
            d2 = d2+piGMM2(l)*gs([x1(i),x2(j)],mean2(l,:),Var2(:,:,l));
            d3 = d3+piGMM3(l)*gs([x1(i),x2(j)],mean3(l,:),Var3(:,:,l));
        end
    
        mink(1) = d1;
        mink(2) = d2;
        mink(3) = d3;
    
        [mind3,I] = max(mink);
        
        if I == 1
            class1 =[class1;[x1(i),x2(j)]];

        elseif I == 2
            class2= [class2;[x1(i),x2(j)]];

        elseif I == 3
            class3= [class3;[x1(i),x2(j)]];
        end
    end    
end
        
%Creating the Confusion Matrix
confusion = zeros(3,3);
for i = 1:testsize1
    if prob1(i) == 1
        confusion(1,1) = confusion(1,1) + 1;
    elseif prob1(i) == 2
        confusion(2,1) = confusion(2,1) + 1;
    elseif prob1(i) == 3
        confusion(3,1) = confusion(3,1) + 1;
    end
end
for i = 1:testsize2
    if prob2(i) == 1
        confusion(1,2) = confusion(1,2) + 1;
    elseif prob2(i) == 2
        confusion(2,2) = confusion(2,2) + 1;
    elseif prob2(i) == 3
        confusion(3,2) = confusion(3,2) + 1;
    end
end
for i = 1:testsize3
    if prob3(i) == 1
        confusion(1,3) = confusion(1,3) + 1;
    elseif prob3(i) == 2
        confusion(2,3) = confusion(2,3) + 1;
    elseif prob3(i) == 3
        confusion(3,3) = confusion(3,3) + 1;
    end
end



disp('Confusion matrix');
disp(confusion);

%Accuracy
Acc = (confusion(1,1)+confusion(2,2)+confusion(3,3))/(testsize1+testsize2+testsize3)*100;
disp('Accuracy');
disp(Acc);

%Recall
rec1 = (confusion(1,1)/(confusion(1,1)+confusion(1,2)+confusion(1,3)))*100;
rec2 = (confusion(2,2)/(confusion(2,1)+confusion(2,2)+confusion(2,3)))*100;
rec3 = (confusion(3,3)/(confusion(3,1)+confusion(3,2)+confusion(3,3)))*100;

avgrec = (rec1+rec2+rec3)/3;
disp('Recall for class 1');
disp(rec1);
disp('Recall for class 2');
disp(rec2);
disp('Recall for class 3');
disp(rec3);
disp('Average Recall ');
disp(avgrec);

%Precision
pre1 = (confusion(1,1)/(confusion(1,1)+confusion(2,1)+confusion(3,1)))*100;
pre2 = (confusion(2,2)/(confusion(1,2)+confusion(2,2)+confusion(3,2)))*100;
pre3 = (confusion(3,3)/(confusion(1,3)+confusion(2,3)+confusion(3,3)))*100;


avgpre = (pre1+pre2+pre3)/3;
disp('Precision for class 1');
disp(pre1);
disp('Precision for class 2');
disp(pre2);
disp('Precision for class 3');
disp(pre3);
disp('Average Precision ');
disp(avgpre);

%fscore
fscore1 = 2*(pre1*rec1)/(pre1 + rec1);
fscore2 = 2*(pre2*rec2)/(pre2 + rec2);
fscore3 = 2*(pre3*rec3)/(pre3 + rec3);

avgfscore = (fscore1+fscore2+fscore3)/3;

disp('F-score for class 1');
disp(fscore1);
disp('F-score for class 2');
disp(fscore2);
disp('F-score for class 3');
disp(fscore3);
disp('Average F-score ');
disp(avgfscore);

figure;
scatter(class1(:,1),class1(:,2),5,'r');
hold on
scatter(class2(:,1),class2(:,2),5,'g');
hold on
scatter(class3(:,1),class3(:,2),5,'b');

% %Plotting the Test data 
% 
% 
% Cost1 = [];
% Cost2 = [];
% Cost3 = [];
% 
% for i = 2:size(J1,2)
%     Cost1(i-1) = J1(i);
% end
% 
% for i = 2:size(J2,2)
%     Cost2(i-1) = J2(i);
% end
% 
% for i = 2:size(J3,2)
%     Cost3(i-1) = J3(i);
% end
% 
% figure;
% subplot(1,3,1);
% plot(Cost1);
% title('Cost for class1');
% 
% subplot(1,3,2);
% plot(Cost2);
% title('Cost for class2');
% subplot(1,3,3);
% plot(Cost3);
% title('Cost for class3');
