clc
clear
close
eta = 0.5;
f = '2';
% Import Data
X1 = importdata('c1.txt');   
X2 = importdata('c2.txt');

%Determine the maximum and minimum value for feature 1 (Will be used for determining the region)
maxdatax1 = max([max(X1(:,1)) , max(X2(:,1))]);
mindatax1 = min([min(X1(:,1)) , min(X2(:,1))]);

%Determine the maximum and minimum value for feature 2 (Will be used for determining the region)
maxdatax2 = max([max(X1(:,2)) , max(X2(:,2))]);
mindatax2 = min([min(X1(:,2)) , min(X2(:,2))]);


size1 = size(X1);                                                           %size of Class1
size2 = size(X2);                                                           %size of Class2

%Defining 80 percent training data
N1 = round(0.8*size1(1));                                                   
N2 = round(0.8*size2(1));

%Rest 20 Percent test data
testsize1 = size1(1)-N1;
testsize2 = size2(1)-N2;

%Preallocation of memory and Making the Train matrices
trainx1 = zeros(N1,2);
trainx2 = zeros(N2,2);

for i = 1:N1
    trainx1(i,:) = X1(i,:);
end
for i = 1:N2
    trainx2(i,:) = X2(i,:);
end


for i = 1:N1
    trainx1(i,3) = 1;
end
for i = 1:N2
    trainx2(i,3) = -1;
end

trainx = [trainx1;trainx2];
x = trainx(randperm(size(trainx, 1)), :);

d = zeros(1,size(x,1));
for i = 1:size(x,1)
    d(i) = x(i,3);
end

for i = 1:size(x,1)
    x(i,3) = x(i,2);
    x(i,2) = x(i,1);
    x(i,1) = 1;
end

w = [0,0,0];
y = zeros(1,size(x,1));
i1=0;
while 1
    for i = 1:size(x,1)
        y(i) = sign(w*x(i,:)');
        w = w + eta*(d(i)-y(i))*x(i,:);
      
    end
    i1 = i1+1;
    if y==d
        break
    end
end
   

%Preallocation of memory and Making the Test matrices

testx1 = zeros(testsize1,2);
testx2 = zeros(testsize2,2);

for i = N1+1:testsize1+N1
    testx1(i-N1,:) = X1(i,:);
end
for i = N2+1:testsize2+N2
    testx2(i-N2,:) = X2(i,:);
end

testx1(:,3)=testx1(:,2);
testx1(:,2)=testx1(:,1);
testx1(:,1)=1;

testx2(:,3)=testx2(:,2);
testx2(:,2)=testx2(:,1);
testx2(:,1)=1;

ytest1 = sign(w*testx1');
ytest2 = sign(w*testx2');

dtest1 = ones(1,testsize1);
dtest2 = -ones(1,testsize2);

e1 = 1/2*(dtest1-ytest1);
e2 = -1/2*(dtest2-ytest2);

scatter(x(:,2),x(:,3))
hold on 
plotline(w,mindatax1,maxdatax1);
% hold on
% scatter(testx1(:,2),testx1(:,3));
% hold on
% scatter(testx2(:,2),testx2(:,3));

confusion = [testsize1-sum(e1) sum(e2);sum(e1) testsize2-sum(e2)];
disp('Confusion matrix');
disp(confusion);

%Accuracy
Acc = (confusion(1,1)+confusion(2,2))/(testsize1+testsize2)*100;
disp('Accuracy');
disp(Acc);

%Recall
rec1 = (confusion(1,1)/(confusion(1,1)+confusion(1,2)))*100;
rec2 = (confusion(2,2)/(confusion(2,1)+confusion(2,2)))*100;

avgrec = (rec1+rec2)/2;
disp('Recall for class 1');
disp(rec1);
disp('Recall for class 2');
disp(rec2);

disp('Average Recall ');
disp(avgrec);

%Precision
pre1 = (confusion(1,1)/(confusion(1,1)+confusion(2,1)))*100;
pre2 = (confusion(2,2)/(confusion(1,2)+confusion(2,2)))*100;

avgpre = (pre1+pre2)/2;
disp('Precision for class 1');
disp(pre1);
disp('Precision for class 2');
disp(pre2);

disp('Average Precision ');
disp(avgpre);

%fscore
fscore1 = 2*(pre1*rec1)/(pre1 + rec1);
fscore2 = 2*(pre2*rec2)/(pre2 + rec2);

avgfscore = (fscore1+fscore2)/2;

disp('F-score for class 1');
disp(fscore1);
disp('F-score for class 2');
disp(fscore2);
disp('Average F-score ');
disp(avgfscore);
            
    
    