clc
clear
close all
f = 'linear2';


% Import Data
X1 = importdata('Clas1_b7.txt');   
X2 = importdata('Clas2_b7.txt');
X3 = importdata('Clas3_b7.txt');

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
%storing of training data for class 1,2,3
for i = 1:N1
    trainx1(i,:) = X1(i,:);
end
for i = 1:N2
    trainx2(i,:) = X2(i,:);
end
for i = 1:N3
    trainx3(i,:) = X3(i,:);
end
train1r = [trainx2;trainx3];
train2r = [trainx1;trainx3];
train3r = [trainx2;trainx1];

%for class1 vs rest
[w1,error1] = percep(trainx1,train1r);

%for class1 vs rest
[w2,error2] = percep(trainx2,train2r);

%for class1 vs rest
[w3,error3] = percep(trainx3,train3r);



%Preallocation of memory and Making the Test matrices

testx1 = zeros(testsize1,2);
testx2 = zeros(testsize2,2);
testx3 = zeros(testsize3,2);
%storing of test data
for i = N1+1:testsize1+N1
    testx1(i-N1,:) = X1(i,:);
end
for i = N2+1:testsize2+N2
    testx2(i-N2,:) = X2(i,:);
end
for i = N3+1:testsize3+N3
    testx3(i-N3,:) = X3(i,:);
end

testx1(:,3)=testx1(:,2);
testx1(:,2)=testx1(:,1);
testx1(:,1)=1;

testx2(:,3)=testx2(:,2);
testx2(:,2)=testx2(:,1);
testx2(:,1)=1;

testx3(:,3)=testx3(:,2);
testx3(:,2)=testx3(:,1);
testx3(:,1)=1;

%class 1 test
ytest11 = sign(w1*testx1');
ytest12 = sign(w2*testx1');
ytest13 = sign(w3*testx1'); 

dtest1 = ones(1,testsize1);
dtest2 = -ones(1,testsize1);

e1 = 1/2*(dtest1-ytest11);
e2 = -1/2*(dtest2-ytest12);
e3 = -1/2*(dtest2-ytest13);

c11 = testsize1 - sum(e1);
c12 = sum(e2);
c13 = sum(e3);


%class 2 test
ytest21 = sign(w1*testx2');
ytest22 = sign(w2*testx2');
ytest23 = sign(w3*testx2'); 

dtest1 = ones(1,testsize2);
dtest2 = -ones(1,testsize2);

e1 = -1/2*(dtest2-ytest21);
e2 = 1/2*(dtest1-ytest22);
e3 = -1/2*(dtest2-ytest23);

c21 = sum(e1);
c22 = testsize2 - sum(e2);
c23 = sum(e3);

%class 3 test
ytest31 = sign(w1*testx3');
ytest32 = sign(w2*testx3');
ytest33 = sign(w3*testx3'); 

dtest1 = ones(1,testsize3);
dtest2 = -ones(1,testsize3);

e1 = -1/2*(dtest2-ytest31);
e2 = -1/2*(dtest2-ytest32);
e3 = 1/2*(dtest1-ytest33);

c31 = sum(e1);
c32 = sum(e2);
c33 = testsize3 - sum(e3);

confusion = [c11 c12 c13;c21 c22 c23;c31 c32 c33];

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

scatter(X1(:,1),X1(:,2),'r')
hold on 
plotline(w1,mindatax1,maxdatax1);
hold on

scatter(X2(:,1),X2(:,2),'g')
hold on 
plotline(w2,mindatax1,maxdatax1);
hold on

scatter(X3(:,1),X3(:,2),'b')
hold on 
plotline(w3,mindatax1,maxdatax1);




