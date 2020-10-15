clc
clear
close all
f = 'Nonlinear3';

%Parzen Window size
h = 6;


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
%storing of training data in trainx1,x2,x3
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
%storing of testdata of class1,2,3
for i = N1+1:testsize1+N1
    testx1(i-N1,:) = X1(i,:);
end
for i = N2+1:testsize2+N2
    testx2(i-N2,:) = X2(i,:);
end
for i = N3+1:testsize3+N3
    testx3(i-N3,:) = X3(i,:);
end

%Clubing all the 3 class data into one train matrix
train = [trainx1;trainx2;trainx3];

%First N1 data belong to class 1 
d1 = ones(N1,1);

%N1+1 to N2 data belong to class 2 
d2 = 2*ones(N2,1);

%N2+1 to N3 data belong to class 3 
d3 = 3*ones(N3,1);

%Ground truth vector 
d = [d1;d2;d3];

% test for class 1
for i = 1:testsize1
    k1 = 0;
    k2 = 0;
    k3 = 0;
    for j = 1:N1+N2+N3
        
        %finding distance from the given test sample to all the training
        %samples
        ker = (testx1(i,:)-train(j,:))*(testx1(i,:)-train(j,:))';
        
        %if the distance is less than h/2 than we include that data in k
        if ker < h/2 
            if d(j) == 1                %k belongs to class 1
                k1 = k1 + 1;
            elseif d(j) == 2            %k belongs to class 2
                k2 = k2 + 1;
            else 
                k3 = k3 + 1;            %k belongs to class 3
                
            end
        end
    end
    if k1+k2+k3~=0       
    %Calculating the probabilities    
    p11(i) = k1/(k1+k2+k3);             
    p12(i) = k2/(k1+k2+k3);
    p13(i) = k3/(k1+k2+k3);
    [p1(i),I1(i)] = max([p11(i),p12(i),p13(i)]);
    end
end

% test for class 2
for i = 1:testsize2
    k1 = 0;
    k2 = 0;
    k3 = 0;
    for j = 1:N1+N2+N3
        
        ker = (testx2(i,:)-train(j,:))*(testx2(i,:)-train(j,:))';
        if ker < h/2 
            if d(j) == 1
                k1 = k1 + 1;
            elseif d(j) == 2
                k2 = k2 + 1;
            else 
                k3 = k3 + 1;
                
            end
        end
    end
    if k1+k2+k3~=0
    p21(i) = k1/(k1+k2+k3);
    p22(i) = k2/(k1+k2+k3);
    p23(i) = k3/(k1+k2+k3);
    [p2(i),I2(i)] = max([p21(i),p22(i),p23(i)]);
    end
end

% test for class 3
for i = 1:testsize3
    k1 = 0;
    k2 = 0;
    k3 = 0;
    for j = 1:N1+N2+N3
        
        ker = (testx3(i,:)-train(j,:))*(testx3(i,:)-train(j,:))';
        if ker < h/2 
            if d(j) == 1
                k1 = k1 + 1;
            elseif d(j) == 2
                k2 = k2 + 1;
            else 
                k3 = k3 + 1;
                
            end
        end
    end
    if k1+k2+k3~=0
        p31(i) = k1/(k1+k2+k3);
        p32(i) = k2/(k1+k2+k3);
        p33(i) = k3/(k1+k2+k3);
        [p3(i),I3(i)] = max([p31(i),p32(i),p33(i)]);
    end
end

   %Creating the Confusion Matrix
confusion = zeros(3,3);
for i = 1:testsize1
    if I1(i) == 1
        confusion(1,1) = confusion(1,1) + 1;
    elseif I1(i) == 2
        confusion(2,1) = confusion(2,1) + 1;
    elseif I1(i) == 3
        confusion(3,1) = confusion(3,1) + 1;
    end
end
for i = 1:testsize2
    if I2(i) == 1
        confusion(1,2) = confusion(1,2) + 1;
    elseif I2(i) == 2
        confusion(2,2) = confusion(2,2) + 1;
    elseif I2(i) == 3
        confusion(3,2) = confusion(3,2) + 1;
    end
end
for i = 1:testsize3
    if I3(i) == 1
        confusion(1,3) = confusion(1,3) + 1;
    elseif I3(i) == 2
        confusion(2,3) = confusion(2,3) + 1;
    elseif I3(i) == 3
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
scatter(X1(:,1),X1(:,2),5,'r');
hold on
scatter(X2(:,1),X2(:,2),5,'g');
hold on
scatter(X3(:,1),X3(:,2),5,'b');

%Region Boundaries
        
%Making a vector having 100 values from min to max  for each feature (Will %be used for determining the Region)
x1 = linspace(mindatax1,maxdatax1,100);
x2 = linspace(mindatax2,maxdatax2,100);
x = [x1' x2'];

% Preallocating Memory for determining which grid point lies in which class
class1 =[];
class2 =[];
class3 =[];
no = [];

for i1 = 1:100
    for j1 = 1:100
        k1 = 0;
        k2 = 0;
        k3 = 0;
        for j = 1:N1+N2+N3
        
        ker = ([x1(i1),x2(j1)]-train(j,:))*([x1(i1),x2(j1)]-train(j,:))';
        if ker < h/2 
            if d(j) == 1
                k1 = k1 + 1;
            elseif d(j) == 2
                k2 = k2 + 1;
            else 
                k3 = k3 + 1;
                
            end
        end
    end
    if k1+k2+k3~=0
    p01 = k1/(k1+k2+k3);
    p02 = k2/(k1+k2+k3);
    p03 = k3/(k1+k2+k3);
    [p2,I] = max([p01,p02,p03]);
    if I == 1
            class1 =[class1;[x1(i1),x2(j1)]];

    elseif I == 2
            class2= [class2;[x1(i1),x2(j1)]];

    elseif I == 3
            class3= [class3;[x1(i1),x2(j1)]];
    end
    else
        no = [no;[x1(i1),x2(j1)]];
    end
end
end        

%Plotting the Region of Discrimination 

figure;
scatter(class1(:,1),class1(:,2),5,'r');
hold on
scatter(class2(:,1),class2(:,2),5,'g');
hold on
scatter(class3(:,1),class3(:,2),5,'b');


%         
        
        