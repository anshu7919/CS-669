function [w,error] = percep(trainx1,trainx2)
    n1=100;
    N1 = size(trainx1,1);
    N2 = size(trainx2,1);
    eta = 0.5;
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
while i1<n1
    for i = 1:size(x,1)
        y(i) = sign(w*x(i,:)');
        w = w + eta*(d(i)-y(i))*x(i,:);
      
    end
    i1 = i1+1;
    error(i1) = sum(abs(y-d));
    if y==d
        break
    end
end

return
   
