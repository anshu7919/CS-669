function plotgauss(mean1,covar1,c)

%Ploting the Ellipse from Covarience Matrix

%Getting the Eigen Values and Eigen vectors of Covariance Matrix
[evec1, eval1] = eig(covar1);


%Getting the maximum Eigen Value (Will be used to get Major axis of ellipse)
[maxeigval1, Ieigval1] = max(max(eval1));


%Getting the maximum Eigen Vector (Will be used to get direction Major axis of ellipse)
maxeigvec1 = evec1(:,Ieigval1);


%Getting the Minimum Eigen Value and Eigen Vector (Will be used to get Minor axis of ellipse)
if (Ieigval1 == 1)
    mineigval1 = max(eval1(:,2));
    mineigvec1 = evec1(:,2);
else
    mineigval1 = max(eval1(:,1));
    mineigvec1 = evec1(:,1);
end

%Getting the Angle of orientation of ellipse
phi1 = atan2(maxeigvec1(2),maxeigvec1(1));


%defining 95% Confidence value for ellipse
%This confidence ellipse defines the region that contains 95% of all samples that can be drawn
%from the underlying Gaussian distribution.
chival = 2.4477;
theta = linspace(0,2*pi);               %defining a range for theta

%Getting the Centre point of all the ellipses
X01 = mean1(1);
Y01 = mean1(2);


%defining the Major axis
a1 = chival*sqrt(maxeigval1);


%Defining the Minor Axis
b1 = chival*sqrt(mineigval1);


%We found the polar parameters of ellipse as they were easy will now convert these to cartesian
%Converting the Polar Form ellipse to cartesian Form
ellipse1x = a1*cos(theta);


ellipse1y = b1*sin(theta);


%Defining the Rotation matrix by the angle of orientation Calculated above 
R1 = [ cos(phi1) sin(phi1); -sin(phi1) cos(phi1) ];


%Rotating the ellipse
r_ellipse1 = [ellipse1x;ellipse1y]' * R1;


%Plotting the ellipse
plot(r_ellipse1(:,1) + X01,r_ellipse1(:,2) + Y01,c);
hold on

%Plotting the Eigen value and eigen vector
quiver(X01, Y01, maxeigvec1(1)*sqrt(maxeigval1), maxeigvec1(2)*sqrt(maxeigval1), c, 'LineWidth',2);
quiver(X01, Y01, mineigvec1(1)*sqrt(mineigval1), mineigvec1(2)*sqrt(mineigval1), c, 'LineWidth',2);
hold on

title('Training data and Gaussian ellipse');
grid on

return