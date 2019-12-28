% close all;
clear all;
load('cw1e.mat')

meanfunc = [];
likfunc = @likGauss;

covfunc = {@covSum, {@covSEard, @covSEard}}
hyp.cov = 0.1 * randn(6, 1);

hyp.lik = 0;

% ---

% Minimise the nlml so as to optimise hyp.
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% Train the GP with optimised hyp2.
% Since predictive error bars are desired, return predictive output mean
% and *variance* from equally spaced test inputs.
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x);

figure();
CO_data(:,:,1) = ones(11)*0.8; % red
CO_data(:,:,2) = ones(11)*0.2; % green
CO_data(:,:,3) = ones(11)*0.2; % blue
mesh(reshape(x(:,1),11,11), reshape(x(:,2),11,11),reshape(y,11,11),CO_data, 'FaceAlpha','0.5', 'FaceColor', 'flat')

hold on;
CO_mu(:,:,1) = ones(11)*0; % red
CO_mu(:,:,2) = ones(11)*0; % green
CO_mu(:,:,3) = ones(11)*0; % blue

mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(mu,11,11), CO_mu, 'FaceAlpha','0.5', 'FaceColor', 'flat')


CO_s2(:,:,1) = ones(11)*0.8; % red
CO_s2(:,:,2) = ones(11)*0.8; % green
CO_s2(:,:,3) = ones(11)*0.8; % blue

mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(mu+2*sqrt(s2),11,11), CO_s2, 'FaceAlpha','0.1', 'FaceColor', 'flat')
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(mu-2*sqrt(s2),11,11), CO_s2, 'FaceAlpha','0.1', 'FaceColor', 'flat')

xlabel('x1')
ylabel('x2')
zlabel('y')

nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

title(strcat(sprintf('nlml = %g, covfunc = covSEard_1 + covSEard_2; \n Red: Training data, black: predictive mean function, \n light grey: 95', nlml),'% predictive error bars'))

% -----------------------------
figure();
Y = reshape(y,11,11);
X2 = reshape(x(:,2),11,11);
Mu = reshape(mu,11,11);
Upper = reshape(mu+2*sqrt(s2),11,11);
Lower = reshape(mu-2*sqrt(s2),11,11);

col = 4;

f = [Upper(:,col); flipdim(Lower(:,col),1)];
fill([X2(:,col); flipdim(X2(:,col),1)], f, [7.5 7.5 7.5]/8);
hold on
plot(X2(:,col), Mu(:,col), 'k--');
plot(X2(:,col), Y(:,col), 'ro');
plot(X2(:,col), Y(:,col), 'r');
xlabel('x2 at x1=-2');
ylabel('y')
legend('95% predictive error bars', 'predictive mean function', 'training data')
title('Performance of the sum of two covSEard')

hyp2_cov = hyp2.cov
hyp2_lik = hyp2.lik