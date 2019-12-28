%close all;
load('cw1a.mat');

meanfunc = [];
covfunc = @covPeriodic;
likfunc = @likGauss;

% Prior hyp.
hyp.mean = [];
log_ell = -1; log_p = 0.3; log_sf = 0; hyp.cov = [log_ell; log_p; log_sf];  
log_sn = 0; hyp.lik = log_sn;

% Minimise the nlml so as to optimise hyp.
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% Train the GP with optimised hyp2.
% Since predictive error bars are desired, return predictive output mean
% and *variance* from equally spaced test inputs.
xs = linspace(-4,4,801)';
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% Compute 95% predictive error bar, and plot the prediction and the
% training data.
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
figure();
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu, 'b'); plot(x, y, 'r+')

nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
hyp2_cov = hyp2.cov
hyp2_lik = hyp2.lik

fname = sprintf('log(ell)=%g, log(p)=%g, log(sf)=%g, log(sn)=%g', log_ell, log_p, log_sf, log_sn);
str_hyp2 = sprintf('\nOptimised hyp: log(ell)=%g, log(p)=%g, log(sf)=%g, log(sn)=%g', hyp2.cov(1), hyp2.cov(2), hyp2.cov(3), hyp2.lik);
str_nlml = sprintf('\nFinal nlml after training = %g', nlml);
title( strcat('Initial hyp:', fname, str_hyp2, str_nlml) );

%title('Training data (red), predictive mean function (blue), and 95% predictive error bars (grey)');
xlabel('1D input value, x');
ylabel('1D output value, y');
legend('95% predictive error bars', 'predictive mean function', 'Training data');


folder = 'D:\Uni\IIB\Michaelmas (5)\(C) 4F13 Probabilistic machine learning\Coursework#1 - Regression about Gaussian Processes\c_results';
suffix = '.fig';
fFullname = fullfile(folder, strcat(fname,suffix));
savefig(fFullname);