% close all;

meanfunc = [];
covfunc = {@covProd, {@covPeriodic, @covSEiso}};
likfunc = @likGauss;

% Final hyp.
% hyp.mean = [];

% for covPeriodic.
log_ell1 = -0.5; log_p = 0; log_sf1 = 0; 
% for covSEiso.
log_ell2 = 2; log_sf2 = 1;
hyp.cov = [log_ell1; log_p; log_sf1; log_ell2; log_sf2];  

% log_sn = -1000000000000; hyp.lik = log_sn;

n =200;
x = linspace(-5,5,n)';
K = feval(covfunc{:}, hyp.cov, x);

figure();
hold on;
for seed = 5:7
    z = gpml_randn(seed, n, 1);
    y = chol( K + (1e-6*eye(n)) )'*z;
    plot(x,y)
end
xlabel('x')
ylabel('f(x)')
legend('random function 1', 'random function 2', 'random function 3')
title(sprintf('Random functions from Gaussian Process with covariance function \n being the product of Squared Exponential and Periodic function'))
hold off;

% figure()
% L = chol(K + (1e-6*eye(n)) )';
% indices = linspace(1,200,200);
% plot(indices, L(100,:));
% hold on
% plot(indices, L(150,:));
% plot(indices, L(200,:));
% xlabel('index of element in the row')
% ylabel('"standard deviation" contributing to x_i')
% legend('row i=100 in R transposed', 'row i=150 in R transposed', 'row i=200 in R transposed')
% title(sprintf('"standard deviations" of f(x_i) contributed by the i-th row of R transposed'))