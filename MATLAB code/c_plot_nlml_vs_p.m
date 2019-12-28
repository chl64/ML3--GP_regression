load('cw1a.mat');

meanfunc = [];
covfunc = @covPeriodic;
likfunc = @likGauss;

% Prior hyp.
hyp.mean = [];
log_sn = 0; hyp.lik = log_sn;

figure();
hold on;

for log_p = 0.4:-0.01:-1.4
    log_ell = -1; log_sf = 0; hyp.cov = [log_ell; log_p; log_sf];

    % Minimise the nlml so as to optimise hyp.
    hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

    % Train the GP with optimised hyp2.
    nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    plot(log_p, nlml, 'r.')
    plot(hyp2.cov(2), nlml, 'kx')
end

title('Relationship b/w model fit metric (nlml) and covariance period (p)')
xlabel('log(p)')
ylabel('nlml')
legend('initial log(p)', 'optimised log(p)')