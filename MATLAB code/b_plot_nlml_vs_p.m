load('cw1a.mat');

meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;

% Prior hyp.
hyp.mean = [];

figure();
hold on;

% for log_ell = -12:0.1:0
%     log_sf = 0; hyp.cov = [log_ell; log_sf];
%     log_sn = 0; hyp.lik = log_sn;

% for log_sf = -2.3:0.1:1
%     log_ell = -1; hyp.cov = [log_ell; log_sf];
%     log_sn = 0; hyp.lik = log_sn;

for log_sn = 1:-0.1:-9.5
    log_ell = -1; log_sf = 0; hyp.cov = [log_ell; log_sf];
    hyp.lik = log_sn;
    
    % Minimise the nlml so as to optimise hyp.
    hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

    % Train the GP with optimised hyp2.
    nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
%     plot(log_ell, nlml, 'r.')
%     plot(hyp2.cov(1), nlml, 'kx')
    
%     plot(log_sf, nlml, 'r.')
%     plot(hyp2.cov(2), nlml, 'kx')
    
    plot(log_sn, nlml, 'r.')
    plot(hyp2.lik, nlml, 'kx')
    
end


ylabel('nlml')

% title(sprintf('Relationship b/w model fit metric (nlml)\nand length scale (ell)'))
% xlabel('log(ell)')
% legend('initial log(ell)', 'optimised log(ell)')

% title(sprintf('Relationship b/w model fit metric (nlml)\nand signal variance (sf)'))
% xlabel('log(sf)')
% legend('initial log(sf)', 'optimised log(sf)')

title(sprintf('Relationship b/w model fit metric (nlml)\nand noise variance (sn)'))
xlabel('log(sn)')
legend('initial log(sn)', 'optimised log(sn)')