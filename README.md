# Regression using different Gaussian Processes (GP)

Given a dataset of 1D continuous-valued input-output pairs `cw1a.mat`, Gaussian Process models with **different covariance functions** were individually fit to the data and their performance compared with respect to **marginal likelihood**. For each model, its hyper-parameters had been chosen to maximise the marginal likelihood, so as to avoid overfitting.

Below show the data fits for ***isotopic squared exponential*** and ***periodic*** covariance functions respectively:

<p align="center">
  <img width=600 src="demo_images/a_result.jpg">
</p>

<p align="center">
  <img width=530 src="demo_images/c_result.jpg">
</p>

Gaussian Process had also been used to perform regression on a 2D-input 1D-output dataset `cw1e.mat`. The best result is demonstrated below, where the red surface maps the observed data; black surface gives the prediction:

<p float="center">
  <img align="middle" width=400 src="demo_images/e_result_prediction_covSum.jpg" \>
  <img align="middle" width=400 src="demo_images/e_result_performance_covSum.jpg" \>
</p>
