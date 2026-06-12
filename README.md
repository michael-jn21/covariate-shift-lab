# Covariate Shift Lab

A Python simulation framework for studying linear regression under covariate shift.

## Problem

We consider a transfer learning setting where source and target covariates follow different distributions,

\[
X_S \sim \mathcal N(0, \Sigma_S),
\qquad
X_T \sim \mathcal N(0, \Sigma_T),
\]

while the conditional model remains invariant:

\[
Y = X^\top \beta + \varepsilon.
\]

## Goal

The project compares different least-squares estimators under covariance mismatch:

- source-only estimator
- target-only estimator
- pooled estimator
- weighted pooled estimator

## Main Experiments

- spectral mismatch between source and target covariance
- rotated covariance matrices in dimension two
- target excess risk as a function of source and target sample size

## Tech Stack

Python, NumPy, SciPy, pandas, scikit-learn, matplotlib, pytest, Ruff.

## Status

Work in progress.