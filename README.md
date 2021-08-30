# Neural network uncertainty estimation

## Content of package

This package contains a collection of additions to Tensorflow.

Currently, it includes

* `losses.py`:
    * Implementation of Negative Log Likelihood loss for regression tasks

* `metrics.py`:
    * Extension of MSE suitable for a networks that predicts means and variances

# Building and installing the package

Run

 1. `pip install -r requirements.txt` for installing the build requirements
 2. `python -m build` for building the package
 3. `pip install dist/nn_uncertainty_estimation-0.0.1-py3-none-any.whl`

`TODO`: Include step for running the tests automatically.
