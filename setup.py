"""setup.py file for nn_uncertainty_estimation."""

from setuptools import setup  # type: ignore

setup(
    name="nn_uncertainty_estimation",
    version="0.0.2",
    author="ae137",
    author_email="a_e_mailings@posteo.de",
    packages=["nn_uncertainty_estimation"],
    url="https://github.com/ae137/nn_uncertainty_estimation",
    license="LICENSE",
    description="A collection of custom additions to Tensorflow / Keras",
    long_description=open("README.md").read(),
    install_requires=[
        "tensorflow==2.11.1",
    ],
    python_requires=">=3.8",
)
