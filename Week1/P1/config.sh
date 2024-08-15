# Configure conda environment for running P1
# Author: Simon Nguyen
# Date: 2024-02-15

# Generate conda environment with the name of cos30018-env-w1-p1
# using python 3.11 (You can change this to any name after -n flag you want)
conda create -n cos30018-env-w1-p1 python=3.11

# Activate the environment
conda activate cos30018-env-w1-p1

# Install packages from requirements.txt
pip install -U -r requirements.txt
