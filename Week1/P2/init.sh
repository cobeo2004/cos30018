# Configure conda for deprecated version (python 3.6)
conda create -n cos30018-env-w1-p2
conda activate cos30018-env-w1-p2
conda config --env --set subdir osx-64
conda install python=3.6
pip install -U -r requirements.txt
