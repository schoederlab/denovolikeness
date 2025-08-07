conda create --name denovolikeness
conda activate denovolikeness
conda install python=3.11 numpy pandas joblib scikit-learn matplotlib pytorch imbalanced-learn tensorflow==2.16.1 cudnn cudatoolkit
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'