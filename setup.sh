#!/bin/bash

# Stop on error
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install conda."
    exit 1
fi

# Create the conda environment and install packages.
conda create --name denovolikeness -y python=3.11 \
    numpy pandas joblib scikit-learn matplotlib \
    pytorch imbalanced-learn tensorflow==2.16.1 cudnn cudatoolkit

# Install pyrosetta using pip
conda run -n denovolikeness pip install pyrosetta-installer

# Run the pyrosetta installer
conda run -n denovolikeness \
    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

echo "setup.sh finished successfully."
echo "To activate the new environment, run:"
echo "conda activate denovolikeness"
