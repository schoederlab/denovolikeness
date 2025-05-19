# Protein Structure Prediction and Analysis

This project provides tools for protein structure prediction and analysis, implementing the methods described in Norn et al. 2021. It calculates various structural scores and probabilities based on predicted and actual protein structures.

## Features

- Protein structure prediction using deep learning models
- Calculation of structural scores:
  - -logP(structure|sequence)
  - -logP(contacts|sequence)
  - Distance-based scores
  - Contact analysis
- Support for PDB file processing
- Integration with PyRosetta for structural analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate denovolike
```

## Environment Setup

The project requires the following key dependencies:
- Python 3.9
- TensorFlow 2.14.0
- PyRosetta 2023.45
- NumPy
- Pandas
- scikit-learn
- matplotlib

The complete environment can be set up using the provided `environment.yml` file.

## Usage

### Basic Usage

To run predictions on a directory of PDB files:

```bash
python predict.py <pdb_directory>
```

This will:
1. Process each PDB file in the directory
2. Calculate structural features
3. Generate predictions
4. Save results to CSV files:
   - `<directory_name>_features.csv`: Contains calculated structural features
   - `<directory_name>_predictions.csv`: Contains final predictions

### Output Files

1. Features File (`<directory_name>_features.csv`):
   - Contains structural features for each protein
   - Includes: -logP_Struc, -logP_Contact, area_sc, per_con, per_no_con, contact_order

2. Predictions File (`<directory_name>_predictions.csv`):
   - Contains final predictions for each protein
   - Format: `<pdb_file>,<prediction_score>`

## Project Structure

```
denovolikeness/
├── predict/
│   ├── code/
│   │   ├── functions.py    # Core calculation functions
│   │   ├── utils.py        # Utility functions
│   │   └── config.py       # Configuration settings
│   └── predict.py          # Main prediction script
├── models/                 # Trained model directory
└── environment.yml         # Conda environment specification
```

## Key Functions

### Structure Analysis
- `P_struc_given_seq()`: Calculates -logP(structure|sequence)
- `P_contact_given_seq()`: Calculates -logP(contacts|sequence)
- `distance_score()`: Calculates distance-based score
- `d_close()`: Calculates close contact score

### Contact Analysis
- `perc_contact()`: Calculates percentage of residues in contact
- `perc_no_contact()`: Calculates percentage of residues not in contact
- `distance_distribution()`: Calculates distance distribution

## Requirements

- CUDA-capable GPU (recommended)
- Sufficient disk space for model files
- Python 3.9 or compatible version

## Troubleshooting

Common issues and solutions:

1. Model Loading Error:
   - Ensure TensorFlow version matches the model version
   - Check model file paths in config.py

2. PyRosetta Initialization:
   - Verify PyRosetta installation
   - Check database paths

3. Memory Issues:
   - Reduce batch size in config.py
   - Process files in smaller batches

## Citation

If you use this software in your research, please cite:
```
Norn, C., et al. (2021). [Paper Title]. [Journal Name].
```

## License

[Specify your license here]

## Contact

[Your contact information] 