# Denovolikeness
A tool for assessing the de novo likeness of protein structures through structural scoring and classification.

## Project Structure

The project is organized into two main components:

### 1. Prediction (`/predict`)
Machine learning model for classifying proteins based on their structural scores.

### 2. Training (`/training`)
Code for training of the model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JohannesKlier/denovolikeness
cd denovolikeness
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate denovolike
```

3. Install the Pytorch version of [trRossetta](https://github.com/lucidrains/tr-rosetta-pytorch)
```bash
pip install tr-rosetta-pytorch
```

## Classifying Proteins

To run predictions on a directory of PDB files:

```bash
cd predict
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
   - Includes: -logP_Struc, -logP_Contact, dist_acc, per_con, per_no_con, contact_order

2. Predictions File (`<directory_name>_predictions.csv`):
   - Contains final predictions for each protein
   - Format: `<pdb_file>,<prediction_score>`


## Citation

If you use this software in your research, please cite:
```
Klier, J., et al. (2025).
```


## Contact

klier@izbi.uni-leipzig.de 