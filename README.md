# Denovolikeness
A tool for assessing the de novo likeness of protein structures through structural scoring and classification.

## Project Structure

The project is organized into several main components:

### 1. Prediction (`/predict`)
Machine learning model for classifying proteins based on their structural scores.

### 2. Training (`/training`)
Code for training the classification model.

### 3. Data (`/data`)
Contains training data, feature calculations, and prediction results:
- `training_data.csv`: Training dataset
- `features_screen/`: Calculated features for different organisms
- `predictions_screen/`: Prediction results for screened organisms
- `sequence_info/`: Organism sequence information and metadata

### 4. Figures (`/figures`)
Analysis notebooks and visualization code:
- `figure1/`: Feature analysis and visualization (Fig1.ipynb)
- `figure3/`: Cross-organism screening analysis and plots (screening.ipynb, esm.ipynb)

### 5. Supplements (`/supplements`)
Additional analysis and experimental work:
- `pda/`: Protein Data Archive (PDA) analysis including:
  - `pda_jan25_analysis.ipynb`: Comprehensive PDA dataset analysis with PDB processing, Rosetta scoring, and BLAST alignment integration
  - `best_relax/`: Rosetta relaxation predictions
- `non_pred_features/`: Alternative feature analysis
- `other_models/`: Cross-validation and alternative model evaluations

### 6. Results (`/results`)
Output data from analyses including PDA predictions and scoring results

## Installation

### Prerequisites
- Conda or Miniconda
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JohannesKlier/denovolikeness
cd denovolikeness
```

2. Set up the conda environment:

**Option A: Using the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

**Option B: Using the environment file directly**
```bash
conda env create -f environment.yml
```

This will create a conda environment named `denovolikeness` with all necessary dependencies including:
- Python 3.11
- TensorFlow 2.16.1
- PyTorch
- PyRosetta
- Scientific computing libraries (numpy, pandas, scikit-learn, matplotlib, seaborn, scipy)
- Jupyter for notebook analysis

3. Activate the environment:
```bash
conda activate denovolikeness
```

### Verification

To verify the installation:
```bash
python -c "import tensorflow as tf; import pyrosetta; print('Installation successful!')"
```

## Classifying Proteins

### Method 1: Process PDB Files (Full Pipeline)

To run predictions on a directory of PDB files:

```bash
cd predict
python predict.py <pdb_directory>
```

This will:
1. Process each PDB file in the directory
2. Calculate structural features using trRosetta
3. Generate predictions using the trained model
4. Save results to CSV files:
   - `<directory_name>_features.csv`: Contains calculated structural features
   - `<directory_name>_predictions.csv`: Contains final predictions

### Method 2: Predict from Pre-computed Features

If you already have computed features (e.g., from `data/features_screen/`), use the Jupyter notebook for batch predictions:

```bash
jupyter notebook predict_homo_denovolikeness.ipynb
```

This notebook:
1. Loads pre-computed features from CSV files
2. Applies the trained model and scaler
3. Generates predictions for all entries
4. Provides comprehensive visualization and analysis:
   - Distribution plots and statistical summaries
   - Top/bottom ranking proteins by denovolikeness
   - Feature correlation analysis
   - Threshold-based classification (e.g., % above 0.5)
5. Exports results with predictions included

This method is much faster for large-scale screening when features are already available.

### Command-line options

The prediction script `predict.py` accepts the following command-line arguments:

-   `pdb_directory`: (Required) The path to the directory containing the PDB files you want to analyze.
-   `--no-logging`: (Optional) Disable logging to the console.
-   `--save-structure`: (Optional) Save the PDB files of the first chain of each processed protein.
-   `--no-overwrite`: (Optional) Prevent the script from overwriting existing prediction and feature files.

### Output Files

1. Features File (`<directory_name>_features.csv`):
   - Contains structural features for each protein
   - Includes: -logP_Struc, -logP_Contact, area_sc, per_con, per_no_con, contact_order

2. Predictions File (`<directory_name>_predictions.csv`):
   - Contains final predictions for each protein
   - Format: `UniProtID,scores`

## Analysis and Visualization

The project includes several Jupyter notebooks for comprehensive analysis:

### Main Analysis Notebooks

#### Batch Prediction (`predict_homo_denovolikeness.ipynb`)
Located in the root directory, this notebook demonstrates:
- Loading pre-computed features from CSV files
- Batch prediction using the trained model
- Statistical analysis and visualization of results
- Threshold-based analysis (e.g., percentage above 0.5)
- Feature correlation with denovolikeness scores

#### Cross-Organism Screening (`/figures/figure3/screening.ipynb`)
Complete analysis pipeline including:
- Feature calculation across multiple organisms (human, mouse, E. coli, etc.)
- Model predictions and scoring
- Phylogenetic analysis with dendrograms
- Statistical visualization (bar plots, violin plots, swarm plots)
- Cross-species comparison of denovolikeness

#### Feature Analysis (`/figures/figure1/Fig1.ipynb`)
- Analysis of structural features used in the classification model
- Feature distribution and importance visualization

### Supplementary Analysis

#### PDA Analysis (`/supplements/pda/pda_jan25_analysis.ipynb`)
Comprehensive analysis of Protein Data Archive (PDA) data:
- PDA dataset parsing and filtering
- PDB structure downloading and processing
- Rosetta relaxation and energy scoring
- BLAST alignment integration for cluster representatives
- Multi-chain protein handling and prim_key generation
- Data pipeline diagnostics and quality control
- Merge operations between PDA metadata and Rosetta scores

#### Model Evaluation (`/supplements/other_models/CV_ALL.ipynb`)
- Cross-validation analysis
- Alternative model comparisons

#### Additional Features (`/supplements/non_pred_features/features.ipynb`)
- Exploration of alternative structural features


## Citation

If you use this software in your research, please cite:
```
Klier, J., et al. (2025).
```


## Contact

klier@izbi.uni-leipzig.de 