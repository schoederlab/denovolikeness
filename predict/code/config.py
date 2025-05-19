"""
Configuration settings for protein structure prediction.

This module contains all configuration parameters used in the prediction pipeline.
"""

from pathlib import Path

# Model parameters
BATCH_SIZE = 1024

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
CODE_DIR = BASE_DIR / 'code'
SCALER_PATH = CODE_DIR / 'scaler.pkl'

# Structure parameters
CONTACT_DISTANCE = 10  # Angstroms

# Output settings
OUTPUT_FEATURES = [
    '-logP_Struc',
    '-logP_Contact',
    'area_sc',
    'per_con',
    'per_no_con',
    'contact_order'
]

# File patterns
PDB_PATTERN = "*.pdb"
FASTA_EXTENSION = ".fa"
NPZ_EXTENSION = ".npz"

# Visualization settings
STYLE = {
    'color': '0.2',
    'fontname': 'Arial',
    'fontsize': {
        'title': 35,
        'label': 25,
        'ticks': 20,
        'legend': 20
    },
    'linewidth': 2.5,
    'colors': ["#3d348b", "#7678ed", "#f7b801", "#f18701", "#f35b04"]
} 
