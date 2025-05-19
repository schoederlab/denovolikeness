#!/usr/bin/env python3
"""
Main script for protein structure prediction and analysis.

This script processes PDB files to predict and analyze protein structures,
calculating various structural features and making predictions about their
characteristics.
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pyrosetta
import tensorflow as tf
from tensorflow.keras import models
import joblib

from code.utils import (
    get_ensembled_predictions,
    prep_input,
    split_feat,
    get_dist_acc
)
from code.functions import (
    P_struc_given_seq,
    P_contact_given_seq,
    distance_score,
    d_close,
    perc_contact,
    perc_no_contact
)
from code.config import (
    BATCH_SIZE,
    MODEL_DIR,
    SCALER_PATH,
    PDB_PATTERN,
    FASTA_EXTENSION,
    NPZ_EXTENSION,
    OUTPUT_FEATURES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PyRosetta
pyrosetta.init(silent=True)

class PredictionPipeline:
    """Main class for protein structure prediction pipeline."""
    
    def __init__(self, input_path: str):
        """Initialize the prediction pipeline.
        
        Args:
            input_path: Path to directory containing PDB files
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        self.directory_name = self.input_path.name
        self.features_file = Path(f"{self.directory_name}_features.csv")
        self.predictions_file = Path(f"{self.directory_name}_predictions.csv")
        
        # Load model and scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler()
    
    def _load_model(self) -> Any:
        """Load the trained model using Keras.
        
        Returns:
            Loaded Keras model
        """
        try:
            model_path = MODEL_DIR / 'model_1'
            return models.load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_scaler(self) -> Any:
        """Load the feature scaler.
        
        Returns:
            Loaded scaler object
        """
        try:
            return joblib.load(SCALER_PATH)
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def process_structure(self, pdb_file: Path) -> Dict[str, float]:
        """Process a single PDB structure.
        
        Args:
            pdb_file: Path to PDB file
        
        Returns:
            Dictionary of calculated features
        """
        try:
            # Load and process structure
            pose = pyrosetta.pose_from_file(str(pdb_file)).split_by_chain(1)
            pose.dump_pdb('ChainA.pdb')
            
            # Create FASTA file
            seq = pose.sequence()
            header = f">{pdb_file}"
            fasta_path = pdb_file.with_suffix(FASTA_EXTENSION)
            with open(fasta_path, 'w') as fasta:
                fasta.write(f"{header}\n{seq}\n")
            
            # Get predictions
            get_ensembled_predictions(str(fasta_path))
            npz_path = pdb_file.with_suffix(NPZ_EXTENSION)
            
            # Process structure features
            pdb_out = prep_input('ChainA.pdb')
            pdb_feat = pdb_out["feat"][None]
            
            # Extract features
            features = split_feat(pdb_feat)
            target = {
                'dist': features['dist'][0],
                'omega': features['omega'][0],
                'phi': features['phi'][0],
                'theta': features['theta'][0]
            }
            
            # Load predictions
            pred_data = np.load(npz_path)
            pred = {
                'dist': pred_data['dist'],
                'omega': pred_data['omega'],
                'phi': pred_data['phi'],
                'theta': pred_data['theta']
            }
            
            # Calculate scores
            scores = {
                '-logP_Struc': P_struc_given_seq(
                    pred['dist'], target['dist'],
                    pred['omega'], target['omega'],
                    pred['theta'], target['theta'],
                    pred['phi'], target['phi']
                ),
                '-logP_Contact': P_contact_given_seq(
                    pred['dist'], target['dist'],
                    pred['omega'], target['omega'],
                    pred['theta'], target['theta'],
                    pred['phi'], target['phi']
                ),
                'area_sc': get_dist_acc(
                    np.concatenate([
                        pred['theta'],
                        pred['phi'],
                        pred['dist'],
                        pred['omega']
                    ], -1),
                    pdb_feat
                )[0],
                'per_con': perc_contact(target['dist']),
                'per_no_con': perc_no_contact(target['dist']),
                'contact_order': pyrosetta.rosetta.core.energy_methods.ContactOrderEnergy().calculate_contact_order(pose)
            }
            
            return scores
            
        except Exception as e:
            logger.error(f"Error processing {pdb_file}: {str(e)}")
            raise
    
    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction based on calculated features.
        
        Args:
            features: Dictionary of calculated features
        
        Returns:
            Prediction score
        """
        try:
            X = np.array([[
                features[feat] for feat in OUTPUT_FEATURES
            ]])
            X = self.scaler.transform(X)
            return self.model.predict(X, batch_size=BATCH_SIZE)[0][0]
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def run(self) -> None:
        """Run the prediction pipeline on all PDB files."""
        try:
            # Process each PDB file
            for pdb_file in self.input_path.glob(PDB_PATTERN):
                logger.info(f"Processing {pdb_file.name}")
                
                # Calculate features
                features = self.process_structure(pdb_file)
                
                # Save features
                feature_line = "\t".join([
                    str(pdb_file),
                    *[str(features[feat]) for feat in OUTPUT_FEATURES]
                ])
                with open(self.features_file, "a") as f:
                    f.write(f"{feature_line}\n")
                
                # Make and save prediction
                prediction = self.predict(features)
                with open(self.predictions_file, "a") as f:
                    f.write(f"{pdb_file.name},{prediction}\n")
                
                logger.info(f"Prediction saved to {self.predictions_file}")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the prediction pipeline."""
    try:
        if len(sys.argv) != 2:
            print("Usage: predict.py <pdb_directory>")
            sys.exit(1)
        
        pipeline = PredictionPipeline(sys.argv[1])
        pipeline.run()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

