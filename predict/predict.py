#!/usr/bin/env python3

import os
import shutil
import sys
import logging
import warnings
import argparse
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

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
    format='[%(asctime)s] %(name)s | %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PyRosetta
pyrosetta.init('-mute all', silent=True)

class PredictionPipeline:
    
    def __init__(self, input_path: str, overwrite: bool = True, save_structure: bool = False):

        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        self.directory_name = self.input_path.name
        self.features_file = self.input_path / f"{self.directory_name}_features.csv"
        self.predictions_file = self.input_path / f"{self.directory_name}_predictions.csv"

        self.overwrite = overwrite
        if self.overwrite or not self.features_file.exists():
            with open(self.features_file, "w") as f:
                f.write("pdb_file\t" + '\t'.join(OUTPUT_FEATURES) + "\n")
        if self.overwrite or not self.predictions_file.exists():
            with open(self.predictions_file, "w") as f:
                f.write("pdb_file,prediction\n")

        # Create output directories
        self.npz_dir = self.input_path / "npz_files"
        self.fasta_dir = self.input_path / "fasta"
        self.pred_structure_dir = self.input_path / "pred_structure"
        logger.info(f"Creating output directories: {self.npz_dir}, {self.fasta_dir} and {self.pred_structure_dir}")
        self.npz_dir.mkdir(exist_ok=True)
        self.fasta_dir.mkdir(exist_ok=True)
        self.pred_structure_dir.mkdir(exist_ok=True)
        
        # Load model and scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.save_structure = save_structure
    
    def _load_model(self) -> Any:

        try:
            #model_path = MODEL_DIR / 'model_1'
            model_path = MODEL_DIR / 'model_0.keras'
            return models.load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_scaler(self) -> Any:

        try:
            return joblib.load(SCALER_PATH)
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def process_structure(self, pdb_file: Path) -> Optional[Dict[str, float]]:

        try:
            # Load and process structure
            pose = pyrosetta.pose_from_file(str(pdb_file)).split_by_chain(1)
            temp_pdb_path = self.pred_structure_dir / f"{pdb_file.stem}_firstChain.pdb"
            pose.dump_pdb(str(temp_pdb_path))
            
            # Create FASTA file
            seq = pose.sequence()
            header = f">{pdb_file.name}"
            fasta_path = self.fasta_dir / pdb_file.with_suffix(FASTA_EXTENSION).name
            logger.info(f"Creating FASTA file: {fasta_path}")
            with open(fasta_path, 'w') as fasta:
                fasta.write(f"{header}\n{seq}\n")
            
            npz_path = self.npz_dir / pdb_file.with_suffix(NPZ_EXTENSION).name
            logger.info(f"Creating NPZ file: {npz_path}")
            get_ensembled_predictions(str(fasta_path), str(npz_path))
            
            # Process structure features
            pdb_out = prep_input(str(temp_pdb_path))
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
            return None
    
    def predict(self, features: Dict[str, float]) -> Optional[float]:

        try:
            logger.info("Starting prediction...")
            X = np.array([[
                features[feat] for feat in OUTPUT_FEATURES
            ]])
            X = self.scaler.transform(X)
            return self.model.predict(X, batch_size=BATCH_SIZE)[0][0]
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def run(self) -> None:
        try:
            # Process each PDB file
            for pdb_file in self.input_path.glob(PDB_PATTERN):
                logger.info(f"Processing {pdb_file.name}")
                
                # Calculate features
                features = self.process_structure(pdb_file)
                
                if features is None:
                    logger.warning(f"Skipping {pdb_file.name} due to processing errors.")
                    continue
                
                # Save features
                feature_line = "\t".join([
                    str(pdb_file),
                    *[str(features[feat]) for feat in OUTPUT_FEATURES]
                ])
                with open(self.features_file, "a") as f:
                    f.write(f"{feature_line}\n")
                
                # Make and save prediction
                prediction = self.predict(features)
                if prediction is None:
                    logger.warning(f"Skipping prediction for {pdb_file.name} due to prediction errors.")
                    continue

                with open(self.predictions_file, "a") as f:
                    f.write(f"{pdb_file.name},{prediction}\n")
                
                logger.info(f"Prediction saved to {self.predictions_file}")

            logger.info("Pipeline finished.")
            logger.info(f"Feature file saved to: {self.features_file}")
            logger.info(f"Predictions file saved to: {self.predictions_file}")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            if not self.save_structure:
                if self.pred_structure_dir.exists():
                    shutil.rmtree(self.pred_structure_dir)
                    logger.info(f"Removed temporary structure directory: {self.pred_structure_dir}")

def main():

    try:
        parser = argparse.ArgumentParser(description="Predict denovolikeness of PDB structures.")
        parser.add_argument("pdb_directory", type=str, help="Directory containing PDB files.")
        parser.add_argument("--no-logging", dest="logging", action="store_false", help="Disable logging.")
        parser.add_argument("--save-structure", action="store_true", help="Save the PDB files of the first chain.")
        parser.add_argument("--no-overwrite", dest='overwrite', action="store_false", help="Prevent overwriting existing prediction files.")
        parser.set_defaults(overwrite=True)
        parser.set_defaults(logging=True)
        args = parser.parse_args()

        if not args.logging:
            logging.disable(logging.CRITICAL)

        pipeline = PredictionPipeline(args.pdb_directory, overwrite=args.overwrite, save_structure=args.save_structure)
        pipeline.run()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

