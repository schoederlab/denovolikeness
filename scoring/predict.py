#!/usr/bin/env/python

import sys
import os
import warnings, logging
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from code.utils import get_ensembled_predictions
from code.utils import prep_input, split_feat, get_dist_acc
from code.functions import *

import pyrosetta;pyrosetta.init()
from pyrosetta import *
from pyrosetta.rosetta.core.energy_methods import *
# init()

# from tensorflow import keras
from tensorflow.keras import models
#print(tf.__version__)
import numpy as np
# import pandas as pd
from code.util import *
from code.visualisation import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import joblib


def main():
    
    path = sys.argv[1]
    directory = os.path.dirname(path)  # Returns "/home/user/documents"
    directory_name = os.path.basename(directory)  # Returns "documents"
    for file in os.listdir(path):
        
        if not file.endswith('.pdb'):
            continue;
           
        file_path = os.path.join(path, file) 
        pose = pyrosetta.pose_from_file(file_path).split_by_chain(1)

        pose.dump_pdb('ChainA.pdb')


        header = '>' + file_path
        seq = pose.sequence()

        print(f"Sequence from pose: {seq}")
        with open(f'{file_path[:-4]}.fa','w') as fasta:
            fasta.writelines(header+'\n')
            fasta.writelines(str(seq))


        get_ensembled_predictions(f'{file_path[:-4]}.fa')
        NPZFile = f'{file_path[:-4]}.npz'


        pdb_out = prep_input('ChainA.pdb')
        pdb_feat = pdb_out["feat"][None]


        # Target
        dist0 = split_feat(pdb_feat)['dist'][0]
        omega0 = split_feat(pdb_feat)['omega'][0]
        phi0 = split_feat(pdb_feat)['phi'][0]
        theta0 = split_feat(pdb_feat)['theta'][0]

        # Prediction
        data = np.load(NPZFile)

        dist = data['dist']
        omega = data['omega']
        phi = data['phi']
        theta = data['theta']

        # Scores
        p_struc = P_struc_given_seq(dist,dist0,omega,omega0,theta,theta0,phi,phi0)
        p_con = P_contact_given_seq(dist,dist0,omega,omega0,theta,theta0,phi,phi0)
        dist_sc = distance_score(dist,dist0)
        con_sc = d_close(dist,dist0,angst=8)

        pred = np.concatenate([theta, phi, dist, omega],-1)
        area_sc = get_dist_acc(pred,pdb_feat)[0]

        per_con = perc_contact(dist0)
        per_no_con = perc_no_contact(dist0)


        coe = pyrosetta.rosetta.core.energy_methods.ContactOrderEnergy()
        contact_order = coe.calculate_contact_order(pose)

        # out
        output = f"{file_path}\t{p_struc}\t{p_con}\t{dist_sc}\t{con_sc}\t{area_sc}\t{per_con}\t{per_no_con}\t{contact_order}\n"

        with open(f'{directory_name}_features.csv', "a") as f:
            f.write(output.format())

        X = np.array([[p_struc,p_con,area_sc,per_con,per_no_con,contact_order]])
        scaler = joblib.load("./code/scaler.pkl")
        X = scaler.transform(X)

        model = models.load_model('./models/model_1')
        pred = model.predict(X, batch_size = BATCH_SIZE)

        with open(f'{directory_name}_predictions.csv','a') as f:
            f.write(f"{file},{pred[0][0]}\n")
            print(f'prediction saved to {directory_name}_predictions.csv')
        # print(pred[0][0])

if __name__ == "__main__":
    main()
