#!/usr/bin/python
import warnings, logging, os
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import parse_args, prep_input, split_feat, get_dist_acc
import sys
from functions import *

import pyrosetta;pyrosetta.init()
from pyrosetta import *
from pyrosetta.rosetta.core.energy_methods import *
init()

def main(argv):
    ag = parse_args()
    ag.txt("-------------------------------------------------------------------------------------")
    ag.txt("Scoring PDB features")
    ag.txt("-------------------------------------------------------------------------------------")
    ag.add(["pdb=",  "p:"], None,  str,   ["PDB to score"])
    ag.add(["npz=",  "n:"], None,  str,   ["Features from prediction"])
    ag.add(["out=",  "s:"], None,  str,   ["Scorefile"])
    ag.txt("-------------------------------------------------------------------------------------")
	
    o = ag.parse(argv)

    if o.pdb is None or o.npz is None:
        ag.usage(f"ERROR: --pdb={o.pdb} AND --npz={o.npz} must be defined")
        
    
    pdb_out = prep_input(o.pdb)
    pdb_feat = pdb_out["feat"][None]       
    
    # Target
    dist0 = split_feat(pdb_feat)['dist'][0]
    omega0 = split_feat(pdb_feat)['omega'][0]
    phi0 = split_feat(pdb_feat)['phi'][0]
    theta0 = split_feat(pdb_feat)['theta'][0]

    # Prediction
    data = np.load(o.npz)

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
    
    pose = pyrosetta.pose_from_file(o.pdb)
    coe = pyrosetta.rosetta.core.energy_methods.ContactOrderEnergy()
    contact_order = coe.calculate_contact_order(pose)    
    
    # out
    output = f"{o.pdb}\t{p_struc}\t{p_con}\t{dist_sc}\t{con_sc}\t{area_sc}\t{per_con}\t{per_no_con}\t{contact_order}\n"

    with open(o.out, "a") as f:
        f.write(output.format())


if __name__ == "__main__":
   main(sys.argv[1:])