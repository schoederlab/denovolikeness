"""
Core functions for protein structure prediction and analysis.

This module provides functions for calculating various structural scores
and probabilities based on predicted and actual protein structures.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from .config import CONTACT_DISTANCE

def calculate_log_probabilities(pred: np.ndarray,
                              true: np.ndarray) -> np.ndarray:
    """Calculate log probabilities for structural features.
    
    Args:
        pred: Predicted probabilities
        true: True probabilities
    
    Returns:
        Log probabilities of the predictions at true coordinates
    """
    probs = np.sum(pred * true, axis=2)
    return np.log10(np.maximum(probs, 1e-10))  # Avoid log(0)

def P_struc_given_seq(dist, dist0, omega, omega0,
                     theta, theta0, phi, phi0):
    """Calculate -logP(structure|sequence) as defined in Norn et al. 2021."""
    
    # log-likelihoods
    d_prob = dist * dist0                     # networkpredictions at coordinates of given structure
    d_prob = np.sum(d_prob, axis=2)          # shrink array on last axis
    d_prob = np.log10(d_prob)                # log

    o_prob = omega * omega0                   # networkpredictions at coordinates of given structure
    o_prob = np.sum(o_prob, axis=2)          # shrink array on last axis
    o_prob = np.log10(o_prob)                # log

    t_prob = theta * theta0                   # networkpredictions at coordinates of given structure
    t_prob = np.sum(t_prob, axis=2)          # shrink array on last axis
    t_prob = np.log10(t_prob)                # log

    p_prob = phi * phi0                       # networkpredictions at coordinates of given structure
    p_prob = np.sum(p_prob, axis=2)          # shrink array on last axis
    p_prob = np.log10(p_prob)                # log

    L = d_prob.shape[1]                      # sequence lengths
    y0 = [d_prob, o_prob, t_prob, p_prob]
    
    return -(1/(4*L**2)) * np.sum(y0)        # sum all y0 in {distance,omega,theta,phi} and all coordinates

def get_contact_mask(dist0: np.ndarray,
                    contact_dist: float = CONTACT_DISTANCE) -> np.ndarray:
    """Generate contact mask for residue pairs within contact distance.
    
    Args:
        dist0: True distance probabilities
        contact_dist: Maximum distance to consider as contact
    
    Returns:
        Boolean mask indicating residue pairs in contact
    """
    d_max_index0 = dist0.argmax(axis=2)
    d_max_bin0 = (d_max_index0/2) + 2  # Convert to angstroms
    return np.where(d_max_bin0 <= contact_dist, 1, 0)

def P_contact_given_seq(dist, dist0, omega, omega0,
                       theta, theta0, phi, phi0):
    """Calculate -logP(contacts|sequence) as defined in Norn et al. 2021."""
    
    d_max_index0 = dist0.argmax(axis=2)      # index of distance with highest probability
    d_max_bin0 = (d_max_index0/2) + 2        # get values back to angstrom (0.5 A per bin, ranging from 2 to 20)
    m = np.where(d_max_bin0 <= 10, 1, 0)     # Limit to close contact (c_beta - c_beta < 10 Angstrom)
    
    # choose probability of feature at given position
    d_prob = np.sum(dist * dist0, axis=2)
    o_prob = np.sum(omega * omega0, axis=2)
    t_prob = np.sum(theta * theta0, axis=2)
    p_prob = np.sum(phi * phi0, axis=2)
    
    L = d_prob.shape[0]                      # sequence lengths
    
    # implementation of -logP(contacts|sequence)
    sum1 = 0
    sum2 = 0
    sum3 = 0

    # log-likelihoods
    y = [t_prob, p_prob]
    for y0 in y:
        for i in list(range(0, L)):
            for j in list(range(0, L)):
                if m[i][j] == 0:
                    continue
                if j == i:
                    continue
                sum1 += np.log10(y0[i][j])

    y = [d_prob, o_prob]
    for y0 in y:
        for i in list(range(0, L)):
            for j in list(range(i+1, L)):
                if m[i][j] == 0:
                    continue
                sum2 += np.log10(y0[i][j])

    # mask
    for i in list(range(0, L)):
        for j in list(range(0, L)):
            if i == j:
                continue
            if m[i][j] == 0:
                continue
            sum3 += m[i][j]
            
    return -((sum1 + sum2)/(3*sum3))

def distance_score(dist, dist0):
    """Calculate distance-based score."""
    L = dist0.shape[1]
    d_prob = dist0 * dist                     # networkpredictions at coordinates of given structure
    d_prob = np.sum(d_prob, axis=2)          # shrink array on last axis
    d_prob = np.log10(d_prob)                # log
    
    return -(np.sum(d_prob)/L**2)

def d_close(dist, dist0, angst=8):
    """Calculate close contact score."""
    L = dist.shape[1]
    my_bin = (angst*2) + 1
    contact = dist0[..., 1:my_bin] * dist[..., 1:my_bin]
    contact = np.sum(contact, axis=2)
    
    contact = np.where(contact == 0, 1, contact)  # mask zeros
    contact = np.log10(contact)                   # log
    
    return -np.sum(contact)/L**2

def distance_distribution(dist0):
    """Calculate distance distribution."""
    return np.sum(dist0, axis=(0, 1))

def perc_contact(dist0):
    """Calculate percentage of residues in contact."""
    normed_dist = distance_distribution(dist0)[1:] / sum(distance_distribution(dist0)[1:])
    # Contact = 2 - 10 Angstrom
    return sum(normed_dist[0:16])

def perc_no_contact(dist0):
    """Calculate percentage of residues not in contact."""
    normed_dist = distance_distribution(dist0) / sum(distance_distribution(dist0))
    # 2 - 10 Angstrom
    return normed_dist[0]

# # relative contact order
# def CO(s,cutoff_nm=.6):
#     res = {}
#     for chains in s:
#         for chain in chains:
#             for residue in chain:
#                 key =residue.get_id()[1]
#                 if key not in res:
#                     res[key] = list()
#                 res[key].extend([atom.get_vector() for atom in residue if "H" not in atom.get_name()])

#     N = 0
#     seq_distance_sum = 0
#     cutoff_2 = cutoff_nm*cutoff_nm
#     L=len(res)
#     for key1,val1 in res.items():
#         for key2,val2 in res.items():
#             seq_dist=abs(key2-key1)
#             if seq_dist > 0:
#                 L_i=len(res[key1])
#                 L_j=len(res[key2])
#                 for x in range(L_i):
#                     for y in range(L_j):
#                         d = 0.0
#                         d = sum((res[key1][x] - res[key2][y])) ** 2
#                         # print(d)
#                         if d < cutoff_2:
#                             seq_distance_sum+=2*seq_dist
#                             N+=2
#     if N==0.:
#         return 0  
#     return (1/(L*N))*seq_distance_sum
