# load libraries
import numpy as np
import string, sys, getopt

DB_DIR = "/home/iwe18/Documents/joh/trDesign/02-GD" # location of databases

# ivan's natural AA composition
AA_COMP = np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                    0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                    0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                    0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271])

# David Juergens' optimized AA reference weights
# /home/norn/DL/200701_ref_weight_optimization/nelder_mead/scripts/nm_filtered/params_140
AA_REF = np.array([-1.31161863, -0.44993051,  0.06198913, -0.81825899,  2.63941964,
                    0.44087343, -0.93833546, -0.7374156 ,  1.54108622, -0.92757075,
                   -1.70878817, -0.9461753 ,  1.77794612,  0.2156388 ,  0.3293717 ,
                   -1.012154  , -0.60176806,  2.99381739,  0.84557686, -1.02749264])

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
  # ["ARND"] -> [[0,1,2,3]]
  x = np.array(x);
  if x.ndim == 0: x = x[None]
  return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

def parse_PDB(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''
  xyz,seq,min_resn,max_resn = {},{},np.inf,-np.inf
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): resa,resn = resn[-1],int(resn[:-1])-1
        else: resa,resn = "",int(resn)-1
        if resn < min_resn: min_resn = resn
        if resn > max_resn: max_resn = resn
        if resn not in xyz: xyz[resn] = {}
        if resa not in xyz[resn]: xyz[resn][resa] = {}
        if resn not in seq: seq[resn] = {}
        if resa not in seq[resn]: seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  for resn in range(min_resn,max_resn+1):
    if resn in seq:
      for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
    else: seq_.append(20)
    if resn in xyz:
      for k in sorted(xyz[resn]):
        for atom in atoms:
          if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
          else: xyz_.append(np.full(3,np.nan))
    else:
      for atom in atoms: xyz_.append(np.full(3,np.nan))
  return np.array(xyz_).reshape(-1,len(atoms),3), np.array(seq_)

def extend(a,b,c, L,A,D):
  '''
  input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
  output: 4th coord
  '''
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  bc = N(b-c)
  n = N(np.cross(b-a, bc))
  m = [bc,np.cross(n,bc),n]
  d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
  return c + sum([m*d for m,d in zip(m,d)])

def to_len(a,b):
  '''given coordinates a-b, return length or distance'''
  return np.sqrt(np.sum(np.square(a-b),axis=-1))

def to_len_pw(a,b=None):
  '''given coordinates a-b return pairwise distance matrix'''
  a_norm = np.square(a).sum(-1)
  if b is None: b,b_norm = a,a_norm
  else: b_norm = np.square(b).sum(-1)
  return np.sqrt(np.abs(a_norm.reshape(-1,1) + b_norm - 2*(a@b.T)))

def to_ang(a,b,c):
  '''given coordinates a-b-c, return angle'''
  D = lambda x,y: np.sum(x*y,axis=-1)
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  return np.arccos(D(N(b-a),N(b-c)))

def to_dih(a,b,c,d):
  '''given coordinates a-b-c-d, return dihedral'''
  D = lambda x,y: np.sum(x*y,axis=-1)
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  bc = N(b-c)
  n1 = np.cross(N(a-b),bc)
  n2 = np.cross(bc,N(c-d))
  return np.arctan2(D(np.cross(n1,bc),n2),D(n1,n2))

def prep_input(pdb, chain=None, mask_gaps=False):
  '''Parse PDB file and return features compatible with TrRosetta'''
  ncac, seq = parse_PDB(pdb,["N","CA","C"], chain=chain)

  # mask gap regions
  if mask_gaps:
    mask = seq != 20
    ncac, seq = ncac[mask], seq[mask]

  N,CA,C = ncac[:,0], ncac[:,1], ncac[:,2]
  CB = extend(C, N, CA, 1.522, 1.927, -2.143)

  dist_ref  = to_len(CB[:,None], CB[None,:])
  omega_ref = to_dih(CA[:,None], CB[:,None], CB[None,:], CA[None,:])
  theta_ref = to_dih( N[:,None], CA[:,None], CB[:,None], CB[None,:])
  phi_ref   = to_ang(CA[:,None], CB[:,None], CB[None,:])

  def mtx2bins(x_ref, start, end, nbins, mask):
    bins = np.linspace(start, end, nbins)
    x_true = np.digitize(x_ref, bins).astype(np.uint8)
    x_true[mask] = 0
    return np.eye(nbins+1)[x_true][...,:-1]

  p_dist  = mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
  p_omega = mtx2bins(omega_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
  p_theta = mtx2bins(theta_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
  p_phi   = mtx2bins(phi_ref,      0.0, np.pi, 13, mask=(p_dist[...,0]==1))
  feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega],-1)
  return {"seq":N_to_AA(seq), "feat":feat, "dist_ref":dist_ref}

def split_feat(feat):
  out = {}
  for k,i,j in [["theta",0,25],["phi",25,38],["dist",38,75],["omega",75,100]]:
    out[k] = feat[...,i:j]
  return out

def pairwise_id(x):
  '''get pairwise sequence identity'''
  x = np.array(x)
  return (x[:,None] == x[None,:]).mean(-1)

def arr2str(x, d=3):
  return np.array2string(x,formatter={'float_kind':lambda x: f"%.{d}f" % x}).replace("\n","").replace(" ",",")

#####################################################################
# Working with multiple sequence alignments
#####################################################################

def parse_fasta(filename, a3m=False):
  '''function to parse fasta file'''
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if len(line) > 0:
      if line[0] == ">":
        header.append(line[1:])
        sequence.append([])
      else:
        if a3m: line = line.translate(rm_lc)
        else: line = line.upper()
        sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return header, sequence

def mk_msa(seqs):
  '''one hot encode msa'''
  alphabet = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alphabet)

  alpha = np.array(alphabet, dtype='|S1').view(np.uint8)
  msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
  for n in range(states):
    msa[msa == alpha[n]] = n
  msa[msa > states] = states-1

  return np.eye(states)[msa]

############ Functions added by Johannes Klier
def NeighbourWeight(distance,low_bound=7.5,up_bound=10):
    if distance <= low_bound:
        return 1
    if distance >= up_bound:
        return 0
    return 0.5*(np.cos((distance-low_bound)/(up_bound-low_bound)*np.pi)+1)

def WeightDistance(area):
    area = np.array(area,dtype=float)
    for i in range(0,area.shape[-1]):
        area[...,i]= area[...,i]*NeighbourWeight((i*0.5)+2)
    return area
###########


def get_dist_acc(pred, true, true_mask=None,sep=5,eps=1e-8):
  ########### added by Johannes Klier  
  true[...,39:56]= WeightDistance(true[...,39:56])
  ########### 
  
  ## compute accuracy of CB features ##
  pred,true = [x[...,39:56].sum(-1) for x in[pred,true]]
  if true_mask is not None:
    mask = true_mask[:,:,None] * true_mask[:,None,:]
  else: mask = np.ones_like(pred)
  i,j = np.triu_indices(pred.shape[-1],k=sep)
  P,T,M = pred[...,i,j], true[...,i,j], mask[...,i,j]
  ## give equal weighting to positive and negative predictions
  pos = (T*P*M).sum(-1)/((M*T).sum(-1)+eps)
  neg = ((1-T)*(1-P)*M).sum(-1)/((M*(1-T)).sum(-1)+eps)
  return 2.0*(pos*neg)/(pos+neg+eps)

def inv_cov(Y):
  '''given MSA, return contacts'''

  N,L = Y.shape
  K = Y.max()+1
  Y = np.eye(K)[Y]

  # flatten msa (N,L,A) -> (N,L*A)
  Y_flat = Y.reshape(N,-1)

  # compute covariance matrix (L*A,L*A)
  c = np.cov(Y_flat.T)
  # compute shrinkage (l2 regularization)
  shrink = 4.5/np.sqrt(N) * np.eye(c.shape[0])
  # take the inverse to solve for w
  ic = np.linalg.inv(c + shrink)
  # (L,A,L,A)
  ic = ic.reshape(L,K,L,K)

  # take l2norm to reduce (L,A,L,A) to (L,L) matrix
  ic_norm = np.sqrt(np.square(ic).sum((1,3)))
  np.fill_diagonal(ic_norm,0)

  #Average product correction (aka remove largest eigenvector)
  ap = ic_norm.sum(0)
  apc = ic_norm - (ap[:,None]*ap[None,:])/ap.sum()
  np.fill_diagonal(apc,0.0)
  return apc

def to_dict(label, var_list):
  return dict(zip(label,var_list))

def to_list(label, var_dict, default=None):
  return [var_dict.get(k, default) for k in label]

# class for parsing arguments
class parse_args:
  def __init__(self):
    self.long,self.short = [],[]
    self.info,self.help = [],[]

  def txt(self,help):
    self.help.append(["txt",help])

  def add(self, arg, default, type, help=None):
    self.long.append(arg[0])
    key = arg[0].replace("=","")
    self.info.append({"key":key, "type":type,
                      "value":default, "arg":[f"--{key}"]})
    if len(arg) == 2:
      self.short.append(arg[1])
      s_key = arg[1].replace(":","")
      self.info[-1]["arg"].append(f"-{s_key}")
    if help is not None:
      self.help.append(["opt",[arg,help]])

  def parse(self,argv):
    for opt, arg in getopt.getopt(argv,"".join(self.short),self.long)[0]:
      for x in self.info:
        if opt in x["arg"]:
          if x["type"] is None: x["value"] = (x["value"] == False)
          else: x["value"] = x["type"](arg)

    opts = {x["key"]:x["value"] for x in self.info}
    # print(str(opts).replace(" ",""))
    return dict2obj(opts)

  def usage(self, err):
    for type,info in self.help:
      # if type == "txt": print(info)
      if type == "opt":
        arg, helps = info
        help = helps[0]
        if len(arg) == 1: print("--%-15s : %s"     % (arg[0],help))
        if len(arg) == 2: print("--%-10s -%-3s : %s" % (arg[0],arg[1].replace(":",""),help))
        for help in helps[1:]: print("%19s %s" % ("",help))
    print(f"< {err} >")
    print(" "+"-"*(len(err)+2))
    print("        \   ^__^               ")
    print("         \  (oo)\_______       ")
    print("            (__)\       )\/\   ")
    print("                ||----w |      ")
    print("                ||     ||      ")
    sys.exit()

class dict2obj():
  def __init__(self, dictionary):
    for key in dictionary:
      setattr(self, key, dictionary[key])




######################################################################################################################
#
# Functions added from tr_rossetta_pytorch
#
######################################################################################################################

import torch
import torch.nn.functional as F
from torch import nn

import tarfile
from pathlib import Path

# from tr_rosetta_pytorch import trRosettaNetwork
# FROM tr_rosetta_pytorch
import torch
from torch import nn, einsum
import torch.nn.functional as F

def elu():
    return nn.ELU(inplace=True)

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)

class trRosettaNetwork(nn.Module):
    def __init__(self, filters=64, kernel=3, num_layers=61):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers

        self.first_block = nn.Sequential(
            conv2d(442 + 2 * 42, filters, 1),
            instance_norm(filters),
            elu()
        )

        # stack of residual blocks with dilations
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters),
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations])

        self.activate = elu()

        # conv to anglegrams and distograms
        self.to_prob_theta = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
        self.to_prob_phi = nn.Sequential(conv2d(filters, 13, 1), nn.Softmax(dim=1))
        self.to_distance = nn.Sequential(conv2d(filters, 37, 1), nn.Softmax(dim=1))
        self.to_prob_bb = nn.Sequential(conv2d(filters, 3, 1), nn.Softmax(dim=1))
        self.to_prob_omega = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
 
    def forward(self, x):
        x = self.first_block(x)

        for layer in self.layers:
            x = self.activate(x + layer(x))
        
        prob_theta = self.to_prob_theta(x)      # anglegrams for theta
        prob_phi = self.to_prob_phi(x)          # anglegrams for phi

        x = 0.5 * (x + x.permute((0,1,3,2)))    # symmetrize

        prob_distance = self.to_distance(x)     # distograms
        # prob_bb = self.to_prob_bb(x)            # beta-strand pairings (not used)
        prob_omega = self.to_prob_omega(x)      # anglegrams for omega

        return prob_theta, prob_phi, prob_distance, prob_omega
    
# ===============================================================================








def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

# preprocessing fn

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    seqs = [line.strip().translate(table) for line in open(filename, 'r') if line[0] != '>']
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)

    # convert letters into numbers
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    return msa

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = w.sum()
    f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
    h_i = (-f_i * torch.log(f_i)).sum(dim=1)
    return torch.cat((f_i, h_i[:, None]), dim=1)

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w

# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 4.5):
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns).to(device) * penalty / torch.sqrt(weights.sum())

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc).to(device))
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc).to(device))
    return torch.cat((features, contacts[:, :, None]), dim=2)

def preprocess(msa_file, wmin=0.8, ns=21):
    a3m = torch.from_numpy(parse_a3m(msa_file)).long()
    nrow, ncol = a3m.shape

    msa1hot = F.one_hot(a3m, ns).float().to(d())
    w = reweight(msa1hot, wmin).float().to(d())

    # 1d sequence

    f1d_seq = msa1hot[0, :, :20].float()
    f1d_pssm = msa2pssm(msa1hot, w)

    f1d = torch.cat((f1d_seq, f1d_pssm), dim=1)
    f1d = f1d[None, :, :].reshape((1, ncol, 42))

    # 2d sequence

    f2d_dca = fast_dca(msa1hot, w) if nrow > 1 else torch.zeros((ncol, ncol, 442)).float().to(d())
    f2d_dca = f2d_dca[None, :, :, :]

    f2d = torch.cat((
        f1d[:, :, None, :].repeat(1, 1, ncol, 1), 
        f1d[:, None, :, :].repeat(1, ncol, 1, 1),
        f2d_dca
    ), dim=-1)

    f2d = f2d.view(1, ncol, ncol, 442 + 2*42)
    return f2d.permute((0, 3, 2, 1))










# from tr_rosetta_pytorch.tr_rosetta_pytorch import trRosettaNetwork
# from tr_rosetta_pytorch.utils import preprocess, d

# paths

CURRENT_PATH = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = CURRENT_PATH / 'models'
MODEL_PATH =  DEFAULT_MODEL_PATH / 'models.tar.gz'
MODEL_FILES = [*Path(DEFAULT_MODEL_PATH).glob('*.pt')]

# extract model files if not extracted

if len(MODEL_FILES) == 0:
    print(MODEL_PATH)
    tar = tarfile.open(str(MODEL_PATH))
    tar.extractall(DEFAULT_MODEL_PATH)
    tar.close()

# prediction function

@torch.no_grad()
def get_ensembled_predictions(input_file, output_file=None, model_dir=DEFAULT_MODEL_PATH):
    net = trRosettaNetwork()
    i = preprocess(input_file)

    if output_file is None:
        input_path = Path(input_file)
        output_file = f'{input_path.parents[0] / input_path.stem}.npz'

    outputs = []
    model_files = [*Path(model_dir).glob('*.pt')]

    if len(model_files) == 0:
        raise 'No model files can be found'

    for model_file in model_files:
        net.load_state_dict(torch.load(model_file, map_location=torch.device(d())))
        net.to(d()).eval()
        output = net(i)
        outputs.append(output)

    averaged_outputs = [torch.stack(model_output).mean(dim=0).cpu().numpy().squeeze(0).transpose(1,2,0) for model_output in zip(*outputs)]
    # prob_theta, prob_phi, prob_distance, prob_omega
    output_dict = dict(zip(['theta', 'phi', 'dist', 'omega'], averaged_outputs))
    np.savez_compressed(output_file, **output_dict)
    print(f'predictions for {input_file} saved to {output_file}')

# def predict():
#     fire.Fire(get_ensembled_predictions)
