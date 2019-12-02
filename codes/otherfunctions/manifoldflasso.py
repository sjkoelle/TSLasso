import argparse
import matplotlib
matplotlib.use('Agg')
import os
import datetime
import numpy as np
import dill as pickle
import random
import sys
from itertools import combinations_with_replacement
np.random.seed(0)
random.seed(0)
now = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
#workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
#sys.path.append(workingdirectory)
#print(workingdirectory + 'asdf')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--angles', help='file with angles')
parser.add_argument('--manisamkgradients', help = 'directory with code')
parser.add_argument('--xyz', help='xyz')
parser.add_argument('--pca_projected', help='pca projection')
parser.add_argument('--output_folder', help='where to save outputs')
parser.add_argument('--dim', help='slow mode dimension', type = int)
parser.add_argument('--dimnoise', help='tangent space dimension')
parser.add_argument('--nsel', help='number of points for manifold flasso')
parser.add_argument('--nreps', help='number of reps manifold flasso')
parser.add_argument('--epsilon', help='epsilon for embedding', type = np.float64)
parser.add_argument('--ncomponents', help='n components for embedding', type = int)
parser.add_argument('--atoms4', help='torsions to consider')
parser.add_argument('--ncores', help='number of cores')
parser.add_argument('--itermax', help='lasso iteration max')
parser.add_argument('--lambdamin', help='lambda min non-zero')
parser.add_argument('--nlambdas', help = 'number of lambdas including lmin and lmax but not including 0')
parser.add_argument('--lambdamax', help='lambda max')
parser.add_argument('--n', help = 'npoints',type = int)
parser.add_argument('--tol', help  = 'lasso convergence tolerance')
parser.add_argument('--nnebs', help  = 'number of neighbors of embedding', type = int)
parser.add_argument('--natoms', help  = 'number of atoms')
parser.add_argument('--angle_indices', help  = 'angle_indices')
parser.add_argument('--title', help  = 'title')
parser.add_argument('--projector', help = 'pca projection matrix')
args = parser.parse_args()
#print(args.angles)

n = args.n #number of data points to simulate
nsel = args.nsel #number of points to analyze with lasso
itermax = args.itermax #maximum iterations per lasso run
tol = args.tol
output_folder = args.output_folder
projected = args.pca_projected
pca_projector = args.projector
lambdamin=np.float16(args.lambdamin)
nlambdas = args.nlambdas
lambdamax=np.float16(args.lambdamax)
angles = args.angles
angle_indices = args.angle_indices
print(angle_indices)
xyz = args.xyz
qqq = args.atoms4
strs = qqq.replace('[','').split('],')
lists = [(s.replace(']','').split(',')) for s in strs]
atoms4 = np.asarray(lists, dtype = int)
projector = args.projector
#print(type(lambdamax), 'lambdamax')
n_neighbors = args.nnebs #number of neighbors in megaman
n_components = args.ncomponents #number of embedding dimensions (diffusion maps)
diffusion_time = args.epsilon #diffusion time controls gaussian kernel radius per gradients paper
dim = args.dim #slow manifold dimension
dimnoise = args.dimnoise #noise dimension
ncores = args.ncores #number of cores for parallel processing
print(args.natoms, type(args.natoms), 'natoms')
natoms = int(args.natoms)
title = args.title
code_directory = args.manisamkgradients
sys.path.append(code_directory)
#cwd = os.getcwd()
#os.chdir(code_directory)
print(code_directory)
from codes.experimentclasses.AtomicRegression import AtomicRegression
import scipy
from codes.otherfunctions.multirun import get_coeffs_reps
from codes.otherfunctions.multirun import get_grads_reps_pca2
from codes.otherfunctions.multiplot import plot_betas, plot_betas3reorder
from codes.geometer.RiemannianManifold import RiemannianManifold
#os.chdir(cwd)

class AtomicRegressionLoader(AtomicRegression):

    def __init__(self, angles,angle_indices, xyz,dim, dimnoise,itermax,tol,lambdas,n_neighbors, n_components, atoms4, ncores, diffusion_time):
	    self.angles = angles
	    self.xyz = xyz
	    self.n = n
	    self.natoms = natoms
	    self.d = 50
	    self.atoms3, self.d = self.get_atoms_3()
	    self.dim = dim
	    self.dimnoise = dimnoise
	    self.cores = ncores
	    ii = np.asarray([0, 0, 0, 0, 1, 6, 5, 6, 5, 4, 4, 3, 3, 2, 2])
	    jj = np.asarray([8, 9, 7, 1, 6, 14, 13, 5, 4, 12, 3, 11, 2, 10, 1])
	    self.ii = ii
	    self.jj = jj
	    AtomicRegression.__init__(self, dim, n, ii, jj, natoms, ncores, False)
	    self.atoms4 = atoms4
	    self.p = atoms4.shape[0]
	    self.angle_indices  = angle_indices

    def load_data(self):
        n = self.n
        d = self.d
        dim = self.dim
        natoms = self.natoms
        atoms3 = self.atoms3
        angle_indices = self.angle_indices

        filename_xyz = xyz
        filename_angle_indices = angle_indices
        print(filename_xyz[-4:])
        if filename_xyz[-4:] == '.mat':
        	data_xyz_loaded = scipy.io.loadmat(filename_xyz)
        if filename_xyz[-4:] == '.npz':
        	data_xyz_loaded = np.load(filename_xyz)
        print(filename_angle_indices)
        angle_indices = np.load(filename_angle_indices)
        positions = data_xyz_loaded['R'][angle_indices]
        self.positions = positions
        filename_angles = angles
        data = np.reshape(np.load(filename_angles), (n, 3 * len(atoms3)))
        data = np.arccos(data)
        return (RiemannianManifold(data, dim))

lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(lambdamin,lambdamax,nlambdas)]), dtype = np.float16)
experiment = AtomicRegressionLoader( angles,angle_indices, xyz,dim, dimnoise,itermax,tol,lambdas,n_neighbors, n_components, atoms4, ncores, diffusion_time)
experiment.M = experiment.load_data() #if noise == False then noise parameters are overriden
experiment.Mpca = RiemannianManifold(np.load(projected), dim)
experiment.q = n_components
experiment.dimnoise = dimnoise
projector  = np.load(pca_projector)
print(n_neighbors, type(n_neighbors), 'nebstuff')
experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, n_components, diffusion_time, dim)
#experiment.g0 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[0])
folder = output_folder + now
os.mkdir(folder)
axislist = list(combinations_with_replacement(list(range(n_components)), 3))
for i in range(len(axislist)):
	experiment.N.plot(axislist[i], list(range(n)),np.ones(n),.1,.1, folder + '/' + str(axislist[i]))
#print('pregrad',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
#experiments = get_grads_reps_pca2(experiment, nreps, nsel,cores,projector)
#print('precoeff',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
#experiments = get_coeffs_reps(experiments, nreps, lambdas, itermax,nsel,tol)
#xaxis = np.asarray(lambdas, dtype = np.float64) * np.sqrt(n * n_components)
#gnames = np.asarray(list(range(experiment.p)), dtype = str)
#gnames = np.asarray([r"$\displaystyle g_{{}}$".format(i) for i in range(p)])
#print(gnames)
#filename = folder + '/betas'
#print('preplot',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
#plot_betas3reorder(experiments, xaxis, title,filename, gnames,nsel,20)
#filenamescript = folder + '/script.py'
#from shutil import copyfile
#src = workingdirectory + '/codes/experiments/toluenepca_multirun.py'
#copyfile(src, filenamescript)