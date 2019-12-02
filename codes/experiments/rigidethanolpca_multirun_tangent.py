#Samson Koelle
#Meila group
#021419


#rootdirectory = '/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients'
#f = open(rootdirectory + '/code/source/packagecontrol.py')
#source = f.read()
#exec(source)
#f = open(rootdirectory + '/code/source/sourcecontrol.py')
#source = f.read()
#exec(source)
#f = open(rootdirectory + '/code/source/RigidEthanol.py')
#source = f.read()
#exec(source)
import matplotlib
matplotlib.use('Agg')
import os
import datetime
import numpy as np
import dill as pickle
import random
import sys
np.random.seed(0)
random.seed(0)
now = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)
#print(os.getcwd())
from codes.experimentclasses.RigidEthanolPCA import RigidEthanolPCA
from codes.otherfunctions.multirun import get_coeffs_reps_tangent
from codes.otherfunctions.multirun import get_grads_reps_pca2_tangent
from codes.otherfunctions.multiplot import plot_betas, plot_betas2reorder
from codes.geometer.RiemannianManifold import RiemannianManifold

#set parameters
n = 10000 #number of data points to simulate
nsel = 100 #number of points to analyze with lasso
itermax = 1000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([0,.01,.1,1,10,100], dtype = np.float16)#lambda values for lasso
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,1,11)]), dtype = np.float16)
n_neighbors = 1000 #number of neighbors in megaman
n_components = 3 #number of embedding dimensions (diffusion maps)
#diffusion_time = 1. #diffusion time controls gaussian kernel radius per gradients paper
diffusion_time =.05 #(yuchia suggestion)
dim = 2 #manifold dimension
dimnoise = 2
cores = 3 #number of cores for parallel processing
cor = 0.0 #correlation for noise
var = 0.00001 #variance scaler for noise
ii = np.asarray([0,0,0,0,1,1,1,2]) # atom adjacencies for dihedral angle computation
jj = np.asarray([1,2,3,4,5,6,7,8])

#run experiment
atoms4 = np.asarray([[6,1,0,4],[4,0,2,8],[7,6,5,1],[3,0,2,4]],dtype = int)
experiment = RigidEthanolPCA(dim, cor,var,ii,jj, cores, False, atoms4)
experiment.M, experiment.Mpca,projector  = experiment.generate_data(noise = True)
experiment.q = n_components
experiment.dimnoise = dimnoise
experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, n_components, diffusion_time, dim)
experiment.g0 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[0])
experiment.g1 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[1])
folder = workingdirectory + '/Figures/rigidethanol/' + now
os.mkdir(folder)
experiment.N.plot([0,1,2], list(range(n)),experiment.g0,.1,.1, folder + '/g1')
experiment.N.plot([0,1,2], list(range(n)),experiment.g1,.1,.1, folder + '/g2')

#experiment.M.selected_points = np.random.choice(list(range(n)),nsel,replace = False)
nreps = 5
#import pickle
#with open('ethanolsavegeom1.pkl', 'wb') as output:
#    pickle.dump(experiment.N, output, pickle.HIGHEST_PROTOCOL)
print('pregrad',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
experiments = get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores,projector)
#experiments.df_M
#with open('tolueneexperiments0306_3custom_1000.pkl', 'wb') as output:
#    pickle.dump(experiments, output, pickle.HIGHEST_PROTOCOL)
print('precoeff',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
experiments = get_coeffs_reps_tangent(experiments, nreps, lambdas, itermax,nsel,tol)
xaxis = np.asarray(lambdas, dtype = np.float64) * np.sqrt(n * n_components)
title ='RigidEthanolTangent'
gnames = np.asarray([r"$\displaystyle g_1$",
	r"$\displaystyle g_2$",
	r"$\displaystyle g_3$",
	r"$\displaystyle g_4$"])
#gnames = np.asarray(list(range(experiment.p)), dtype = str)
#folder = workingdirectory + '/Figures/rigidethanol/' + now
#os.mkdir(folder)
filename = folder + '/betas'
print('preplot',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
plot_betas2reorder(experiments, xaxis, title,filename, gnames,nsel)
filenamescript = folder + '/script.py'
from shutil import copyfile
src = workingdirectory + '/codes/experiments/rigidethanolpca_multirun_tangent.py'
copyfile(src, filenamescript)
