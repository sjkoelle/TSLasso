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
from codes.experimentclasses.ResorcinolAngles import ResorcinolAngles
from codes.otherfunctions.multirun import get_coeffs_reps_tangent
from codes.otherfunctions.multirun import get_grads_reps_pca2_tangent
from codes.otherfunctions.multiplot import plot_betas, plot_betas2reorder_tangent
from codes.geometer.RiemannianManifold import RiemannianManifold

#set parameters
n = 50000 #number of data points to simulate
nsel = 50 #number of points to analyze with lasso
itermax = 100000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([5,10,15,20,25,50,75,100], dtype = np.float16)#lambda values for lasso
#lambdas = np.asarray([0,2.95339658e-06, 5.90679317e-06, 8.86018975e-06, 1.18135863e-05,
#       1.47669829e-05, 2.95339658e-05, 4.43009487e-05, 5.90679317e-05])
#lambdas = np.asarray([0,.001,.01,.1,1,10], dtype = np.float16)
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,3,11)]), dtype = np.float16)
#lambdas = np.asarray([0,1,2,3,4,5,6,7,8,9,10], dtype = np.float16)
n_neighbors = 1000 #number of neighbors in megaman
n_components = 5 #number of embedding dimensions (diffusion maps)
#diffusion_time = 1. #diffusion time controls gaussian kernel radius per gradients paper
diffusion_time = 3. #diffusion time controls gaussian kernel radius per gradients paper
dim = 2 #slow manifold dimension
dimnoise = 2 #noise dimension
cores = 16 #number of cores for parallel processing
natoms = 14
ii = np.asarray(list(range(natoms)), dtype = int)
jj = np.asarray(list(range(natoms)), dtype = int)

#run experiment
#atoms4 = np.asarray([[9,0,1,2],[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,1],[5,6,1,0]],dtype = int)
atoms4 = np.asarray([[12,6,0,5],[13,10,4,5],[12,6,0,11],[13,10,4,11],[10,13,9,3],[6,12,7,1],[13,10,11,12],[13,10,11,6],[12,6,11,10],[12,6,11,13]],dtype = int)
experiment = ResorcinolAngles(dim, n, ii, jj,cores,atoms4)
experiment.M = experiment.load_data() #if noise == False then noise parameters are overriden
experiment.Mpca = RiemannianManifold(np.load(workingdirectory + '/untracked_data/chemistry_data/resorcinol_anglespca_pca50.npy'), dim)
projector  = np.load(workingdirectory + '/untracked_data/chemistry_data/resorcinol_anglespca_pca50_components.npy')
folder = workingdirectory + '/Figures/resorcinol/' + now
experiment.q = n_components
experiment.dimnoise = dimnoise
n = experiment.n

experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, n_components, diffusion_time, dim)
experiment.g0 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[0])
experiment.g1 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[1])

os.mkdir(folder)
experiment.N.plot([0,1], list(range(n)),experiment.g0,.1,.1, folder + '/g1_01')
experiment.N.plot([0,1,2], list(range(n)),experiment.g0,.1,.1, folder + '/g1_012')
experiment.N.plot([0,1], list(range(n)),experiment.g1,.1,.1, folder + '/g2_01')
experiment.N.plot([0,1,2], list(range(n)),experiment.g1,.1,.1, folder + '/g2_012')
experiment.N.plot([1,2,3], list(range(n)),experiment.g0,.1,.1, folder + '/g1_123')
experiment.N.plot([1,2,3], list(range(n)),experiment.g1,.1,.1, folder + '/g2_123')
experiment.N.plot([2,3,4], list(range(n)),experiment.g0,.1,.1, folder + '/g1_234')
experiment.N.plot([2,3,4], list(range(n)),experiment.g1,.1,.1, folder + '/g2_234')


#experiment.M.selected_points = np.random.choice(list(range(n)),nsel,replace = False)
nreps = 5
#import pickle
#with open('ethanolsavegeom1.pkl', 'wb') as output:
#    pickle.dump(experiment.N, output, pickle.HIGHEST_PROTOCOL)
print('pregrad',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
#experiments = get_grads_reps_pca(experiment, nreps, nsel,cores,projector)
experiments = get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores,projector)
#with open('tolueneexperiments0306_3custom_1000.pkl', 'wb') as output:
#    pickle.dump(experiments, output, pickle.HIGHEST_PROTOCOL)
print('precoeff',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
experiments = get_coeffs_reps_tangent(experiments, nreps, lambdas, itermax,nsel,tol)
xaxis = np.asarray(lambdas, dtype = np.float64) * np.sqrt(n * n_components)
title ='Toluene'
#gnames = np.asarray(list(range(experiment.p)), dtype = str)
gnames = np.asarray([r"$\displaystyle g_1$",
	r"$\displaystyle g_2$",
	r"$\displaystyle g_3$",
	r"$\displaystyle g_4$",
	r"$\displaystyle g_5$",
	r"$\displaystyle g_6$",
	r"$\displaystyle g_7$",
	r"$\displaystyle g_8$",
	r"$\displaystyle g_9$",
	r"$\displaystyle g_{10}$"])
filename = folder + '/betas_tangent'
print('preplot',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
plot_betas2reorder_tangent(experiments, xaxis, title,filename, gnames,nsel)
filenamescript = folder + '/script.py'
from shutil import copyfile
src = workingdirectory + '/codes/experiments/resorcinol_multirun_tangent.py'
copyfile(src, filenamescript)

