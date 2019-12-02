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
from codes.experimentclasses.SwissRoll49 import SwissRoll49
from codes.otherfunctions.multirun import get_coeffs_reps
from codes.otherfunctions.multirun import get_grads_reps_noshape_tangent
from codes.otherfunctions.multiplot import plot_betas_customcolors2

#set parameters
n = 100000 #number of data points to simulate
nsel = 100 #number of points to analyze with lasso
itermax = 1000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([5,10,15,20,25,50,75,100], dtype = np.float16)#lambda values for lasso
#lambdas = np.asarray([0,2.95339658e-06, 5.90679317e-06, 8.86018975e-06, 1.18135863e-05,
#       1.47669829e-05, 2.95339658e-05, 4.43009487e-05, 5.90679317e-05])
#lambdas = np.asarray([0,.0001,.001,.01,.1,1,10], dtype = np.float64)
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,1,11)]), dtype = np.float16)
n_neighbors = 1000 #number of neighbors in megaman
n_components = 2 #number of embedding dimensions (diffusion maps)
#10000 points and .17 dt is best so far
#100000 and .05 is better
#thats with 50 neighbors
#diffusion_time = 0.05 #diffusion time controls gaussian kernel radius per gradients paper
#diffusion_time = 0.1
diffusion_time = 0.05
dim = 2 #manifold dimension
cores = 16 #number of cores for parallel processing

experiment = SwissRoll49(xvar = 0.0,cores = cores, noise = False)
experiment.M = experiment.generate_data(n = n,theta = np.pi/4) #if noise == False then noise parameters are overriden
experiment.q = n_components
print('pregeom',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
experiment.M.geom = experiment.M.compute_geom(diffusion_time, n_neighbors)
print('preembed',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
experiment.N = experiment.M.get_embedding3(experiment.M.geom, n_components, diffusion_time, dim)
#experiment.M.selected_points = np.random.choice(list(range(n)),nsel,replace = False)
#experiment.selected_points = experiment.M.selected_points
nreps = 5
#import pickle
#with open('ethanolsavegeom1.pkl', 'wb') as output:
#    pickle.dump(experiment.N, output, pickle.HIGHEST_PROTOCOL)
print('pregrad',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
import matplotlib.pyplot as plt
#plt.scatter(experiment.N.data[:,0], experiment.N.data[:,1])
folder = workingdirectory + '/Figures/swissroll/' + now
os.mkdir(folder)
#experiment.N.plot([0,1],list(range(n)), experiment.ts,.25,1,folder + '/c1', False)
#experiment.N.plot([0,1],list(range(n)), experiment.ys,.25,1,folder + '/c0', False)
#plt.scatter(experiment.N.data[:,0], experiment.N.data[:,1], c = experiment.ts)
#plt.savefig('/Users/samsonkoelle/Desktop/swizz' + str(now))
experiments = get_grads_reps_noshape_tangent(experiment, nreps, nsel,cores)
#with open('tolueneexperiments0306_3custom_1000.pkl', 'wb') as output:
#    pickle.dump(experiments, output, pickle.HIGHEST_PROTOCOL)
print('precoeff',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
#experiments = get_coeffs_reps(experiments, nreps, lambdas, itermax,nsel,tol)


def get_betas_spam2(xs, ys, groups, lambdas, n, q, itermax, tol):
    # n = xs.shape[0]
    p = len(np.unique(groups))
    lambdas = np.asarray(lambdas, dtype=np.float64)
    yadd = np.expand_dims(ys, 1)
    groups = np.asarray(groups, dtype=np.int32) + 1
    W0 = np.zeros((xs.shape[1], yadd.shape[1]), dtype=np.float32)
    Xsam = np.asfortranarray(xs, dtype=np.float32)
    Ysam = np.asfortranarray(yadd, dtype=np.float32)
    coeffs = np.zeros((len(lambdas), q, n, p))
    for i in range(len(lambdas)):
        # alpha = spams.fistaFlat(Xsam,Dsam2,alpha0sam,ind_groupsam,lambda1 = lambdas[i],mode = mode,itermax = itermax,tol = tol,numThreads = numThreads, regul = "group-lasso-l2")
        # spams.fistaFlat(Y,X,W0,TRUE,numThreads = 1,verbose = TRUE,lambda1 = 0.05, it0 = 10, max_it = 200,L0 = 0.1, tol = 1e-3, intercept = FALSE,pos = FALSE,compute_gram = TRUE, loss = 'square',regul = 'l1')
        output = spams.fistaFlat(Ysam, Xsam, W0, True, groups=groups, numThreads=-1, verbose=True,
                                     lambda1=lambdas[i], it0=100, max_it=itermax, L0=0.5, tol=tol, intercept=False,
                                     pos=False, compute_gram=True, loss='square', regul='group-lasso-l2', ista=False,
                                     subgrad=False, a=0.1, b=1000)
        coeffs[i, :, :, :] = np.reshape(output[0], (q, n, p))
        # print(output[1])
    return (coeffs)

def get_coeffs(experiment, lambdas, itermax, nsel, tol):
    experiment.xtrain, experiment.groups = experiment.construct_X(experiment.dg_M)
    experiment.ytrain = experiment.construct_Y(experiment.df_M, list(range(nsel)))
    experiment.coeffs = get_betas_spam2(experiment.xtrain, experiment.ytrain, experiment.groups, lambdas,
                                        nsel, experiment.dim, itermax, tol)
    return (experiment)


def get_coeffs_parallel(experiments, nreps, lambdas, itermax, nsel, tol, cores):
    p = Pool(cores)
    results = p.map(
        lambda i: get_coeffs(experiment=experiments[i], lambdas=lambdas, itermax=itermax, nsel=nsel, tol=tol),
        range(nreps))
    output = {}
    for i in range(nreps):
        output[i] = results[i]
    return (output)


from pathos.multiprocessing import ProcessingPool as Pool
import spams





experiments = get_coeffs_parallel(experiments, nreps, lambdas, itermax, nsel, tol, cores)

xaxis = np.asarray(lambdas, dtype = np.float64) * np.sqrt(nsel * 2)
title ='Swiss Roll'
gnames = np.asarray(list(range(experiment.p)), dtype = str)
#folder = workingdirectory + '/Figures/swissroll/' + now
#os.mkdir(folder)
filename = folder + '/betas'
print('preplot',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
colors = np.hstack([np.repeat('red',2), np.repeat('black',49)])
legtitle = 'Function type'
color_labels = np.asarray(['Manifold coordinates','Ambient coordinates'])
#plot_betas(experiments, xaxis, title,filename, gnames,nsel)
plot_betas_customcolors2(experiments, xaxis, title,filename, gnames,nsel,colors, legtitle, color_labels)

2+2

