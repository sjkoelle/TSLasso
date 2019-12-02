from codes.flasso.FlassoManifold import FlassoManifold
from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.otherfunctions.data_stream import data_stream
from codes.geometer.ShapeSpace import compute3angles
import numpy as np
import math
from pathos.multiprocessing import ProcessingPool as Pool

def rotator(rotatee, theta):
    rm = np.asarray([[np.cos(theta), - np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rm3 = np.zeros((3,3))
    rm3[2,2] = 1.
    rm3[0:2,0:2] = rm
    output = np.matmul(rm3, rotatee.transpose())
    return(output.transpose())

def get_grad(t):
    output = np.zeros((3,2))
    output[0,0] = ((np.cos(t) - t*np.sin(t)) / (np.sin(t) + t*np.cos(t)))
    output[2,0] = 1.
    output[1,1] = 1.
    output = output / np.linalg.norm(output, axis = 0)
    return(output)

class SwissRoll(FlassoManifold):
    """
    Parameters
    ----------
    cor : string,
        Data file to load
    xvar : np.array(dtype = int),
        List of adjacencies
    jj : np.array,
        List of adjacencies part 2
    d : int,
        dimension over which to evaluate the radii (smaller usually better)
    rmin : float,
        smallest radius ( = rad_bw_ratio * bandwidth) to consider
    rmax : float,
        largest radius ( = rad_bw_ratio * bandwidth) to consider
    ntry : int,
        number of radii between rmax and rmin to try
    run_parallel : bool,
        whether to run the analysis in parallel over radii
    search_space : str,
        either 'linspace' or 'logspace', choose to search in log or linear space
    rad_bw_ratio : float,
        the ratio of radius and kernel bandwidth, default to be 3 (radius = 3*h)
    Methods
    -------
    generate_data :
        Simulates data
    get_atoms_4 :
    	Gets atomic tetrahedra based off of ii and jj
    get_atoms_3 :
    	Gets triples of atoms

    """

    # AtomicRegression(dim, ii, jj, filename)
    def __init__(self,  xvar,cores, noise):
        natoms = 9
        self.xvar = xvar
        self.cores = cores
        self.noise = noise
        self.dim = 2
        self.d = 3
        self.p = 5

    def generate_data(self, n,theta):
        self.n = n
        d = self.d
        xvar = self.xvar
        dim = self.dim
        noise = self.noise
        self.theta = theta
        ts = 1.5 * np.pi * (1 + 2 * np.random.uniform(low=0.0, high=1.0, size=n))
        x = ts * np.cos(ts)
        y = 21 * np.random.uniform(low=0.0, high=1.0, size=n)
        z = ts * np.sin(ts)
        self.ts = ts
        X = np.vstack((x, y, z))
        #X += noise * generator.randn(3, n_samples)
        X = X.T
        data = rotator(X,theta)
        #data = np.reshape(results, (n, (d)))
        return (RiemannianManifold(data, dim))

    def get_dx_g_full(self, data):
        d = self.d
        p = self.p
        n = data.shape[0]
        theta = self.theta
        ts = self.ts[self.selected_points]
        grads = np.asarray([get_grad(t) for t in ts])
        grads2 = rotator(grads, theta)
        fullgrads = np.zeros((n, d, p))
        for i in range(n):
            fullgrads[:, 0, 2] = np.ones(n)
            fullgrads[:, 1, 3] = np.ones(n)
            fullgrads[:, 2, 4] = np.ones(n)
            fullgrads[:, :, 0:2] = grads2
        output = np.swapaxes(fullgrads,1,2)
        return (output)

