from codes.experimentclasses.AtomicRegression import AtomicRegression
from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.otherfunctions.data_stream import data_stream
from codes.geometer.ShapeSpace import compute3angles
import numpy as np
import math
from pathos.multiprocessing import ProcessingPool as Pool

class RigidEthanolAngles(AtomicRegression):
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
    def __init__(self, dim, cor, xvar,ii,jj, cores, noise, custom_bonds = None):
        natoms = 9
        n = 10000
        self.cor = cor
        self.xvar = xvar
        self.cores = cores
        self.noise = noise
        AtomicRegression.__init__(self, dim,n,ii,jj, natoms, cores)
        if custom_bonds.any() != None:
            self.atoms4 = custom_bonds
            self.p = custom_bonds.shape[0]

    def generate_data(self, noise=False):
        n = self.n
        d = self.d
        cor = self.cor
        xvar = self.xvar
        natoms = self.natoms
        cores = self.cores
        atoms3 = self.atoms3
        dim = self.dim
        noise = self.noise

        positions = np.zeros((n, 9, 3))
        # positions[0,0,:] = np.asarray([0.,0.,0.])
        # positions[0,1,:] = np.asarray([-10.,0.,0.])
        # positions[0,2,:] = np.asarray([1.,10.,0.])
        # #positions[0,8,:] = np.asarray([1.,1.,0.])
        # positions[0,8,:] = np.asarray([2.,10.,-0])
        # positions[0,3,:] = np.asarray([1., np.cos(2/3 * np.pi), np.sin(2/3 * np.pi)])
        # positions[0,4,:] = np.asarray([1., np.cos(2/3 * np.pi), np.sin(4/3 * np.pi)])
        # positions[0,5,:] = np.asarray([-12.,1.,0.])
        # positions[0,6,:] = np.asarray([-12., np.cos(2/3 * np.pi),np.sin(2/3 * np.pi)])
        # positions[0,7,:] = np.asarray([-12.,np.cos(2/3 * np.pi), np.sin(4/3 * np.pi)])
        positions[0,0,:] = np.asarray([0.,0.,0.])
        positions[0,1,:] = np.asarray([-10.,0.,np.sqrt(2)/100])
        positions[0,2,:] = np.asarray([0.,10.,np.sqrt(3)/100])
        #positions[0,8,:] = np.asarray([1.,1.,0.])
        positions[0,8,:] = np.asarray([1.,10.,np.sqrt(5)/100])
        positions[0,3,:] = np.asarray([np.sqrt(7)/100, np.cos(2/3 * np.pi), np.sin(2/3 * np.pi)])
        positions[0,4,:] = np.asarray([np.sqrt(11)/100, np.cos(2/3 * np.pi), np.sin(4/3 * np.pi)])
        positions[0,5,:] = np.asarray([-11.,1.,np.sqrt(13)/100])
        positions[0,6,:] = np.asarray([-11., np.cos(2/3 * np.pi),np.sin(2/3 * np.pi)])
        positions[0,7,:] = np.asarray([-11.,np.cos(2/3 * np.pi), np.sin(4/3 * np.pi)])

        angles1 = np.tile(np.linspace(start=0., stop=2 * math.pi, num=int(np.sqrt(n)), endpoint=False),
                          int(np.sqrt(n)))
        angles2 = np.repeat(np.linspace(start=0., stop=2 * math.pi, num=int(np.sqrt(n)), endpoint=False),
                            int(np.sqrt(n)))
        for i in range(1, n):
            rotationmatrix1 = np.zeros((3, 3))
            rotationmatrix1[1, 1] = 1
            rotationmatrix1[0, 0] = np.cos(angles1[i])
            rotationmatrix1[0, 2] = -np.sin(angles1[i])
            rotationmatrix1[2, 2] = np.cos(angles1[i])
            rotationmatrix1[2, 0] = np.sin(angles1[i])
            rotationmatrix2 = np.zeros((3, 3))
            rotationmatrix2[0, 0] = 1
            rotationmatrix2[1, 1] = np.cos(angles2[i])
            rotationmatrix2[1, 2] = -np.sin(angles2[i])
            rotationmatrix2[2, 2] = np.cos(angles2[i])
            rotationmatrix2[2, 1] = np.sin(angles2[i])
            positions[i, np.asarray([3, 4]), :] = positions[0, np.asarray([3, 4]), :]
            positions[i, np.asarray([2, 8]), :] = np.matmul(rotationmatrix1,
                                                            positions[0, np.asarray([2, 8]),
                                                            :].transpose()).transpose()
            positions[i, np.asarray([1, 5, 6, 7]), :] = np.matmul(rotationmatrix2,
                                                                  positions[0, np.asarray([1, 5, 6, 7]),
                                                                  :].transpose()).transpose()

        covariance = np.identity(natoms)
        for i in range(natoms):
            for j in range(natoms):
                if i != j:
                    covariance[i, j] = cor
        covariance = xvar * covariance
        if noise == True:
            for i in range(n):
                for j in range(3):
                    positions[i, :, j] = np.random.multivariate_normal(positions[i, :, j], covariance, size=1)
        self.positions = positions
        p = Pool(cores)
        results = p.map(lambda i: compute3angles(position=positions[i[0], atoms3[i[1]], :]),
                            data_stream(10000, 84))
        data = np.reshape(results, (n, (d)))
        return (RiemannianManifold(data, dim))

