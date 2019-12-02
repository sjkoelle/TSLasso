from codes.experimentclasses.AtomicRegression import AtomicRegression
from codes.geometer.RiemannianManifold import RiemannianManifold
import numpy as np
import scipy
import os
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]

class EthanolAngles(AtomicRegression):
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
    def __init__(self, dim, ii,jj,cores, custom_bonds = None):
        self.ii = ii
        self.jj = jj
        natoms = 9
        n = 50000
        self.n = n
        self.natoms = natoms
        self.atoms4, self.p = self.get_atoms_4(ii,jj)
        self.atoms3, self.d = self.get_atoms_3()
        self.dim = 2
        AtomicRegression.__init__(self,dim,n,ii,jj, natoms, cores)
        if custom_bonds.any() != None:
            self.atoms4 = custom_bonds
            self.p = custom_bonds.shape[0]

    def load_data(self):
        # filename = 'tolueneangles.npz'
        atoms3 = self.atoms3
        dim = self.dim
        # cor = self.cor
        # xvar = self.xvar
        filename_xyz = workingdirectory + '/untracked_data/chemistry_data/ethanol.mat'
        filename_angle_indices = workingdirectory + '/untracked_data/chemistry_data/ethanolindices022119.npy'
        data_xyz_loaded = scipy.io.loadmat(filename_xyz)
        # print(data_xyz_loaded['R'].shape)
        angle_indices = np.load(filename_angle_indices)
        self.timeindices = angle_indices
        positions = data_xyz_loaded['R'][angle_indices]
        self.positions = positions
        filename_angles = workingdirectory + '/untracked_data/chemistry_data/ethanolangles022119.npy'
        data = np.reshape(np.load(filename_angles), (50000, 3 * len(atoms3)))
        data = np.arccos(data)
        # print(positions[0])
        # if angles == True:
        # p = Pool(cores)
        # results = p.map(lambda i: compute3angles(position = positions[i[0],atoms3[i[1]],:]),data_stream(10,84))
        # data2 = np.reshape(results, (10,(d)))
        # else:
        #    data = np.reshape(positions,(n,d))
        return (RiemannianManifold(data, dim))
