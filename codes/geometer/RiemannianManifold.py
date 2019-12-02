import matplotlib.pyplot as plt
from megaman.embedding import spectral_embedding
from megaman.geometry import Geometry
from megaman.geometry import RiemannMetric
from scipy import sparse
from scipy.sparse.linalg import norm
from mpl_toolkits.mplot3d import proj3d
#from codes.geometer import TangentBundle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class RiemannianManifold:

    def __init__(self, data, dim):
        self.data = data
        self.n = data.shape[0]
        self.dim = dim

    def compute_geom(self, diffusion_time, n_neighbors):
        data = self.data
        dim = self.dim
        #set radius according to paper (i think this is dim not d)
        radius = (diffusion_time * (diffusion_time * np.pi * 4)**(dim/2))**(0.5)
        #set adjacency radius large enough that points beyond it have affinity close to zero
        bigradius = 3 * radius
        adjacency_method = 'cyflann'
        cyflann_kwds = {'index_type':'kdtrees', 'num_trees':10, 'num_checks':n_neighbors}
        adjacency_kwds = {'radius':bigradius, 'cyflann_kwds':cyflann_kwds}
        affinity_method = 'gaussian'
        affinity_kwds = {'radius':radius}
        laplacian_method = 'geometric'
        laplacian_kwds = {'scaling_epps':radius}
        geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                        affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                        laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
        geom.set_data_matrix(data)
        adjacency_matrix = geom.compute_adjacency_matrix()
        laplacian_matrix = geom.compute_laplacian_matrix()
        return(geom)

    def compute_geom_brute(self, diffusion_time, n_neighbors):
        data = self.data
        dim = self.dim
        #set radius according to paper (i think this is dim not d)
        radius = (diffusion_time * (diffusion_time * np.pi * 4)**(dim/2))**(0.5)
        #set adjacency radius large enough that points beyond it have affinity close to zero
        bigradius = 3 * radius
        adjacency_method = 'brute'
        adjacency_kwds = {'radius':bigradius}
        affinity_method = 'gaussian'
        affinity_kwds = {'radius':radius}
        laplacian_method = 'geometric'
        laplacian_kwds = {'scaling_epps':1}
        geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                        affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                        laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
        geom.set_data_matrix(data)
        geom.adjacency_matrix = geom.compute_adjacency_matrix()
        geom.laplacian_matrix = geom.compute_affinity_matrix()
        geom.laplacian_matrix = self.get_laplacian(geom, radius)
        return(geom)

    def get_laplacian(self, geom, rad):
        n = geom.affinity_matrix.shape[0]
        x = np.squeeze(geom.affinity_matrix.sum(axis = 1))
        y = sparse.spdiags(x, 0, x.size, x.size)
        yinv = sparse.linalg.inv(y)
        tildepp = yinv @ geom.affinity_matrix @ yinv
        tildex = np.squeeze(tildepp.sum(axis = 1))
        tildey = sparse.spdiags(tildex, 0, tildex.size, tildex.size)
        tildeyinv = sparse.linalg.inv(tildey)
        lapland = (sparse.identity(n) - tildeyinv @ tildepp)
        lb = 4* lapland / (rad**2)
        return(lb)

    def compute_spectral_embedding(self, geom):
        geom.Y0, eigenvalues, eigenvectors = spectral_embedding.spectral_embedding(geom=geom, eigen_solver='amg', random_state=6, diffusion_maps=True, diffusion_time = diffusion_time, n_components=n_components)

    def get_induced_rmetric(self, embedding, geom, selected_points):
        dim = self.dim
        laplacian = geom.laplacian_matrix
        laplacian = laplacian.tocsr()[selected_points].tocoo()
        geom.rmetric = RiemannMetric(embedding[selected_points,:],laplacian,n_dim=dim)
        geom.rmetric.get_rmetric()
        return(geom)

    #this RiemannMetric needs to be rewritten to take fewer arguments
    #this enables more arbitary use of RiemannMetric class
    def get_identity_rmetric(self, embedding, geom, embedding_eigenvectors, selected_points):
        dim = self.dim
        laplacian = geom.laplacian_matrix
        laplacian = laplacian.tocsr()[selected_points].tocoo()
        geom.rmetric = RiemannMetric(embedding[selected_points,:],laplacian,n_dim = dim )
        geom.rmetric.get_rmetric()
        geom.rmetric.Hsvals = np.identity(dim)
        geom.rmetric.Gsvals = np.identity(dim)

    def plot(self, axes, selected_points, c,s,alpha,filename, cbar = True):

        data = self.data
        if len(axes) == 2:
            fig = plt.figure(figsize=plt.figaspect(.5))
            ax = fig.add_subplot(1, 1,1)
            cax = ax.scatter(data[:, axes[0]][selected_points], data[:, axes[1]][selected_points],  c = c, s=  s, alpha=alpha, marker = '.')
            ax.set_axis_off()
            if cbar == True:
                fig.colorbar(cax)
            fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)
        if len(axes) == 3:
            x = data[:, axes[0]][selected_points]
            y = data[:, axes[1]][selected_points]
            z = data[:, axes[2]][selected_points]
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            cax = ax.scatter(x, y, z, c=c, s=s, alpha=alpha, marker='.')
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            fig2 = plt.figure(figsize=plt.figaspect(.5))
            ax2 = fig2.add_subplot(1, 1, 1)
            cax2 = ax2.scatter(x2, y2, c=c, s=s, alpha=alpha, marker='.')
            ax2.set_axis_off()
            if cbar == True:
                fig2.colorbar(cax2)
            fig2.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)


    def get_embedding3(self, geom, n_components, diffusion_time , d):

        Y0, eigenvalues, eigenvectors  = spectral_embedding.spectral_embedding(geom=geom, eigen_solver='amg', random_state=6, diffusion_maps=True, diffusion_time = diffusion_time, n_components=n_components)
        #geom.rmetric = RiemannMetric(Y0,geom.laplacian_matrix,n_dim=n_components)
        geom.rmetric = RiemannMetric(Y0,geom.laplacian_matrix,n_dim=d)
        geom.rmetric.get_rmetric()
        output = RiemannianManifold(Y0, d)
        output.geom = geom
        return(output)

    def plot_induced_rmetric(self, axes, selected_points,s,alpha):
        data = self.data
        geom = self.geom

        svals = geom.rmetric.Gsvals
        n_svals = geom.rmetric.Gsvals.shape[1]
        if len(axes) == 2:
            2+2
        if len(axes) == 3:
            for j in range(n_svals):
                fig = plt.figure(figsize=(20,8))
                ax = fig.add_subplot(1, (n_svals + 1), (j+1), projection='3d')
                cax = ax.scatter(data[:, axes[0]][selected_points], data[:, axes[1]][selected_points], data[:,axes[2]][selected_points], c = svals[:,j][selected_points], s=  s, alpha=alpha, marker = '.')
                ax.set_axis_off()
                fig.colorbar(cax)
    #
    # def get_flat_tangent_bundle(self, manifold):
    #     affinity_matrix = manifold.geom.affinity_matrix
    #     n = self.n
    #     d = self.d
    #     dim = self.dim
    #     data = manifold.data
    #
    #     flat_tangent_bases = np.zeros((n, d, dim))
    #     for k in range(n):
    #         weights = affinity_matrix[k].tocsr()
    #         nbr = affinity_matrix[k].indices
    #         nbr_weights = weights[:, nbr].todense()
    #         xbar = np.dot(nbr_weights, data[nbr, :]) / np.sum(nbr_weights)
    #         cent = (data[nbr, :] - xbar)
    #         wcent = np.multiply(nbr_weights, cent.transpose())
    #         sig = np.matmul(wcent, wcent.transpose())
    #         e_vals, e_vecs = np.linalg.eigh(sig)
    #         j = e_vals.argsort()[::-1]  # Returns indices that will sort array from greatest to least.
    #         e_vec = e_vecs[:, j]
    #         e_vec = e_vec[:, :dim]
    #         flat_tangent_bases[k, :, :] = e_vec
    #
    #     output = TangentBundle(manifold, flat_tangent_bases)
    #     return (output)
    def compute_nbr_wts(self, A, sample):
        Ps = list()
        nbrs = list()
        for ii in range(len(sample)):
            w = np.array(A[sample[ii],:].todense()).flatten()
            p = w / np.sum(w)
            nbrs.append(np.where(p > 0)[0])
            Ps.append(p[nbrs[ii]])
        return(Ps, nbrs)

    def get_wlpca_tangent_sel(self, M, selectedpoints, dim = None):

        n = self.n
        nsel = len(selectedpoints)
        if dim == None:
            dim = self.dim
        data = M.data
        A = M.geom.affinity_matrix
        (PS, nbrs) = self.compute_nbr_wts(A, selectedpoints)
        d = M.data.shape[1]
        tangent_bases = np.zeros((nsel, d, dim))
        for i in range(nsel):
            # print(i)
            p = PS[i]
            nbr = nbrs[i]
            Z = (data[nbr, :] - np.dot(p, data[nbr, :])) * p[:, np.newaxis]
            sig = np.dot(Z.transpose(), Z)
            e_vals, e_vecs = np.linalg.eigh(sig)
            j = e_vals.argsort()[::-1]  # Returns indices that will sort array from greatest to least.
            e_vec = e_vecs[:, j]
            e_vec = e_vec[:, :dim]
            tangent_bases[i, :, :] = e_vec
        return (tangent_bases)

    def __copy__(self):
        return(RiemannianManifold(self.data, self.dim))