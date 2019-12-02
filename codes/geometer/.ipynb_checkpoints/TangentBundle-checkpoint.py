import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())
#from codes.geometer.RiemannianManifold import RiemannianManifold

class TangentBundle(RiemannianManifold):

    """
    This class stores data points, bases of their tangent space, and the riemannian metric induced by the positions of the points
    Parameters
    ----------
    embedding : int
        Number of samples
    tangent_bases : 

    Methods
    -------
    get_local_rmetric(self, embedding, tangent_bases)
        estimates rmetric in given embedding and tangent_bases
    """
    def __init__(self, manifold, tangent_bases):
        self.manifold = manifold
        self.tangent_bases = tangent_bases

    def plot_tangent_bases(self, show_pts, arrow_pts, axes, tangent_axes, arrow_scaler = .1):
        data = self.manifold.data
        tangent_bases = self.tangent_bases
        dim  = self.manifold.dim
        #colors = new color for every tan basis vector
        colors = ['r','g']
        if len(axes) == 3:
            fig = plt.figure(figsize=plt.figaspect(.5))
            ax = fig.add_subplot(1, 1,1, projection='3d')
            cax = ax.scatter(data[show_pts, 0], data[show_pts, 1], data[show_pts,2], s=  .5, alpha=.1, marker = '.', cmap = parula_map)
            for i in arrow_pts:
                for j in tangent_axes:
                    #print(i)
                    X,Y,Z = data[i,:]
                    U,V,W = tangent_bases[i,:,j] * arrow_scaler
                    ax.quiver(X,Y,Z,U,V,W, alpha = .5, color = colors[j])
                    #U, V, W = tangent_bases[i,:,1] * arrow_scaler
                    #ax.quiver(X,Y,Z,U,V,W, alpha = .5, color = 'r')
            ax.set_axis_off()
        if len(axes) == 2:
            2+2
        #fig.colorbar(cax)

    def plot_local_riemann_metric(self, axes, selectedpoints):
        data = self.manifold.data
        #G_sval = self.G_sval

        if len(axes == 3):
            fig = plt.figure(figsize=plt.figaspect(.5))
            for k in range(dim):
                ax = fig.add_subplot(dim, 1,(k+1) , projection='3d')
                cax = ax.scatter(data[selectedpoints, 0], data[selectedpoints, 1], data[selectedpoints,2], s=  .5, alpha=1, marker = '.', c = G_sval[selectedpoints,k], cmap = parula_map)
                ax.set_axis_off()
                fig.colorbar(cax)
        if len(axes == 2):
            2+2

    #these should be moved to the base class
    def get_local_riemann_metric(self, tangent_bases, embedding):
        n = self.manifold.data.shape[0]
        dim = self.manifold.dim
        affinity_matrix = self.geom.affinity_matrix
        
        metric = np.zeros((n, dim, dim))
        dualmetric = np.zeros((n, dim, dim))
        G_sval = np.zeros((n, dim))
        H_sval = np.zeros((n, dim))
        for i in range(n):
            nnebs = len(affinity_matrix[i].indices)
            nebs = affinity_matrix[0:nnebs]
            projected_positions = {}
            projected_positions[i] = np.matmul(tangent_bases[i,:,:].transpose(), (embedding[nebs,:] - embedding[i,:]).transpose())
            laplacian_pt = affinity_matrix[i].copy()
            pp = projected_positions[i].transpose()
            nbr = laplacian_pt.indices[1:]
            dualmetric[i,:,:], metric[i,:,:], G_sval[i,:], H_sval[i,:] = self._local_riemann_metric(pp, laplacian_pt)

        return(dualmetric, metric, G_sval, H_sval)

    def _local_riemann_metric(self, pp, laplacian_pt):
        dim = self.dim
        
        n_neighbors = len(laplacian_pt.indices)
        h_dual_metric = np.zeros((dim, dim ))
        lp = laplacian_pt.tocsr()
        nebs = laplacian_pt.indices[0:len(laplacian_pt.indices)]
        lpnebs = lp[:,nebs]
        for i in np.arange(dim):
            for j in np.arange(i,dim):
                #print(lpnebs.shape, pp.shape)
                h_dual_metric[i, j] = 0.5*lpnebs.multiply(pp[:,j]).multiply(pp[:,i]).sum()
        for j in np.arange(dim-1):
            for i in np.arange(j+1,self.dim):
                h_dual_metric[i,j] = h_dual_metric[j,i]

        # compute rmetric if requested
        riemann_metric = np.linalg.inv(h_dual_metric)
        return h_dual_metric, riemann_metric, np.linalg.svd(riemann_metric)[1], np.linalg.svd(h_dual_metric)[1]


    def __copy__(self):
        return(TangentBundle(self.manifold, self.tangent_bases))

 