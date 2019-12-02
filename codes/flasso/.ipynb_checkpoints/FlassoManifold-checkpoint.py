import autograd.numpy as np
#from codes.flasso.FlassoExperiment import FlassoExperiment

class FlassoManifold(FlassoExperiment):
    """
    This class provides methods for estimating Df and projecting into tangent spaces with Riemannian metric where f is a data reparameterization
    Parameters
    ----------
    n : int,
        Number of samples
    selectedpoints : np.array(dtype = int),
        Which points to pass to function regression
    dim : int,
        dimension of manifold
    data:
    embedding:
    Dg : function,
        the differential of g
    Optional Parameters
    ----------
    tangent_bundle: np.array((n, d, dim), dtype = float)
        Bases for tangent bundle
    Methods
    -------
    get_flat_tangent_bundle(self),
        get tangent bundle with flat bases
    estimate_Df(self, tangent_bases),
        estimates Df in given tangent bases
    get_rmetric(self, embedding, tangent_bases)
        estimates rmetric in given embedding and tangent_bases
    """

    #A FlassoManifold experiment explains the results of a manifold embedding by a FlassoExperiment
    #The key method is the embedding gradient estimation method (gets dF)
    def __init__(self):
        2+2

    def project(self, tangent_bases, vectors):
        n = vectors.shape[0]
        p = vectors.shape[1]
        dim = tangent_bases.shape[2]

        dg_M = np.zeros((n,p,dim))

        for i in range(n):
            dg_M[i] = np.matmul(tangent_bases[i].transpose(), vectors[i].transpose()).transpose()
        return(dg_M)

    def get_dF_js_idM(self, M, N, M_tangent_bundle_sub, N_tangent_bundle, selectedpoints):

        dim = self.dim
        q = self.q
        affinity_matrix = M.geom.affinity_matrix

        nsel = len(selectedpoints)
        dF = np.zeros((nsel, dim, q))

        for i in range(nsel):
            pt = selectedpoints[i]
            neighborspt = affinity_matrix[selectedpoints[i]].indices
            deltap0 = M.data[neighborspt, :] - M.data[pt, :]
            deltaq0 = N.data[neighborspt, :] - N.data[pt, :]
            projected_M = np.matmul(M_tangent_bundle_sub.tangent_bases[i, :, :].transpose(),
                                    deltap0.transpose()).transpose()
            # projected_rescaled_M = np.matmul(np.diag(M_tangent_bundle_sub.rmetric.Gsvals[selectedpoints[i]]),projected_M.transpose())
            projected_rescaled_M = projected_M.transpose()
            b = np.linalg.pinv(projected_rescaled_M)
            a = np.zeros((len(neighborspt), q))
            rescaled_basis = np.matmul(N_tangent_bundle.tangent_bases[selectedpoints[i], :, :][:, :],
                                       np.diag(N.geom.rmetric.Gsvals[selectedpoints[i]]))
            projected_N = np.dot(rescaled_basis.transpose(), deltaq0.transpose())
            projected_N_expanded = np.matmul(N_tangent_bundle.tangent_bases[selectedpoints[i], :, :][:, :], projected_N)
            a = projected_N_expanded
            dF[i, :, :][:, :] = np.matmul(a, b).transpose()
        return (dF)

    def get_central_point(self, nsel, fitmodel, data):
        centralpoint = np.zeros(nsel)
        for i in range(nsel):
            # print(i)
            distomean = np.zeros(len(fitmodel.adjacency_matrix[i].indices))
            meanpoint = data[fitmodel.adjacency_matrix[i].indices, :].mean(axis=0)
            for j in range(len(fitmodel.adjacency_matrix[i].indices)):
                distomean[j] = np.linalg.norm(data[fitmodel.adjacency_matrix[i].indices[j], :] - meanpoint)
            centralpoint[i] = fitmodel.adjacency_matrix[i].indices[distomean.argmin()]
        centralpoint = np.asarray(centralpoint, dtype=int)
        return (centralpoint)



