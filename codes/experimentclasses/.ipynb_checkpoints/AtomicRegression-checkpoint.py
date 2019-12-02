from codes.flasso.FlassoManifold import FlassoManifold
import autograd.numpy as np
import scipy.stats
#import autograd.numpy as np
from autograd import grad
from scipy import sparse
import itertools
import collections


class AtomicRegression(FlassoManifold):
    """
    Parameters
    ----------
    filename : string,
        Data file to load
    ii : np.array(dtype = int),
        List of adjacencies
    jj : np.array,
        List of adjacencies part 2

    Methods
    -------
    load_data :
        Loads filename as AtomicRegression.data
    get_atoms_4 :
    	Gets atomic tetrahedra based off of ii and jj
    get_atoms_3 :
    	Gets triples of atoms

    """

    # AtomicRegression(dim, ii, jj, filename)
    def __init__(self, dim, n, ii,jj,natoms, cores):
        self.ii = ii
        self.jj = jj
        self.n = n
        self.natoms = natoms
        self.cores = cores
        self.atoms4, self.p = self.get_atoms_4(ii,jj)
        self.d = natoms
        self.atoms3, self.d = self.get_atoms_3()
        self.dim = dim
        self.gradg4 = grad(self.g4)

    def get_atoms_3(self):
        natoms = self.natoms

        atoms3 = np.asarray(list(itertools.combinations(range(natoms), 3)))
        return (atoms3, atoms3.shape[0] * 3)

    def get_atoms_4(self, ii, jj):
        # ii = self.ii
        # jj = self.jj
        natoms = self.natoms
        adj = np.asarray([ii, jj])
        adj2 = adj.copy()
        adj2[0, :] = adj[1, :]
        adj2[1, :] = adj[0, :]
        adjacencymatrix = np.concatenate([adj, adj2], axis=1)
        ii = adjacencymatrix[0, :]
        jj = adjacencymatrix[1, :]
        # ii = ii - 1
        # jj = jj - 1
        # known adjacencies
        molecularadjacencymatrix = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)))
        # compute atomic tetrahedra with central atoms in middle two coordinates
        atoms4 = np.ones((1, 4))
        for i in range(natoms):
            nebs = molecularadjacencymatrix[i].indices
            # nnebs = len(molecularadjacencymatrix[i].indices)
            for j in nebs:
                if j > i:
                    i1s = np.setdiff1d(molecularadjacencymatrix[i].indices, j)
                    j1s = np.setdiff1d(molecularadjacencymatrix[j].indices, i)
                    for j1 in j1s:
                        for i1 in i1s:
                            atom4 = np.reshape(np.asarray([i1, i, j, j1, ]), (1, 4))
                            atoms4 = np.concatenate((atoms4, atom4), axis=0)
        atoms4 = atoms4[1:atoms4.shape[0], :]
        atoms4 = np.asarray(atoms4, dtype=int)
        return (atoms4, atoms4.shape[0])

    # def get_atoms_4(self):
    #     atoms4 = np.asarray([[6, 1, 0, 4], [5, 1, 0, 4], [4, 0, 2, 8], [7, 6, 5, 1], [3, 0, 2, 4]], dtype=int)
    #     return (atoms4, atoms4.shape[0])

    def get_dx_g_full(self, data):
        d = self.d
        p = self.p
        n = data.shape[0]

        output = np.zeros((n, p, d))
        for i in range(n):
            output[i, :, :] = self.get_dx_g(data[i]).transpose()
        return (output)

    def get_dx_g(self, x):
        atoms4 = self.atoms4
        atoms3 = self.atoms3
        p = self.p
        d = self.d

        combos = np.asarray([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])
        output = np.zeros((d, p))
        for k in range(p):
            atom4 = atoms4[k, :]
            angles4 = []
            # get identities of triangles on boundary of tetrahedron
            # which atoms3 are in the atoms4...

            actived = [np.where([set(item).issubset(atom4[combos[i, :]]) for item in atoms3])[0][0] for i in range(4)]

            actived = np.asarray(actived, dtype=int)
            naive = np.reshape(x, (int(x.shape[0] / 3), 3))[actived, :]
            for i in range(4):
                a = atoms3[actived[i]]
                b = atom4[np.in1d(atom4, atoms3[actived[i]])]
                for j in range(3):
                    angles4.append(naive[i, np.where(a == b[j])[0]])
            # the jth positEion in the ith row contains the gradient corresponding to the jth position in the truncated atom4
            a4 = np.reshape(angles4, (4, 3))
            # fitin = g4(a4)[1]
            fitin = self.gradg4(a4)
            faceindex = np.zeros(4)
            for j in range(4):
                face = atom4[combos[j]]
                for i in range(4):
                    if collections.Counter(atoms3[actived][i]) == collections.Counter(face):
                        faceindex[j] = i
            faceindex = np.asarray(faceindex, dtype=int)
            anglerowtooutput = actived[faceindex]
            #print(anglerowtooutput)
            for i in range(4):
                face = atom4[combos[i]]
                buffer = np.asarray(scipy.stats.rankdata(face) - 1, dtype=int)
                for j in range(3):
                    output[3 * anglerowtooutput[i] + buffer[j], k] = fitin[i, j]
        return (output)

    def get_g_full(self,data):
        n = data.shape[0]
        p = self.p

        output = np.zeros((n,p))
        for i in range(n):
            output[i,:] = self.get_g(data[i]).transpose()
        return(output)

    def get_g(self,x):
        atoms3 = self.atoms3
        atoms4 = self.atoms4
        p = self.p

        output = np.zeros(p)
        combos = np.asarray([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])
        for k in range(p):
            atom4 = atoms4[k, :]
            angles4 = []
            # get identities of triangles on boundary of tetrahedron
            actived = np.zeros(4)
            for i in range(4):
                actived[i] = np.where([set(item).issubset(atom4[combos[i, :]]) for item in atoms3])[0][0]
            actived = np.asarray(actived, dtype=int)
            naive = np.reshape(x, (int(x.shape[0] / 3), 3))[actived, :]
            for i in range(4):
                a = atoms3[actived[i]]
                b = atom4[np.in1d(atom4, atoms3[actived[i]])]
                for j in range(3):
                    angles4.append(naive[i, np.where(a == b[j])[0]])
            # the jth position in the ith row contains the gradient corresponding to the jth position in the truncated atom4
            a4 = np.reshape(angles4, (4, 3))
            fitin = self.g4(a4)
            # plus the lowest index first
            output[k] = fitin
        return (output)
    #
    # # need to check range of arccos
    # def positions_to_torsion(self, positions4):
    #     positions4 = torch.tensor(positions4, requires_grad=True)
    #     d1 = positions4[0]
    #     c1 = positions4[1]
    #     c2 = positions4[2]
    #     d2 = positions4[3]
    #     cc = c2 - c1
    #     ip = torch.sum((d1 - c1) * (c2 - c1)) / (torch.sum((c2 - c1) ** 2))
    #     tilded1 = [d1[0] - ip * cc[0], d1[1] - ip * cc[1], d1[2] - ip * cc[2]]
    #     iq = torch.sum((d2 - c2) * (c1 - c2)) / (torch.sum((c1 - c2) ** 2))
    #     cc2 = (c1 - c2)
    #     tilded2 = [d2[0] - iq * cc2[0], d2[1] - iq * cc2[1], d2[2] - iq * cc2[2]]
    #     tilded2star = [tilded2[0] + cc2[0], tilded2[1] + cc2[1], tilded2[2] + cc2[2]]
    #     ab = torch.sqrt((tilded2star[0] - c1[0]) ** 2 + (tilded2star[1] - c1[1]) ** 2 + (tilded2star[2] - c1[2]) ** 2)
    #     bc = torch.sqrt((tilded2star[0] - tilded1[0]) ** 2 + (tilded2star[1] - tilded1[1]) ** 2 + (
    #                 tilded2star[2] - tilded1[2]) ** 2)
    #     ca = torch.sqrt((tilded1[0] - c1[0]) ** 2 + (tilded1[1] - c1[1]) ** 2 + (tilded1[2] - c1[2]) ** 2)
    #     output = torch.acos((ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca))
    #     return (output)
    #
    # def g4(self, positions4, grad=True):
    #     positions4 = torch.tensor(positions4, requires_grad=True)
    #     torsion = self.positions_to_torsion(positions4)
    #     torsion.backward(retain_graph=True)
    #     return (torsion, positions4.grad)

    def g4(self,angles4):
        # output = np.ones((4,3))
        # determine distances between points
        # angles411 = np.arccos(angles4[1, 1])
        # angles410 = np.arccos(angles4[1, 0])
        # angles421 = np.arccos(angles4[2, 1])
        # angles431 = np.arccos(angles4[3, 1])
        # angles420 = np.arccos(angles4[2, 0])
        # angles430 = np.arccos(angles4[3, 0])
        # angles400 = np.arccos(angles4[0, 0])
        # angles401 = np.arccos(angles4[0, 1])
        angles411 = (angles4[1, 1])
        angles410 = (angles4[1, 0])
        angles421 = (angles4[2, 1])
        angles431 = (angles4[3, 1])
        angles420 = (angles4[2, 0])
        angles430 = (angles4[3, 0])
        angles400 = (angles4[0, 0])
        #angles401 = (angles4[0, 1])
        c2d2 = 1 / np.sin(angles411)
        c1d2 = 1 / np.sin(angles410)
        d2_d1c2 = c2d2 * np.sin(angles421)
        d2_c1d1 = np.sin(angles431) * c1d2
        d1c2 = (d2_d1c2 / np.tan(angles420)) + (d2_d1c2 / np.tan(angles421))
        d1c1 = (d2_c1d1 / np.tan(angles430)) + (d2_c1d1 / np.tan(angles431))
        d1d2 = d2_d1c2 / np.sin(angles420)
        #d1_c1c2 = np.sin(angles401) * d1c1
        # law of cosines
        #c1c2 = (d1c1 ** 2 + d1c2 ** 2 - 2 * d1c2 * d1c1 * np.cos(angles400)) ** 0.5
        result = []
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(d1c1)
        result.append(0.0)
        result.append(0.0)
        result.append(d1c2 * np.cos(angles400))
        result.append(d1c2 * np.sin(angles400))
        result.append(0.0)
        s = c1d2 * np.cos(angles410) / (c2d2 * np.cos(angles411) + c1d2 * np.cos(angles410))
        asdf = [(1 - s) * d1c1 + s * d1c2 * np.cos(angles400), s * d1c2 * np.sin(angles400), 0.]
        slope = (result[3] - result[6]) / (result[7])
        x = d1d2 * np.cos(angles430)
        result.append(x)
        # rise = slope * (x - d1c1)
        rise = slope * (x - asdf[0])
        y = asdf[1] + rise
        result.append(y)
        z = (d1d2 ** 2 - (x ** 2 + y ** 2)) ** 0.5
        result.append(z)
        #    return(asdf[2])
        # print(result)
        result = np.asarray(result)
        #    return(result[11])
        d1 = result[0:3]
        #    return(d1[])
        c1 = result[3:6]
        c2 = result[6:9]
        d2 = result[9:12]
        cc = (c2 - c1)
        ip = np.inner((d1 - c1), (c2 - c1)) / (np.sum((c2 - c1) ** 2))
        # return(output)
        tilded1 = [d1[0] - ip * cc[0], d1[1] - ip * cc[1], d1[2] - ip * cc[2]]
        iq = np.inner((d2 - c2), (c1 - c2)) / (np.sum((c1 - c2) ** 2))
        cc2 = (c1 - c2)
        tilded2 = [d2[0] - iq * cc2[0], d2[1] - iq * cc2[1], d2[2] - iq * cc2[2]]
        tilded2star = [tilded2[0] + cc2[0], tilded2[1] + cc2[1], tilded2[2] + cc2[2]]
        ab = np.sqrt((tilded2star[0] - c1[0]) ** 2 + (tilded2star[1] - c1[1]) ** 2 + (tilded2star[2] - c1[2]) ** 2)
        bc = np.sqrt((tilded2star[0] - tilded1[0]) ** 2 + (tilded2star[1] - tilded1[1]) ** 2 + (tilded2star[2] - tilded1[2]) ** 2)
        ca = np.sqrt((tilded1[0] - c1[0]) ** 2 + (tilded1[1] - c1[1]) ** 2 + (tilded1[2] - c1[2]) ** 2)
        output = (ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca)
        return (output)
