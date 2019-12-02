import matplotlib.pyplot as plt
from megaman.embedding import spectral_embedding
from megaman.geometry import Geometry
from megaman.geometry import RiemannMetric
from scipy import sparse
from scipy.sparse.linalg import norm
from codes.geometer import TangentBundle
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import torch
from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.otherfunctions.data_stream_custom_range import data_stream_custom_range

#from code.source.utilities import data_stream_custom_range

class ShapeSpace(RiemannianManifold):

    def __init__(self, positions, angular_coordinates):
        self.positions = positions
        self.angular_coordinates = angular_coordinates

    def torchComputeAngle(self, poses):
        combos = torch.tensor([[0, 1], [1, 2], [2, 0]])
        ab = torch.norm(poses[combos[0, 0], :] - poses[combos[0, 1], :])
        bc = torch.norm(poses[combos[1, 0], :] - poses[combos[1, 1], :])
        ca = torch.norm(poses[combos[2, 0], :] - poses[combos[2, 1], :])
        output = torch.acos((ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca))
        return (output)

    def torchCompute3angles(self, position):
        angles = np.ones(3)
        gradients = np.zeros((3, position.shape[0], 3))
        for i in range(3):
            #print(i)
            poses = torch.tensor(position[[i, (i + 1) % 3, (i + 2) % 3], :], requires_grad=True)
            tempang = self.torchComputeAngle(poses)
            tempang.backward()
            angles[i] = tempang.detach().numpy()
            gradients[i] = poses.grad[[(2 * i) % 3, (2 * i + 1) % 3, (2 * i + 2) % 3], :]
            # del(poses)
        return(angles, gradients)

    def reshapepointdata(self, pointdata, atoms3):
        natoms = len(np.unique(atoms3))
        output = np.zeros((pointdata.shape[0] * pointdata.shape[1], natoms * 3))
        for i in range(pointdata.shape[0]):
            for j in range(pointdata.shape[1]):
                for k in range(pointdata.shape[2]):
                    for l in range(pointdata.shape[3]):
                        # print(atoms3[k]*3 + l)
                        output[i * 3 + j, atoms3[i][k] * 3 + l] = pointdata[i, j, k, l]
        return(output)

    def get_wilson(self, selind, atoms3, tdata):
        natoms = len(np.unique(atoms3))
        jacobien = np.zeros((len(selind), len(atoms3) * 3, natoms * 3))
        for i in range(len(selind)):
            pointdata = tdata[i * len(atoms3): (i + 1) * len(atoms3)]
            jacobien[i] = self.reshapepointdata(pointdata, atoms3)
        return(jacobien)

    def get_internal_projector(self, natoms, jacobien, selind):
        nnonzerosvd = 3 * natoms - 7
        internalprojector = np.zeros((len(selind), jacobien.shape[1], nnonzerosvd))
        for i in range(len(selind)):
            # when rescaling vectors, big directions must be shrunk... this is the riemannian bundle and should be treated as such
            asdf = np.linalg.svd(jacobien[i])
            #internalprojector[i] = (asdf[0][:, :nnonzerosvd] * asdf[1][:nnonzerosvd] ** (-1))
            internalprojector[i] = (asdf[0][:, :nnonzerosvd] )
        return (internalprojector)

    def get_dw(self,cores,atoms3,natoms, selected_points):
        positions = self.positions
        self.selected_points = selected_points
        p = Pool(cores)
        n = len(selected_points)
        results = p.map(lambda i: self.torchCompute3angles(position=positions[i[0], atoms3[i[1]], :]),
                        data_stream_custom_range(selected_points, len(atoms3)))
        tdata = np.asarray([results[i][1] for i in range(n * len(atoms3))])
        # for i in range(len(selected_points)):
        #     pointdata = tdata[i * len(atoms3): (i + 1) * len(atoms3)]
        jacobien = self.get_wilson(selected_points, atoms3, tdata)
        internalprojector = self.get_internal_projector(natoms, jacobien, selected_points)
        return(internalprojector)

    # def project_vectors(self, internalprojector):
    #     p = self.p
    #     natoms = self.natoms
    #     selected_points = self.selected_points
    #     dg_config = np.zeros((len(selected_points), natoms * 3 - 7, p))
    #     for i in range(len(selected_points)):
    #         dg_config[i] = np.matmul(internalprojector[i].transpose(), self.dg_x[i].transpose())
    #     dg_config_norm = self.normalize(dg_config)
    #     naive_tanget_bases = self.get_wlpca_tangent_sel(self.M, selected_points)
    #     config_tangent_bases = np.zeros((nsel, 20, 2))
    #     for i in range(len(selected_points)):
    #         config_tangent_bases[i] = np.matmul(internalprojector[i].transpose(), naive_tanget_bases[i])
    #     dg_M = np.zeros((nsel, p, 2))
    #     for i in range(nsel):
    #         dg_M[i] = np.matmul(config_tangent_bases[i].transpose(), dg_config_norm[i]).transpose()
    #     self.project_dg_x_norm(self, shape_tangent_bundle, dg_config_norm)

def computeAngle(poses):
    combos = np.asarray([[0,1],[1,2],[2,0]])
    ab = np.linalg.norm(poses[combos[0,0],:] - poses[combos[0,1],:])
    bc = np.linalg.norm(poses[combos[1,0],:] - poses[combos[1,1],:])
    ca = np.linalg.norm(poses[combos[2,0],:] - poses[combos[2,1],:])
    output = np.arccos((ab**2 - bc**2 + ca**2) / (2 * ab * ca))
    #output = (ab**2 - bc**2 + ca**2) / (2 * ab * ca)
    return(output)


def compute3angles(position):
    angles = np.ones(3)
    for i in range(3):
        poses = position[[i, (i+1) %3, (i+2) % 3],:]
        angles[i] = computeAngle(poses)
    return(angles)