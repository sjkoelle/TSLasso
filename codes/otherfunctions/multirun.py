from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.geometer.ShapeSpace import ShapeSpace
from codes.geometer.TangentBundle import TangentBundle
import numpy as np
import copy


def get_grads_reps(experiment, nreps, nsel,cores):

    experiments = {}
    dim = experiment.dim

    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].M, experiments[i].M.selected_points)
        subM = RiemannianManifold(experiments[i].M.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].M, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
                                                   experiments[i].M.selected_points)
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        experiments[i].dg_w = experiments[i].project(experiments[i].dw, experiments[i].dg_x)
        tb_w_tangent_bases = experiments[i].project(experiments[i].dw, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
        experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)

def get_grads_reps_pca(experiment, nreps, nsel,cores, projector):

    experiments = {}
    dim = experiment.dim
    dimnoise = experiment.dimnoise
    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
        subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
                                                   experiments[i].M.selected_points,dimnoise)
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        experiments[i].dg_x_pca = np.asarray([np.matmul(projector, experiments[i].dg_x[j].transpose()) for j in range(nsel)])
        experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
        experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
        tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
        experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)

def get_grads_reps_pca2(experiment, nreps, nsel,cores, projector):

    experiments = {}
    dim = experiment.dim
    dimnoise = experiment.dimnoise
    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
        subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
                                                   experiments[i].M.selected_points,dimnoise)
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        experiments[i].dg_w = experiments[i].project(np.swapaxes(experiments[i].dw, 1, 2),
                                             experiments[i].project(experiments[i].dw, experiments[i].dg_x))
        experiments[i].dg_w_pca = np.asarray([np.matmul(projector, experiments[i].dg_w[j].transpose()).transpose() for j in range(nsel)])
        experiments[i].dgw_norm = experiments[i].normalize(experiments[i].dg_w_pca)
        # tb_w_tangent_bases = experiment.project(experiment.dw_pca, np.swapaxes(subM.tb.tangent_bases,1,2))
        # experiment.dgw_norm = experiment.normalize(experiment.dg_w)
        experiments[i].dg_M = experiments[i].project(subM.tb.tangent_bases, experiments[i].dgw_norm)

        #experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
        #experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
        #tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        #experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
        #experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)



def get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores, projector):

    experiments = {}
    dim = experiment.dim
    dimnoise = experiment.dimnoise
    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
        subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        #experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
        #                                           experiments[i].M.selected_points,dimnoise)
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        #experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
        experiments[i].df_M = np.asarray([np.identity(dim) for i in range(nsel)])
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        experiments[i].dg_w = experiments[i].project(np.swapaxes(experiments[i].dw, 1, 2),
                                             experiments[i].project(experiments[i].dw, experiments[i].dg_x))
        experiments[i].dg_w_pca = np.asarray([np.matmul(projector, experiments[i].dg_w[j].transpose()).transpose() for j in range(nsel)])
        experiments[i].dgw_norm = experiments[i].normalize(experiments[i].dg_w_pca)
        # tb_w_tangent_bases = experiment.project(experiment.dw_pca, np.swapaxes(subM.tb.tangent_bases,1,2))
        # experiment.dgw_norm = experiment.normalize(experiment.dg_w)
        #experiment[i].dg_M = experiment.project(subM.tb.tangent_bases, experiment.dgw_norm)
        experiments[i].dg_M = experiments[i].project(subM.tb.tangent_bases, experiments[i].dgw_norm)

        #experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
        #experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
        #tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        #experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
        #experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)


def get_grads_reps_noshape(experiment, nreps, nsel,cores):

    experiments = {}
    dim = experiment.dim

    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        experiments[i].selected_points = experiments[i].M.selected_points
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].M, experiments[i].M.selected_points)
        subM = RiemannianManifold(experiments[i].M.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].M, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
                                                   experiments[i].M.selected_points)
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        #experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        #experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        #experiments[i].dg_w = experiments[i].project(experiments[i].dw, experiments[i].dg_x)
        #tb_w_tangent_bases = experiments[i].project(experiments[i].dw, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        experiments[i].dg_x_norm = experiments[i].normalize(experiments[i].dg_x)
        experiments[i].dg_M = experiments[i].project(tangent_bases, experiments[i].dg_x_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)


def get_grads_reps_noshape_tangent(experiment, nreps, nsel, cores):

    experiments = {}
    dim = experiment.dim

    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        experiments[i].selected_points = experiments[i].M.selected_points
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].M, experiments[i].M.selected_points)
        subM = RiemannianManifold(experiments[i].M.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        #experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].M, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
        #                                           experiments[i].M.selected_points)
        experiments[i].df_M = np.asarray([np.identity(dim) for i in range(nsel)])
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        #experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        #experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        #experiments[i].dg_w = experiments[i].project(experiments[i].dw, experiments[i].dg_x)
        #tb_w_tangent_bases = experiments[i].project(experiments[i].dw, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        experiments[i].dg_x_norm = experiments[i].normalize(experiments[i].dg_x)
        experiments[i].dg_M = experiments[i].project(tangent_bases, experiments[i].dg_x_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)

def get_coeffs_reps(experiments, nreps, lambdas, itermax,nsel,tol):

    for i in range(nreps):
        dimnoise = experiments[i].dimnoise
        experiments[i].xtrain, experiments[i].groups = experiments[i].construct_X_js(experiments[i].dg_M)
        experiments[i].ytrain = experiments[i].construct_Y_js(experiments[i].df_M,dimnoise)
        experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
                                                       nsel, experiments[i].q, itermax, tol)
    return(experiments)

def get_coeffs_reps_tangent(experiments, nreps, lambdas, itermax,nsel,tol):
    dim = experiments[0].dim
    for i in range(nreps):
        dimnoise = experiments[i].dimnoise
        experiments[i].xtrain, experiments[i].groups = experiments[i].construct_X(experiments[i].dg_M)
        experiments[i].ytrain = experiments[i].construct_Y(experiments[i].df_M, list(range(nsel)))
        experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
                                                       nsel, dim, itermax, tol)
    return(experiments)

def get_losses_reps(experiments, lambdas, nreps):

    losses = np.zeros((nreps, lambdas))
    for i in range(nreps):
        losses[i] = experiments[i].get_l2loss(experiments[i].coeffs, experiments[i].ytrain, experiments[i].xtrain)
    return(losses)

def get_penalty_reps(experiments, lambdas, nreps):

    penalties = np.zeros((nreps, lambdas))
    for i in range(nreps):
        penalties[i] = experiments[i].compute_penalty2(experiments[i].coeffs)
    return(penalties)