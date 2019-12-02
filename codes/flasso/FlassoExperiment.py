#import matplotlib
#matplotlib.use('Agg')
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.colors
import spams
from collections import OrderedDict
from pylab import rcParams
from codes.flasso.GLMaccelerated import GLM
rcParams['figure.figsize'] = 25, 10

def cosine_similarity(a,b):
	output = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b) )
	return(output)

class FlassoExperiment:
    """
    FlassoExperiment
    """
    def __init__(self):
        2+2

    def get_norms(self, differential):
        n = differential.shape[0]
        #could be p, or q
        p = differential.shape[1]
        d = differential.shape[2]
        
        differential_normalized = np.zeros(differential.shape)
        vectornorms = np.zeros((n,p))
        for i in range(n):
            for j in range(p):
                if np.linalg.norm(differential[i,j,:]) > 0:
                    vectornorms[i,j] = np.linalg.norm(differential[i,j,:])

        psum = np.sum(vectornorms, axis = 0)
        return(psum / n)
                
    def _flatten_coefficient(self, coeff):
        n = coeff.shape[1]
        p = coeff.shape[2]
        q = coeff.shape[0]

        output = np.zeros((n*p*q))
        for k in range(q):
            for i in range(n):
                output[((k*n*p) + (i*p)):((k*n*p) + (i+1)*p)] = coeff[k,i,:]
        return(output)


    def get_l2loss(self, coeffs,ys,xs):

        n = coeffs.shape[2]
        nlam = coeffs.shape[0]
        output = np.zeros(nlam)
        for i in range(nlam):
            coeffvec = self._flatten_coefficient(coeffs[i])
            output[i] = np.sum((ys - np.dot(coeffvec, xs.transpose()))**2)
        output = output / n
        return(output)

    def normalize(self, differential):
        n = differential.shape[0]
        #could be p, or q
        p = differential.shape[1]
        d = differential.shape[2]
        
        differential_normalized = np.zeros(differential.shape)
        vectornorms = np.zeros((n,p))
        for i in range(n):
            for j in range(p):
                if np.linalg.norm(differential[i,j,:]) > 0:
                    vectornorms[i,j] = np.linalg.norm(differential[i,j,:])
        psum = np.sum(vectornorms, axis = 0)
        for j in range(p):
            if psum[j] > 0:
                differential_normalized[:,j,:] = (differential[:,j,:] / psum[j])#*n
                
        return(differential_normalized)
        
    def get_betas_sam(self, xtrain, ytrain, groups, lambdas, n,q,max_iter, tol, learning_rate):
        
        p = len(np.unique(groups))
        models = GLM(xs = xtrain, ys = ytrain,
            tol=tol,
            group=groups,
            learning_rate=learning_rate,
            max_iter = max_iter,
            #reg_lambda=np.logspace(np.log(100), np.log(0.01), 5, base=np.exp(1)))
            reg_lambda=lambdas,
            parameter = .5)
        models.fit()
        nlam = len(lambdas)
        organizedbetas = np.zeros((nlam, q,n,p))
        for l in range(nlam):
            organizedbetas[l,:,:,:] = np.reshape(models.fit_[l]['beta'], (q,n,p))
        #return(models, organizedbetas)
        return(organizedbetas)

    def construct_X(self, dg_M):
        """ dg_M should have shape n x p x dim
        """
        n = dg_M.shape[0]
        dim = dg_M.shape[2]
        p = dg_M.shape[1]
        
        xmat = np.zeros((n*dim, n*p))
        for i in range(n):
            xmat[(i*dim):(i*dim + dim), (i*p):(i*p + p)] = dg_M[i,:,:].transpose()
        b = [xmat]*dim
        xmatq = scipy.linalg.block_diag(*b) 
        groups = np.zeros(n*p*dim) 
        groups = np.tile(np.tile(np.asarray(np.linspace(start = 0, stop = (p-1), num = p), dtype = int), n),dim)
        
        return(xmatq, list(groups))

    def construct_X_js(self, dg_M):
        """ dg_M should have shape n x p x dim
        """
        n = dg_M.shape[0]
        dim = dg_M.shape[2]
        p = dg_M.shape[1]
        q = self.q
        
        xmat = np.zeros((n*dim, n*p))
        for i in range(n):
            xmat[(i*dim):(i*dim + dim), (i*p):(i*p + p)] = dg_M[i,:,:].transpose()
        b = [xmat]*q
        xmatq = scipy.linalg.block_diag(*b) 
        groups = np.zeros(n*p*q) 
        groups = np.tile(np.tile(np.asarray(np.linspace(start = 0, stop = (p-1), num = p), dtype = int), n),q)
        
        return(xmatq, list(groups))

    def construct_X_js_subset(self, dg_M, selind):
        dg_M_subset = np.zeros(dg_M.shape)
        dg_M_subset[:, selind, :] = dg_M[:, selind, :]
        output = self.construct_X_js(dg_M_subset)
        return (output)

    def construct_Y(self, df_M, sample_pts):
        """ df_M should have shape n x dim x m
        """
        n = len(sample_pts)
        dim = self.dim

        reorg1 = np.swapaxes(df_M, 0, 1)
        yvec = np.reshape(reorg1, (n * dim * dim))
        return(yvec)
        
    def construct_Y_js(self, df_M, dim  = None):
        """ df_M should have shape n x dim x q
        """
        n = df_M.shape[0]
        q = self.q
        if dim == None:
            dim = self.dim
        
        reorg1 = np.swapaxes(df_M, 0,2)
        reorg2 = np.swapaxes(reorg1, 2,1)
        #yvec = np.reshape(reorg2, (n*dim*dim))
        yvec = np.reshape(reorg2, (n*dim*q))
        return(yvec)

    def plot_convergence(self, models, name = 'lossplot.pdf'):
        for key in list(models.keys()):
        #key = list(models.keys())[0]
            xval = np.log(np.asarray(list(range(len(list(models[key].lossresults.values())[0])))) + 1)
            y = list(models[key].lossresults.values())[0]
            plt.plot(xval,y)
        plt.legend(list(models.keys()), loc='upper right')
        plt.savefig(name)

    def plot_convergence_sam(self, models, name = 'lossplot.pdf'):
        for key in list(models.lossresults.keys()):
        #key = list(models.keys())[0]
            xval = np.log(np.asarray(list(range(len(models.lossresults[key])))) + 1)
            y = models.lossresults[key]
            plt.plot(xval,y)
        plt.legend(list(models.lossresults.keys()), loc='upper right')
        plt.savefig(name)

    def plot_bh(self, coeffs, nsample_pts, p, name = 'beta'):
        dim = self.dim
        #p = self.p
        n = nsample_pts
        nlam =coeffs.shape[0]

        for l in range(len(lambdas)):
            if dim > 1:
                fig, axes = plt.subplots(1,q, figsize=(15,30))
                for k in range(q):
                    tempplot = axes[k].imshow(coeffs[l,k,:,:])
                    plt.colorbar(tempplot,ax=axes[k])
            if dim == 1:
                k = 0
                fig, ax = plt.subplots(figsize = (5,5))
                ax.imshow(coeffs[l,k,:,:])
            fig.savefig('lambda' + name + 'heatmap.pdf')  

    def compute_penalty2(self, coeffs):
        n = coeffs.shape[2]
        nlam = coeffs.shape[0]
        q = coeffs.shape[1]
        p = coeffs.shape[3]

        #p = self.p
        pen = np.zeros(nlam)
        for l in range(nlam):
            norm2 = np.zeros(p)
            for j in range(p):
                norm2[j] = np.linalg.norm(coeffs[l,:,:,j])
            pen[l] = np.sum(norm2)
        pen = pen / n
        return(pen)


    def plot_penalty(self, coeffs, xaxis, xlabel,xlog, ylog, title, filename):
        ylabel = r"$\frac{1}{n}\displaystyle \|\beta\|_{1,2}$ "

        if xlog:
            filename = filename + 'xlog'
        if ylog:
            filename = filename + 'ylog'

        pens = self.compute_penalty2(coeffs)
        fig, axes = plt.subplots(1,1, figsize=(15,15))
        axes.plot(xaxis, pens, 'go--', linewidth=5, markersize=0, alpha = 1)
        axes.set_ylim(bottom = 1e-4,top = pens.max())
        if xlog:
            axes.semilogx()
        if ylog:
            axes.semilogy()
        axes.tick_params(labelsize = 50)
        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
        fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
        plt.suptitle(title, fontsize = 55)
        fig.savefig(filename + 'penalty' + str(n))

    def plot_predictions(self,coeffs,xs,ys,xaxis, xlabe, xlog,ylog,title,filename):
        2+2

    def plot_l2loss(self, coeffs, xs,ys,xaxis, xlabel,xlog, ylog, title, filename):

        ylabel = r"$\displaystyle \frac{1}{n}\|y - x\beta\|_2^2$"

        if xlog:
            filename = filename + 'xlog'
        if ylog:
            filename = filename + 'ylog'

        losses = self.get_l2loss(coeffs,ys,xs)
        fig, axes = plt.subplots(1,1, figsize=(15,15))
        axes.plot(xaxis, losses, 'go--', linewidth=5, markersize=0, alpha = 1)
        axes.set_ylim(bottom = 1e-4,top = losses.max())
        if xlog:
            axes.semilogx()
        if ylog:
            axes.semilogy()
        axes.tick_params(labelsize = 50)
        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
        fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
        plt.suptitle(title, fontsize = 55)
        fig.savefig(filename+ 'loss' + str(n))

    def plot_beta_paths_best(self, coeffs, xaxis, gnames, fnames, xlabel,  title, filename, xlog, ylog, norm_betas = False):
        if xlog:
            filename = filename + 'xlog'
        if ylog:
            filename = filename + 'ylog'
        if norm_betas:
            print(filename)
            filename = filename + 'norm'
            print(filename)
            ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
        else:
            ylabel = r"$\displaystyle \hat \beta_{ij}$"
        rcParams['axes.titlesize'] = 30
        plt.rc('text', usetex=True)

        p = coeffs.shape[3]        
        n = coeffs.shape[2]
        q = coeffs.shape[1]
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
        cmap = plt.cm.rainbow

        if norm_betas:
            maxes = np.zeros(q)
            for k in range(q):
                #for j in range(p):
                maxes[k] = np.linalg.norm(coeffs[:,k,:,:], axis = 1).max()
            normax = maxes.max()
        if q>1:
            fig, axes = plt.subplots(1,q, figsize=((15*q),15))
            for k in range(q):
                for j in range(p):
                    if norm_betas == False:
                        for i in range(n):
                            axes[k].plot(xaxis, coeffs[:,k,i,j], 'go--', linewidth=5, markersize=0, alpha = .2, color = cmap(norm(j)), label = gnames[j])
                    else:
                        axes[k].plot(xaxis, np.linalg.norm(coeffs[:,k,:,j], axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
                #axes[k].set_title(fnames[k], fontsize=30)
                axes[k].tick_params(labelsize = 50)
                if xlog:
                    axes[k].semilogx()
                if ylog:
                    axes[k].semilogy()
                if norm_betas == True:
                    axes[k].set_ylim(bottom = 1e-4,top = normax)
                else:
                    if ylog == True:
                        axes[k].set_ylim(bottom = 1e-4, top=coeffs.max())  # adjust the max leaving min unchanged
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            fig, axes = plt.subplots(1,q, figsize=((15*q),15))
            for j in range(p):
                if norm_betas == False:
                    for i in range(n):
                        axes.plot(xaxis, coeffs[:,0,i,j], 'go--', linewidth=5, markersize=0, alpha = .2, color = cmap(norm(j)), label = gnames[j])
                else:
                    axes.plot(xaxis, np.linalg.norm(coeffs[:,0,:,j], axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
            #axes.set_title(fnames[0], fontsize=30)
            axes.tick_params(labelsize = 50)
            if xlog:
                axes.semilogx()
            if ylog:
                axes.semilogy()
            if norm_betas == True:
                axes.set_ylim(bottom = 1e-4,top = normax)
            else:
                if ylog == True:
                    axes.set_ylim(bottom = 1e-4, top=coeffs.max())  # adjust the max leaving min unchanged
            handles, labels = axes.get_legend_handles_labels()

        by_label = OrderedDict(zip(labels, handles))
        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
        fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
        plt.suptitle(title, fontsize = 55)
        fig.subplots_adjust(right=0.75)
        leg_ax = fig.add_axes([.96, 0.15, 0.05, 0.7])
        leg_ax.axis('off')
        #leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
        leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 25})
        leg.set_title('Group',prop={'size':45})
        for l in leg.get_lines():
            l.set_alpha(1)
        fig.savefig(filename + 'beta_paths_log_n' + str(n))

    def plot_beta_paths_best2(self, coeffs, xaxis, gnames, fnames, xlabel,  title, filename, xlog, ylog, colors,lines, norm_betas = False):
        if xlog:
            filename = filename + 'xlog'
        if ylog:
            filename = filename + 'ylog'
        if norm_betas:
            print(filename)
            filename = filename + 'norm'
            print(filename)
            ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
        else:
            ylabel = r"$\displaystyle \hat \beta_{ij}$"
        rcParams['axes.titlesize'] = 30
        plt.rc('text', usetex=True)

        p = coeffs.shape[3]        
        n = coeffs.shape[2]
        q = coeffs.shape[1]
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
        cmap = plt.cm.rainbow

        if norm_betas:
            maxes = np.zeros(q)
            for k in range(q):
                #or j in range(p):
                maxes[k] = np.linalg.norm(coeffs[:,k,:,:], axis = 1).max()
            normax = maxes.max()
        if q>1:
            fig, axes = plt.subplots(1,q, figsize=((15*q),15))
            for k in range(q):
                for j in range(p):
                    if norm_betas == False:
                        for i in range(n):
                            axes[k].plot(xaxis, coeffs[:,k,i,j], 'go--', linewidth=5, markersize=0, alpha = .2, color = colors[j], linestyle = lines[j], label = gnames[j])
                    else:
                        axes[k].plot(xaxis, np.linalg.norm(coeffs[:,k,:,j], axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = colors[j], linestyle = lines[j], label = gnames[j])
                #axes[k].set_title(fnames[k], fontsize=30)
                axes[k].tick_params(labelsize = 50)
                if xlog:
                    axes.set_xscale('symlog')
                if ylog:
                    axes.set_yscale('symlog')
                if norm_betas == True:
                    axes[k].set_ylim(bottom = 1e-4,top = 10*normax)
                else:
                    if ylog == True:
                        axes[k].set_ylim(bottom = 1e-4, top=coeffs.max())  # adjust the max leaving min unchanged
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            fig, axes = plt.subplots(1,q, figsize=((15*q),15))
            for j in range(p):
                if norm_betas == False:
                    for i in range(n):
                        axes.plot(xaxis, coeffs[:,0,i,j], 'go--', linewidth=5, markersize=0, alpha = .2, color = colors[j], linestyle = lines[j], label = gnames[j])
                else:
                    axes.plot(xaxis, np.linalg.norm(coeffs[:,0,:,j], axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = colors[j], linestyle = lines[j], label = gnames[j])
            #axes.set_title(fnames[0], fontsize=30)
            axes.tick_params(labelsize = 50)
            axes.set_xscale('symlog')
            axes.set_yscale('symlog')
            if norm_betas == True:
                axes.set_ylim(bottom = 0,top = 10*normax)
            else:
                if ylog == True:
                    axes.set_ylim(bottom = 0, top=coeffs.max())  # adjust the max leaving min unchanged
            handles, labels = axes.get_legend_handles_labels()

        by_label = OrderedDict(zip(labels, handles))
        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
        fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
        plt.suptitle(title, fontsize = 55)
        fig.subplots_adjust(right=0.75)
        leg_ax = fig.add_axes([.96, 0.15, 0.05, 0.7])
        leg_ax.axis('off')
        #leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
        leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 25})
        leg.set_title('Group',prop={'size':45})
        for l in leg.get_lines():
            l.set_alpha(1)
        fig.savefig(filename + 'beta_paths_log_n2' + str(n))


    def get_cosines(self, dg):
        n = dg.shape[0]
        p = dg.shape[1]
        d = dg.shape[2]

        coses = np.zeros((n,p,p))
        for i in range(n):
            for j in range(p):
                for k in range(p):
                    coses[i,j,k] = cosine_similarity(dg[i,j,:],dg[i,k,:])#sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
        cos_summary = np.abs(coses).sum(axis = 0) / n
        return(cos_summary)

    def plot_norms(self, norms,filename):
        plt.imshow(norms)
        plt.colorbar()
        plt.savefig(filename + 'norms')


    def plot_cosines(self, cos_sumary, filename):
        plt.imshow(cos_sumary)
        plt.colorbar()
        plt.savefig(filename + 'cosines')


    def plot_pairwise_solutions(self, coeffs, lambdas, gnames, fnames,title,filename,log):   

        if log:
            filename = filename + 'xlog'
            filename = filename + 'ylog'
        if norm_betas:
            print(filename)
            filename = filename + 'norm'
            print(filename)
            
        rcParams['axes.titlesize'] = 30
        plt.rc('text', usetex=True)
        p = coeffs.shape[3]        
        n = coeffs.shape[2]
        nlam = coeffs.shape[0]
        fig, axes = plt.subplots(p,q, figsize=((15*q),(15*p)))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=nlam)
        cmap = plt.cm.rainbow
        colors = cmap(norm(range(nlam)))
        for k1 in range(q):
            for k2 in range(q):
                for j1 in range(p):
                    for j2 in range(p):
                        for l in range(nlam):
                            axes[(k1* q + j1),(k2* q +j2)].plot(coeffs[l,k1,:,j1], coeffs[l,k2,:,j2],  alpha = .2, color = cmap(norm(l)), labels = lambdas[l])
                            axes[(k1* q + j1),(k2* q +j2)].set_xlabel(fnames[k1] + gnames[j1], fontsize=30)
                            axes[(k1* q + j1),(k2* q +j2)].set_ylabel(fnames[k1] + gnames[j1], fontsize=30)
                            axes[(k1* q + j1),(k2* q +j2)].tick_params(labelsize = 50)
                            if log:
                                axes[(k1* q + j1),(k2* q +j2)].semilogx()
                                axes[(k1* q + j1),(k2* q +j2)].semilogy()
                
        handles, labels = axes[0,0].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
        fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
        plt.suptitle(title, fontsize = 55)
        fig.subplots_adjust(right=0.75)
        leg_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        leg_ax.axis('off')
        leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
        leg.set_title('Group',prop={'size':45})
        for l in leg.get_lines():
            l.set_alpha(1)
        fig.savefig(filename + 'beta_paths_log_n' + str(n))


    def normalize_xaxis(self, coeff, groups):
        
        #groups = np.asarray(may_experiment.groups)
        tempsum = 0
        i = 0
        ngroups = coeff.shape[3]
        for j in range(ngroups):
            tempsum = tempsum + np.linalg.norm(coeff[i,:,:,j])
        beta0norm = tempsum

        xaxis = np.zeros(len(lambdas))
        for i in range(len(lambdas)):
            tempsum = 0
            for j in range(ngroups):
                tempsum = tempsum + np.linalg.norm(coeff[i,:,:,j])
            xaxis[i] =  tempsum / beta0norm
        return(xaxis)

    def get_betas_spam2(self, xs, ys, groups, lambdas,n,q,itermax,tol):
        
        #n = xs.shape[0]
        p = len(np.unique(groups))
        lambdas = np.asarray(lambdas, dtype = np.float64)
        yadd = np.expand_dims(ys,1)
        groups = np.asarray(groups, dtype = np.int32) + 1
        W0 = np.zeros((xs.shape[1], yadd.shape[1]), dtype = np.float32)
        Xsam = np.asfortranarray(xs,dtype=np.float32)
        Ysam = np.asfortranarray(yadd,dtype=np.float32)
        coeffs= np.zeros((len(lambdas),q,n,p))
        for i in range(len(lambdas)):
            #alpha = spams.fistaFlat(Xsam,Dsam2,alpha0sam,ind_groupsam,lambda1 = lambdas[i],mode = mode,itermax = itermax,tol = tol,numThreads = numThreads, regul = "group-lasso-l2")
            #spams.fistaFlat(Y,X,W0,TRUE,numThreads = 1,verbose = TRUE,lambda1 = 0.05, it0 = 10, max_it = 200,L0 = 0.1, tol = 1e-3, intercept = FALSE,pos = FALSE,compute_gram = TRUE, loss = 'square',regul = 'l1')
            output = spams.fistaFlat(Ysam,Xsam,W0,True,groups = groups, numThreads = -1,verbose = True,lambda1 = lambdas[i], it0 = 100, max_it = itermax,L0 = 0.5, tol = tol, intercept = False,pos = False,compute_gram = True, loss = 'square',regul = 'group-lasso-l2',ista = False,subgrad = False,a = 0.1, b = 1000)
            coeffs[i,:,:,:] = np.reshape(output[0], (q,n,p))
            #print(output[1])
        return(coeffs)

    def plot_coefficient_recovery(self, coeffs,coeffspred, lambdas, filename): 

        p = coeffs.shape[3]
        q = coeffs.shape[1]
        nlam = coeffs.shape[0]

        norm = matplotlib.colors.Normalize(vmin=0, vmax=nlam)
        cmap = plt.cm.rainbow
        colors = cmap(norm(range(p)))
        organizedbetas = coeffs
        ymin = organizedbetas.min() - 0.01
        ymax = organizedbetas.max() + 0.01
        cmap = plt.cm.rainbow
        minimum = lambdas.min()
        #eps = minimum / 10000
        #lambdas = lambdas + eps
        
        if q > 1:
            fig, axes = plt.subplots(nrows = p,ncols = q, figsize=(15,15))
            for k in range(q):
                for j in range(p):
                    #print(k,j)
                    for i in range(nlam):
                        axes[j,k].scatter(coeffspred[k,:,j], coeffs[i,k,:,j],  alpha = .1, color = cmap(norm(i)))
                    axes[j,k].set_ylabel(r"$\displaystyle \widehat{{ d_{{g_{}}} h_{{}}}}$".format(j,k), fontsize = 35)
                    #axes[j,k].set_title('function = ' + str(k + 1) + 'group = ' + str(j+1), fontsize = 15)
                    axes[j,k].set_xlabel(r"$\displaystyle  d_{{g_{}}}^? h_{{}}$".format(j,k), fontsize = 35)
                    axes[j,k].tick_params(labelsize = 40)
                    #axes[j,k].semilogy()
                    #axes[j,k].semilogx()
        if q == 1:
            fig, axes = plt.subplots(nrows = p,ncols = 1, figsize=(15,15))
            for j in range(p):
                #print(j)
                for i in range(nlam):
                    axes[j].scatter(coeffspred[0,:,j], coeffs[i,0,:,j],  alpha = .1, color = cmap(norm(i)))
                axes[j].set_ylabel(r"$\displaystyle \widehat{{d_{{g_{}}} h}}$".format(j), fontsize = 35)
                #axes[j].set_xlabel(predictionlabels[j], fontsize = 35)
                axes[j].set_xlabel(r"$\displaystyle d_{{g_{}}}^? h$".format(j), fontsize = 35, labelpad = -1)
                axes[j].tick_params(labelsize = 40)
                #axes[j].semilogy()
                #axes[j].semilogx()
                #axes[j].set_xlabel('Predicted Beta', fontsize = 15)
                #axes[j].set_title('group = ' + str(j+1), fontsize = 15)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=40)
        cbar.set_label(r"$\displaystyle \lambda$", rotation=270, fontsize = 50)
        #fig.text(0.0, 0.5, 'Estimates', ha='center', va='center', rotation='vertical', fontsize=60)
        #fig.text(0.5, 0.04, r"$\displaystyle d_g^? h$", ha='center', va='center', fontsize=50)
        plt.suptitle(title, fontsize = 55)
        plt.subplots_adjust(wspace = .3, hspace = .7, left=0.2, bottom=0.1, right=0.85, top=0.85)
        #plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.22, right=0.96, top=0.96)
        fig.savefig(filename + 'coefficientrecovery' + str(n))

    def plot_coefficient_error_distribution(self, coeffs,coeffspred, lambdas, filename): 


        p = coeffs.shape[3]
        q = coeffs.shape[1]
        nlam = coeffs.shape[0]
        n_bins = 50

        norm = matplotlib.colors.Normalize(vmin=0, vmax=nlam)
        cmap = plt.cm.rainbow
        colors = cmap(norm(range(p)))
        organizedbetas = coeffs
        ymin = organizedbetas.min() - 0.01
        ymax = organizedbetas.max() + 0.01
        cmap = plt.cm.rainbow
        minimum = lambdas.min()
        #eps = minimum / 10000
        #lambdas = lambdas + eps
        error = coeffs.copy()
        for i in range(nlam):
            error[i] = np.abs(coeffs[i]) - np.abs(coeffspred)
        
        if q > 1:
            fig, axes = plt.subplots(nrows = p,ncols = q, figsize=((15*p),15*q))
            for k in range(q):
                for j in range(p):
                    #print(k,j)
                    for i in range(nlam):
                        #axes[j,k].scatter(coeffspred[k,:,j], coeffs[i,k,:,j],  alpha = .1, color = cmap(norm(i)))
                        n, bins, patches = axes[j,k].hist(error[i,k,:,j], n_bins, normed=1, histtype='step', cumulative=True, label='Empirical', color = cmap(norm(i)))
                        patches[0].set_xy(patches[0].get_xy()[:-1])
                        axes[j,k].set_xlabel(r"$\displaystyle \widehat{{ d_{{g_{}}} h_{{}}}} - d_{{g_{}}}^? h_{{}}$".format(j,k,j,k), fontsize = 40)
                        axes[j,k].tick_params(labelsize = 50)
                        #axes[j,k].set_ylabel(r"$\displaystyle \widehat{{ d_{{g_{}}} h_{{}}}}$".format(j,k), fontsize = 15)
                        #axes[j,k].set_title('function = ' + str(k + 1) + 'group = ' + str(j+1), fontsize = 15)
                        #axes[j,k].set_xlabel(predictionlabels[j,k], fontsize = 35)
                        #axes[j,k].semilogy()
                        #axes[j,k].semilogx()
        if q == 1:
            fig, axes = plt.subplots(nrows = p,ncols = 1, figsize=(15,15))
            for j in range(p):
                print(j)
                for i in range(nlam):
                    #print(j)
                    n, bins, patches = axes[j].hist(error[i,0,:,j], n_bins, normed=1, histtype='step', cumulative=True, label='Empirical', color = cmap(norm(i)))
                    patches[0].set_xy(patches[0].get_xy()[:-1])
                    axes[j].set_xlabel(r"$\displaystyle \widehat{{d_{{g_{}}} h}} - d_{{g_{}}}^? h$".format(j,j), fontsize = 40)
                    axes[j].tick_params(labelsize = 50)
                    #axes[j].scatter(coeffspred[0,:,j], coeffs[i,0,:,j],  alpha = .1, color = cmap(norm(i)))
                    #axes[j].set_ylabel(r"$\displaystyle \widehat{{d_{{g_{}}} h}}$".format(j), fontsize = 25)
                    #axes[j].set_xlabel(predictionlabels[j], fontsize = 35)
                    #axes[j].semilogy()
                    #axes[j].semilogx()
                    #axes[j].set_xlabel('Predicted Beta', fontsize = 15)
                    #axes[j].set_title('group = ' + str(j+1), fontsize = 15)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=40)
        cbar.set_label(r"$\displaystyle \lambda$", rotation=270, fontsize = 60)
        #fig.text(0.0, 0.5, 'Estimates', ha='center', va='center', rotation='vertical', fontsize=60)
        fig.text(0.05, 0.5, r"ECDF", ha='center', va='center', rotation='vertical', fontsize=60)
        #fig.text(0.5, 0.04, r"$\displaystyle d_g^? h$", ha='center', va='center', fontsize=50)
        plt.suptitle(title, fontsize = 55)
        plt.subplots_adjust(wspace = .3, hspace = .85)
        fig.savefig(filename + 'coefficientrecovery.png')


    def plot_coefficient_distribution(self, coeffs,lambdas, filename): 


        p = coeffs.shape[3]
        q = coeffs.shape[1]
        nlam = coeffs.shape[0]
        n_bins = 50

        norm = matplotlib.colors.Normalize(vmin=0, vmax=nlam)
        cmap = plt.cm.rainbow
        colors = cmap(norm(range(p)))
        organizedbetas = coeffs
        ymin = organizedbetas.min() - 0.01
        ymax = organizedbetas.max() + 0.01
        cmap = plt.cm.rainbow
        minimum = lambdas.min()
        
        if q < 1:
            fig, axes = plt.subplots(nrows = p,ncols = q, figsize=(15,15))
            for k in range(q):
                for j in range(p):
                    #print(k,j)
                    for i in range(nlam):
                        #axes[j,k].scatter(coeffspred[k,:,j], coeffs[i,k,:,j],  alpha = .1, color = cmap(norm(i)))
                        n, bins, patches =  axes[j,k].hist(coeffs[i,k,:,j], n_bins, normed=1, histtype='step', cumulative=True, label='Empirical', color = cmap(norm(i)))
                        patches[0].set_xy(patches[0].get_xy()[:-1])
                        axes[j,k].set_xlabel(r"$\displaystyle \widehat{{ d_{{g_{}}} h_{{}}}}$".format(j,k), fontsize = 15)
                        #axes[j,k].set_ylabel(r"$\displaystyle \widehat{{ d_{{g_{}}} h_{{}}}}$".format(j,k), fontsize = 15)
                        #axes[j,k].set_title('function = ' + str(k + 1) + 'group = ' + str(j+1), fontsize = 15)
                        #axes[j,k].set_xlabel(predictionlabels[j,k], fontsize = 35)
                        #axes[j,k].semilogy()
                        #axes[j,k].semilogx()
        if q == 1:
            fig, axes = plt.subplots(nrows = p,ncols = 1, figsize=(15,15))
            for j in range(p):
                print(j)
                for i in range(nlam):
                    #print(j)
                    n, bins, patches = axes[j].hist(coeffs[i,0,:,j], n_bins, normed=1, histtype='step', cumulative=True, label='Empirical', color = cmap(norm(i)))
                    patches[0].set_xy(patches[0].get_xy()[:-1])
                    axes[j].set_xlabel(r"$\displaystyle \widehat{{d_{{g_{}}} h}}$".format(j), fontsize = 25)
                    #axes[j].scatter(coeffspred[0,:,j], coeffs[i,0,:,j],  alpha = .1, color = cmap(norm(i)))
                    #axes[j].set_ylabel(r"$\displaystyle \widehat{{d_{{g_{}}} h}}$".format(j), fontsize = 25)
                    #axes[j].set_xlabel(predictionlabels[j], fontsize = 35)
                    #axes[j].semilogy()
                    #axes[j].semilogx()
                    #axes[j].set_xlabel('Predicted Beta', fontsize = 15)
                    #axes[j].set_title('group = ' + str(j+1), fontsize = 15)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(r"$\displaystyle \lambda$", rotation=270, fontsize = 30)
        #fig.text(0.0, 0.5, 'Estimates', ha='center', va='center', rotation='vertical', fontsize=60)
        fig.text(0.0, 0.5, r"$\displaystyle \frac{1}{n} \sum_{i=1}^n 1_{\widehat{ d_g h} |_i < x}$", ha='center', va='center', rotation='vertical', fontsize=60)
        #fig.text(0.5, 0.04, r"$\displaystyle d_g^? h$", ha='center', va='center', fontsize=50)
        plt.suptitle(title, fontsize = 55)
        plt.subplots_adjust(wspace = .3, hspace = .3)
        fig.savefig(filename + 'coefficientrecovery' + str(n))



