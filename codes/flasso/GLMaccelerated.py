#The true champion

import autograd.numpy as np
from autograd import jacobian
from autograd import elementwise_grad
from autograd import grad

import logging
from copy import deepcopy

#import numpy as np
from scipy.special import expit
from pyglmnet import utils


class GLM:
    
    def __init__(self, xs, ys, reg_lambda, group,max_iter, learning_rate, tol,parameter):
        self.xs = xs
        self.ys = ys
        self.reg_lambda = reg_lambda
        self.group = np.asarray(group)
        #print(self.group.shape)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.Tau = None
        self.alpha = 1.
        self.lossresults = {}
        self.dls = {}
        self.parameter = parameter
        self.l2loss = {}
        self.penalty = {}
        
    def _prox(self,beta, thresh):
        """Proximal operator."""
        
        #print(thresh, beta)
        #print('beginprox', beta[0:2],thresh)
        group_ids = np.unique(self.group)
        result = np.zeros(beta.shape)
        result = np.asarray(result, dtype = float)
        #print('gids',group_ids)
        for i in range(len(group_ids)):
            gid = i 
            #print(self.group)
            idxs_to_update = np.where(self.group == gid)[0]
            #print('idx',idxs_to_update)
            #print('norm', np.linalg.norm(beta[idxs_to_update]))
            if np.linalg.norm(beta[idxs_to_update]) > 0.:
                #print('in here', len(idxs_to_update))
                potentialoutput = beta[idxs_to_update] - (thresh / np.linalg.norm(beta[idxs_to_update])) * beta[idxs_to_update]
                posind = np.where(beta[idxs_to_update] > 0.)[0]
                negind = np.where(beta[idxs_to_update] < 0.)[0]
                po = beta[idxs_to_update].copy()
                #print('potention', potentialoutput[0:2])
                po[posind] = np.asarray(np.clip(potentialoutput[posind],a_min = 0., a_max = 1e15), dtype = float)
                po[negind] = np.asarray(np.clip(potentialoutput[negind],a_min = -1e15, a_max = 0.), dtype = float)
                result[idxs_to_update] = po
        #print('end', result[0:2])
        return result

    def _grad_L2loss(self, beta, X, y):
        #print(beta.shape,X.shape,y.shape)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_samples = np.float(X.shape[0])
        z = np.dot(X, beta)
        #grad_beta = 1. / n_samples * np.transpose(np.dot(np.transpose(z - y), X))
        grad_beta = np.transpose(np.dot(np.transpose(z - y), X))
        #print('gb',grad_beta.shape)
        return grad_beta
    
    def _loss(self,beta, reg_lambda, X, y):
        """Define the objective function for elastic net."""
        L = self._logL(beta, X, y)
        P = self._L1penalty(beta)
        J = -L + reg_lambda * P
        return J
    
    def _logL(self,beta, X, y):
        """The log likelihood."""
        #print('beginlogL', np.linalg.norm(beta), np.linalg.norm(X), np.linalg.norm(y),y.shape,beta.shape,X.shape,)
        l = np.dot(X, beta)
        logL = -0.5 * np.sum((y - l)**2)
        #print('endlogL',logL)
        return logL
    
    def _L2loss(self,beta,X,y):
        #print('beginl2', np.linalg.norm(beta), np.linalg.norm(X), np.linalg.norm(y), y.shape)
        output = -self._logL(beta, X, y)
        #print('outl2',output)
        return(output)
    
    def _L1penalty(self, beta):
        """The L1 penalty"""
        # Compute the L1 penalty
        group = self.group
        group_ids = np.unique(self.group)
        L1penalty = 0.0
        for group_id in group_ids:
            L1penalty += np.linalg.norm(beta[self.group == group_id], 2)
        return L1penalty
    
    #def fhatlambda(self,lamb,x,y):
    def fhatlambda(self,lamb,betanew,betaold):
        xs = self.xs
        ys = self.ys
        #print(ys.shape,'fhatlam')
        #print(self._L2loss(betaold,xs,ys),self._L2loss(betanew,xs,ys),'old','new') 
        output = self._L2loss(betaold,xs,ys) + np.dot(self._grad_L2loss(betaold,xs,ys).transpose(),(betanew-betaold)) + (1/(2*lamb)) * np.linalg.norm(betanew-betaold)**2
        return(output)
    
    #_btalgorithm(yk,lamb,.5,1000, rl)
    def _btalgorithm(self,bet,lam,b,maxx,rl):
        
        #print('lam',lam)
        X = self.xs
        y = self.ys
        #print('beginbt', np.linalg.norm(y))
        #print('beginbt',self._L2loss(bet,X,y))
        #print(np.linalg.norm(bet))
        grad_beta = self._grad_L2loss(beta = bet, X = X, y = y)
        for i in range(maxx):
            betn = bet - lam * grad_beta
            z = self._prox(betn, lam * rl)
            fz = self._L2loss(z,X,y)
            #print(fz,'fz')
            fhatz = self.fhatlambda(lam,z, bet)
            if fz <= fhatz:
            #print(fhatz - fz)
            #if 0 <= 1:
                break
            lam = b*lam
        return(z,lam)
    
    def fit(self):
        

        group  = self.group
        print(group.shape)
        lambdas = self.reg_lambda
        parameter = self.parameter
        X = self.xs
        #print(X.shape)
        y = self.ys
        
        np.random.RandomState(0)
        group = np.asarray(group, dtype = np.int64)

        #print(group.shape[0])
        group.dtype = np.int64
        #print(group.shape[0])
        #print(X.shape[1])
        if group.shape[0] != X.shape[1]:
            raise ValueError('group should be (n_features,)')

        # type check for data matrix
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        n_features = np.float(X.shape[1])
        n_features = np.int64(n_features)
        if y.ndim == 1:
            y = y[:, np.newaxis]
            self.ys = y
        #print(y.shape)
        n_classes = 1
        n_classes = np.int64(n_classes)

        beta_hat = 1 / (n_features) * np.random.normal(0.0, 1.0, [n_features, n_classes])
        fit_params = list()
        
        for l, rl in enumerate(lambdas):
            fit_params.append({'beta': beta_hat})
            if l == 0:
                fit_params[-1]['beta'] = beta_hat
            else:
                fit_params[-1]['beta'] = fit_params[-2]['beta']
            tol = self.tol
            alpha = 1.
            beta = np.zeros([n_features, n_classes])
            beta = fit_params[-1]['beta']
            #print('losser',self._L2loss(beta,X,y))
            g = np.zeros([n_features, n_classes])
            L, DL ,L2,PEN = list(), list() , list(), list()
            lamb = self.learning_rate
            bm1 = beta.copy()
            bm2 = beta.copy()
            for t in range(0, self.max_iter):
                L.append(self._loss(beta, rl, X, y))
                L2.append(self._L2loss(beta,X,y))
                PEN.append(self._L1penalty(beta))
                w = (t / (t+ 3))
                yk = beta + w*(bm1 - bm2)
                #print('losser',self._L2loss(yk,X,y))
                #print('beforebt',np.linalg.norm(yk),np.linalg.norm(X),np.linalg.norm(y))
                beta , lamb = self._btalgorithm(yk,lamb,.5,1000, rl)
                #X = self.xs
                #y = self.ys
                #print('losser2',self._L2loss(beta,X,y))
                bm2 = bm1.copy()
                bm1 = beta.copy()
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < tol:
                        print('converged', rl)
                        msg = ('\tConverged. Loss function:'
                               ' {0:.2f}').format(L[-1])
                        msg = ('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
                        break
                    
            #print(beta)
            fit_params[-1]['beta'] = beta
            self.lossresults[rl] = L
            self.l2loss[rl] = L2
            self.penalty[rl] = PEN
            self.dls[rl] = DL
            #print(L)
        # Update the estimated variables
        
        self.fit_ = fit_params
        self.ynull_ = np.mean(y)

        # Return
        return self
    