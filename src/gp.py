"""
gp.py

Implementation of single- and multi-fidelity Gaussian Process learning models capable of hyperparameter
inference and mean/variance prediction.

created by: Paris Perdikaris, Department of Mechanical Engineering, MIT
first created: 5/20/2017
available: https://github.com/paraklas/GPTutorial

modified by: Andrew McDonald, D-CYPHER Lab, Michigan State University
modifications: simplified and streamlined for use in adaptive multirobot coverage
"""

from __future__ import division
from autograd import value_and_grad
import autograd.numpy as np
from scipy.optimize import minimize


class SFGP:
    """
    A single fidelity Gaussian Process class capable of hyperparameter inference and mean/variance prediction
    """

    def __init__(self, X, y):
        """
        Initialize the SFGP class.

        :param X: [nxD numpy array] of observation points (D is number of dimensions of input space)
        :param y: [nx1 numpy array] of observation values
        :param lenscale: [scalar] approximate lengthscale of GP (accelerates hyperparameter inference convergence)
        """
        self.X = X
        self.y = y

        self.hyp = self.init_params()
        self.jitter = 1e-8
        self.likelihood(self.hyp)

    def __str__(self):
        """
        Represent the model as a string.

        :return: string representation of GP model
        """
        ehyp = np.exp(self.hyp)
        return f"****************************************\n" \
               f"SFGP\n" \
               f"X shape = {self.X.shape}\n" \
               f"log(hyp) = [log(mu)={self.hyp[0]}, log(var)={self.hyp[1]}, " \
               f"log(lenscale)={self.hyp[2]}, log(noise)={self.hyp[3]}]\n" \
               f"hyp = [mu={ehyp[0]}, var={ehyp[1]}, lenscale={ehyp[2]}, noise={ehyp[3]}]\n" \
               f"likelihood = {self.likelihood(self.hyp)}\n" \
               f"****************************************\n"

    def init_params(self):
        """
        Initialize hyperparameters of the GP.

        :return: [1xk numpy array] of GP hyperparameters as initialized (k is number of hyperparameters)
        """
        hyp = np.log(np.ones(4))        # log(mu, var, lenscale, noise)
        self.idx_theta = np.arange(3)   # theta is all but log(noise)

        # manually override starting values to accelerate convergence
        hyp[0] = -1.0       # mu
        hyp[1] = -4.0       # var
        hyp[2] = -2.0       # lenscale
        hyp[3] = -2.0       # noise

        return hyp

    def kernel(self, x, xp, hyp):
        """
        Vectorized implementation of a radial basis function kernel.

        :param x: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param xp: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [nxn numpy array] of kernel values between all n^2 pairs of points in (x, xp)
        """
        output_scale = np.exp(hyp[1])
        lengthscales = np.exp(hyp[2])
        diffs = np.expand_dims(x / lengthscales, 1) - \
                np.expand_dims(xp / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def likelihood(self, hyp):
        """
        Compute the negative log-marginal likelihood of model given observations (self.X, self.y) and hyperparameters.

        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [scalar] negative log-marginal likelihood of model
        """
        X = self.X
        mean = np.exp(hyp[0])
        y = self.y - mean

        N = y.shape[0]

        logsigma_n = hyp[-1]
        sigma_n = np.exp(logsigma_n)

        theta = hyp[self.idx_theta]

        K = self.kernel(X, X, theta) + np.eye(N) * sigma_n
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)
        self.L = L

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        NLML = 0.5 * np.matmul(np.transpose(y), alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5 * np.log(2. * np.pi) * N
        return NLML[0, 0]

    def train(self):
        """
        Trains hyperparameters of GP model by minimizing the negative log-marginal likelihood on given data.
        Prints progress of training at each step.
        For best training results, use with 100-300 training points. Kernel computations are O(n^2) leading to fast
        growth, but small training sets may not lead to reliable hyperparameter inference.

        :return: None
        """
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
                          method='Newton-CG', callback=self.callback)
        self.hyp = result.x

    def predict(self, X_star):
        """
        Return posterior mean and variance conditioned on provided self.X, self.y data
        at a set of test points specified in X_star.

        :param X_star: [nxD numpy array] of test points at which to predict (D is number of dimensions of input space)
        :return: [2-value tuple] of
            [nx1 numpy array] of mean predictions at points in X_star
            [nxn numpy array] of covariance prediction of points in X_star
            [nx1 numpy array] of diagonal of covariance matrix
        """
        X = self.X
        mean = np.exp(self.hyp[0])
        y = self.y - mean

        L = self.L

        theta = self.hyp[self.idx_theta]

        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        pred_u_star = np.matmul(psi, alpha)
        pred_u_star += mean

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi, beta)

        # in our use case, ensure predictions are always positive
        return pred_u_star.clip(min=0), var_u_star, np.diag(var_u_star)

    def callback(self, params):
        """
        Callback evaluated in hyperparameter training process. Computes and displays current negative log-marginal
        likelihood of model.

        :param params: [1xk numpy array] of hyperparameters of GP (k is number of hyperparameters)
        :return: None
        """
        print(f"Log likelihood = {self.likelihood(params)}")

    def update_info(self, X_new, y_new):
        """
        Update model with new X observation points and y observation values. Recompute Cholesky decomposition of
        covariance matrix stored in model.

        :param X_new: [nxD numpy array] of observation points (D is number of dimensions of input space)
        :param y_new: [nx1 numpy array] of observation values
        :return: None
        """
        self.X = X_new
        X = X_new

        self.y = y_new
        y = y_new

        hyp = self.hyp

        N = y.shape[0]

        logsigma_n = hyp[-1]
        sigma_n = np.exp(logsigma_n)

        theta = hyp[self.idx_theta]

        K = self.kernel(X, X, theta) + np.eye(N) * sigma_n
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)
        self.L = L

    def update(self, X_addition=None, y_addition=None):
        """
        Update model with additional X observation points and y observation values. Recompute Cholesky decomposition of
        covariance matrix stored in model.
        Enable delayed update if self.gossip is true by storing into queue and updating in delayed fashion.
        Updates immediately if iteration is passed as a negative number.

        :param X_addition: [nxD numpy array] of observation points to be added (D is number of dims of input space)
        :param y_addition: [nx1 numpy array] of observation values to be added
        :return: None
        """
        if X_addition is not None and len(X_addition) > 0:
            self.X = np.vstack((self.X, X_addition.reshape(-1, 2)))     # in our case, X is always 2D column
            self.y = np.vstack((self.y, y_addition.reshape(-1, 1)))     # in our case, y is always 1D column
        self.update_info(self.X, self.y)


class MFGP:
    """
    A multi-fidelity (2-level) Gaussian Process class capable of hyperparameter inference and mean/variance prediction
    """

    def __init__(self, X_L, y_L, X_H, y_H):
        """
        Initialize the MFGP class.

        :param X_L: [nxD numpy array] of lofi observation points (D is number of dimensions of input space)
        :param y_L: [nxD numpy array] of lofi observation values
        :param X_H: [nxD numpy array] of hifi observation points (D is number of dimensions of input space)
        :param y_H: [nxD numpy array] of hifi observation values
        """
        self.X_L = X_L
        self.y_L = y_L
        self.X_H = X_H
        self.y_H = y_H
        self.L = np.empty([0, 0])
        self.idx_theta_L = np.empty([0, 0])
        self.idx_theta_H = np.empty([0, 0])

        self.hyp = self.init_params()
        self.jitter = 1e-8
        self.likelihood(self.hyp)

    def __str__(self):
        """
        Represent the model as a string.

        :return: string representation of GP model
        """
        ehyp = np.exp(self.hyp)
        return f"****************************************\n" \
               f"MFGP\n" \
               f"X_L shape = {self.X_L.shape}\n" \
               f"X_H shape = {self.X_H.shape}\n" \
               f"log(hyp) = [log(mu_L)={self.hyp[0]}, log(var_L)={self.hyp[1]}, log(lenscale_L)={self.hyp[2]}, " \
               f"log(mu_H)={self.hyp[3]}, log(var_H)={self.hyp[4]}, log(lenscale_H)={self.hyp[5]}, " \
               f"log(rho)={self.hyp[6]}], log(noise_L)={self.hyp[7]}, log(noise_H)={self.hyp[8]}]\n" \
               f"hyp = [mu_L={ehyp[0]}, var_L={ehyp[1]}, lenscale_L={ehyp[2]}, " \
               f"mu_H={ehyp[3]}, var_H={ehyp[4]}, lenscale_H={ehyp[5]}, " \
               f"rho={ehyp[6]}], noise_L={ehyp[7]}, noise_H={ehyp[8]}]\n" \
               f"likelihood = {self.likelihood(self.hyp)}\n" \
               f"****************************************\n"

    def init_params(self):
        """
        Initialize hyperparameters of the GP.

        :return: [1xk numpy array] of GP hyperparameters as initialized (k is number of hyperparameters)
        """
        hyp = np.ones(9)    # log(mu_L, var_L, lenscale_L, mu_H, var_H, lenscale_H, rho, noise_L, noise_H)
        self.idx_theta_L = np.arange(3)         # log(mu_L, var_L, lenscale_L
        self.idx_theta_H = np.arange(3, 6)      # log(mu_H, var_H, lenscale_H

        # manually override starting values to accelerate convergence
        hyp[0] = hyp[3] = -2.0       # mu_L, mu_H
        hyp[1] = hyp[4] = -4.0       # var_L, var_H
        hyp[2] = hyp[5] = -2.0       # lenscale_L, lenscale_H
        hyp[6] = -0.5                # rho
        hyp[7] = hyp[8] = -2.0       # noise_L, noise_H
        return hyp

    def kernel(self, x, xp, hyp):
        """
        Vectorized implementation of a radial basis function kernel.

        :param x: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param xp: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [nxn numpy array] of kernel values between all n^2 pairs of points in (x, xp)
        """
        output_scale = np.exp(hyp[1])
        lengthscales = np.exp(hyp[2])
        diffs = np.expand_dims(x / lengthscales, 1) - \
                np.expand_dims(xp / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def likelihood(self, hyp):
        """
        Compute the negative log-marginal likelihood of model given observations (self.X, self.y) and hyperparameters.

        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [scalar] negative log-marginal likelihood of model
        """
        rho = np.exp(hyp[-3])
        sigma_n_L = np.exp(hyp[-2])
        sigma_n_H = np.exp(hyp[-1])
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]
        mean_L = np.exp(theta_L[0])
        mean_H = rho * mean_L + np.exp(theta_H[0])

        X_L = self.X_L
        y_L = self.y_L
        X_H = self.X_H
        y_H = self.y_H

        y_L = y_L - mean_L
        y_H = y_H - mean_H

        y = np.vstack((y_L, y_H))

        NL = y_L.shape[0]
        NH = y_H.shape[0]
        N = y.shape[0]

        K_LL = self.kernel(X_L, X_L, theta_L) + np.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(X_L, X_H, theta_L)
        K_HH = rho ** 2 * self.kernel(X_H, X_H, theta_L) + \
               self.kernel(X_H, X_H, theta_H) + np.eye(NH) * sigma_n_H
        K = np.vstack((np.hstack((K_LL, K_LH)),
                       np.hstack((K_LH.T, K_HH))))
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)
        self.L = L

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        NLML = 0.5 * np.matmul(np.transpose(y), alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5 * np.log(2. * np.pi) * N
        return NLML[0, 0]

    def train(self):
        """
        Trains hyperparameters of GP model by minimizing the negative log-marginal likelihood on given data.
        Prints progress of training at each step.
        For best training results, use with 100-300 training points in each fidelity. Kernel computations are O(n^2)
        leading to fast growth, but small training sets may not lead to reliable hyperparameter inference.

        :return: None
        """
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
                          method='L-BFGS-B', callback=self.callback)
        self.hyp = result.x

    def predict(self, X_star):
        """
        Return posterior mean and variance conditioned on provided self.X, self.y data
        at a set of test points specified in X_star.

        :param X_star: [nxD numpy array] of test points at which to predict (D is number of dimensions of input space)
        :return: [2-value tuple] of
            [nx1 numpy array] of mean predictions at points in X_star
            [nxn numpy array] of covariance prediction of points in X_star (diagonal is variance at points in X_star)
            [nx1 numpy array] of diagonal of covariance matrix
        """
        hyp = self.hyp
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]
        rho = np.exp(hyp[-3])
        mean_L = np.exp(theta_L[0])
        mean_H = rho * mean_L + np.exp(theta_H[0])

        X_L = self.X_L
        y_L = self.y_L - mean_L
        X_H = self.X_H
        y_H = self.y_H - mean_H
        L = self.L

        y = np.vstack((y_L, y_H))

        psi1 = rho * self.kernel(X_star, X_L, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, X_H, theta_L) + \
               self.kernel(X_star, X_H, theta_H)
        psi = np.hstack((psi1, psi2))

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        pred_u_star = mean_H + np.matmul(psi, alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                     self.kernel(X_star, X_star, theta_H) - np.matmul(psi, beta)

        # in our use case, ensure predictions are always positive
        return pred_u_star.clip(min=0), var_u_star, np.diag(var_u_star)

    def callback(self, params):
        """
        Callback evaluated in hyperparameter training process. Computes and displays current negative log-marginal
        likelihood of model.

        :param params: [1xk numpy array] of hyperparameters of GP (k is number of hyperparameters)
        :return: None
        """
        print(f"Log likelihood = {self.likelihood(params)}")

    def update_info(self, X_L_new, y_L_new, X_H_new, y_H_new):
        """
        Update model with new X observation points and y observation values. Recompute Cholesky decomposition of
        covariance matrix stored in model.

        :param X_L_new: [nxD numpy array] of lofi observation points (D is number of dimensions of input space)
        :param y_L_new: [nx1 numpy array] of lofi observation values
        :param X_H_new: [nxD numpy array] of hifi observation points (D is number of dimensions of input space)
        :param y_H_new: [nx1 numpy array] of hifi observation values
        :return: None
        """
        self.X_L = X_L_new
        self.y_L = y_L_new
        self.X_H = X_H_new
        self.y_H = y_H_new

        hyp = self.hyp
        rho = np.exp(hyp[-3])
        sigma_n_L = np.exp(hyp[-2])
        sigma_n_H = np.exp(hyp[-1])
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]

        X_L = self.X_L
        X_H = self.X_H

        NL = X_L.shape[0]
        NH = X_H.shape[0]
        N = NL + NH

        K_LL = self.kernel(X_L, X_L, theta_L) + np.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(X_L, X_H, theta_L)
        K_HH = rho ** 2 * self.kernel(X_H, X_H, theta_L) + \
               self.kernel(X_H, X_H, theta_H) + np.eye(NH) * sigma_n_H
        K = np.vstack((np.hstack((K_LL, K_LH)),
                       np.hstack((K_LH.T, K_HH))))
        self.L = np.linalg.cholesky(K + np.eye(N) * self.jitter)

    def update(self, X_H_addition=None, y_H_addition=None):
        """
        Update model with additional hifi observation points and values. Recompute Cholesky decomposition of
        covariance matrix stored in model.
        Enable delayed update if self.gossip is true by storing into queue and updating in delayed fashion.
        Updates immediately if iteration is passed as a negative number.


        :param X_H_addition: [nxD numpy array] of hifi observation points to be added (D is num of dims of input space)
        :param y_H_addition: [nx1 numpy array] of hifi observation values to be added
        :return: None
        """
        if X_H_addition is not None and len(X_H_addition) > 0:
            self.X_H = np.vstack((self.X_H, X_H_addition.reshape(-1, 2)))   # in our case, X is always 2D column
            self.y_H = np.vstack((self.y_H, y_H_addition.reshape(-1, 1)))   # in our case, y is always 1D column
        self.update_info(self.X_L, self.y_L, self.X_H, self.y_H)

