import numpy as np
import scipy as sp

def find_root(A, tol = 1e-12):
    # finds square root of a positive semi-definite symmetric matrix A
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues[eigenvalues < tol] = 0 # matrix should be pd, some numerical errors give very small negative eigenvalues
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ (eigenvectors).T

def min_jitter_chol(A, jitter = 1e-20):
    # finds the Cholesky decomposition of A + jitter*I
    # A should be symmetric positive semi-definite
    # jitter is the smallest value such that A + jitter*I is positive definite
    # returns the final jitter and Cholesky decomposition of A + jitter*I

    try:
        cholesky = np.linalg.cholesky(A)
        return 0, cholesky
    except Exception:
        while jitter < 1:
            try:
                cholesky = np.linalg.cholesky(A + jitter*np.eye(np.shape(A)[0]))
                #print("Matrix not pd")
                return jitter, cholesky
            except Exception:
                jitter *= 10
        raise Exception("Failed")

class IPLA:
    def __init__(self, theta0, X0, grads_U, gamma=0.01, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.theta = theta0
        self.X = X0
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.ave_grad_U_theta, self.grad_U_X = grads_U
        self.kwargs = kwargs

    def grad_U_X_fn(self, theta, x):
        return self.grad_U_X(theta, x, **self.kwargs)
    
    def ave_grad_U_theta_fn(self, theta, x):
        return self.ave_grad_U_theta(theta, x, **self.kwargs)
    
    def iterate(self):
        D, N = np.shape(self.X)
        theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X) + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None

class PGD:
    def __init__(self, theta0, X0, grads_U, gamma=0.01, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.theta = theta0
        self.X = X0
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.ave_grad_U_theta, self.grad_U_X = grads_U
        self.kwargs = kwargs

    def grad_U_X_fn(self, theta, x):
        return self.grad_U_X(theta, x, **self.kwargs)
    
    def ave_grad_U_theta_fn(self, theta, x):
        return self.ave_grad_U_theta(theta, x, **self.kwargs)
    
    def iterate(self):
        D, N = np.shape(self.X)
        theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None


class SNIS_IPLA:
    def __init__(self, theta0, X0, U, grads_U, gamma=0.01, test=False, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.theta = theta0
        self.X = X0
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.grad_U_theta, self.grad_U_X = grads_U
        self.kwargs = kwargs
        self.U = U
        self.test=test
        self.ESS_arr = []

    def grad_U_X_fn(self, theta, x):
        return self.grad_U_X(theta, x, **self.kwargs)
    
    def grad_U_theta_fn(self, theta, x):
        return self.grad_U_theta(theta, x, **self.kwargs)
    
    def U_fn(self, theta, x):
        return self.U(theta, x, **self.kwargs)
    
    def iterate(self):
        D, N = np.shape(self.X)
        if not self.test:
            U_fns = [-self.U_fn(self.theta, self.X[:,k]) for k in range(N)] 
            U_fns_stable = U_fns - max(U_fns)
            SNIS_weights = np.exp(U_fns_stable)/sum(np.exp(U_fns_stable))
        else:
            SNIS_weights = [1/N] * N

        ESS = 1/np.sum([w**2 for w in SNIS_weights])
        self.ESS_arr.append(ESS)
        theta_next = self.theta - self.gamma*np.sum([SNIS_weights[i] * self.grad_U_theta_fn(self.theta, self.X[:,i]) for i in range(N)]) + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None

class SVGD_EM:
    def __init__(self, theta0, X0, grads_U, kernel, grad_kernel_x2, gamma=0.01, noise=True, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.noise = noise
        self.theta = theta0
        self.j_iter = []
        self.X = X0
        self.iter = 0
        self.kernel = kernel
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.grad_kernel_x2 = grad_kernel_x2
        self.gamma = gamma
        self.ave_grad_U_theta, self.grad_U_X = grads_U
        self.kwargs = kwargs

    def kernel_fn(self, x1, x2):
        return self.kernel(x1, x2)
    
    def grad_kernel_x2_fn(self, x1, x2):
        return self.grad_kernel_x2(x1, x2)
    
    def grad_U_X_fn(self, theta, x):
        return self.grad_U_X(theta, x, **self.kwargs)
    
    def ave_grad_U_theta_fn(self, theta, x):
        return self.ave_grad_U_theta(theta, x, **self.kwargs)
    
    def mass_matrix(self, X):
        D, N = np.shape(X)
        #mass_matrix_components = [[self.kernel_fn(X[:,i], X[:,j])*np.eye(D) for i in range(N)] for j in range(N)]
        Xbar = X.reshape((-1,1))
        mass_matrix_components = [[self.kernel_fn(Xbar[i], Xbar[j])*np.eye(D) for i in range(N)] for j in range(N)]
        mass_m = (1/N) * np.bmat(mass_matrix_components)
        return mass_m
    
    def iterate(self):
        D, N = np.shape(self.X)
        if self.noise:
            theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X) + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        else:
            theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X)
        update_term = np.zeros((D, N)) 
        if self.noise:
            mass_m = self.mass_matrix(self.X)
            jitter, root_mass_matrix = min_jitter_chol(mass_m)
            #print("Jitter:", jitter) if jitter > 0 else None
            if jitter > 0:
                self.j_iter.append(self.iter)
            root_mass_matrix = np.sqrt(2) * root_mass_matrix
            #print("Root Mass Matrix:", root_mass_matrix)  # Add this line for debugging
        for i in range(N):
            col = np.zeros((D, 1))
            for j in range(N):
                A = -self.gamma * self.kernel_fn(self.X[:,i], self.X[:,j]) * self.grad_U_X_fn(theta_next, self.X[:,j].reshape(-1,1))
                B = self.gamma * self.grad_kernel_x2_fn(self.X[:,i], self.X[:,j]).reshape(-1,1)
                col += (A + B) / N
            update_term[:,i] = col.reshape(-1)
        if self.noise:
            noise = (root_mass_matrix @ np.random.normal(size=(N*D, 1))) * np.sqrt(2*self.gamma)
            reshaped_noise = noise.reshape((D, N))
            update_term += reshaped_noise
        X_next = self.X + update_term 
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        self.iter += 1
        return None
