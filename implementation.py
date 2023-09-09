import numpy as np
import scipy as sp

def find_root(A, tol = 1e-12):
    """
    Finds the root of a matrix A, i.e. the matrix X such that X^2 = A. Accounts for numerical errors.

    Inputs:
    A - matrix to find root of
    tol - tolerance for numerical errors

    Outputs:
    X - root of A
    """
    # finds square root of a positive semi-definite symmetric matrix A
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues[eigenvalues < tol] = 0 # matrix should be pd, some numerical errors give very small negative eigenvalues
    X = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ (eigenvectors).T
    return X

def min_jitter_chol(A, jitter = 1e-20):
    """
    Finds the Cholesky decomposition of a matrix A. If A is not positive definite, adds jitter to the diagonal until it is.
    Fails if jitter is too large.

    Inputs:
    A - matrix to find Cholesky decomposition of
    jitter - amount to add to diagonal of A if it is not positive definite

    Outputs:
    jitter - amount added to diagonal of A (0 if A is pd)
    cholesky - Cholesky decomposition of A
    """

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

class Particle_Optimiser:
    """
    Parent class for particle optimisation algorithms
    """

    def __init__(self, theta0, X0, gamma=0.01, **kwargs):
        """
        Inputs:
        theta0 - initial parameter
        X0 - initial data. This is a vector of size N
        gamma - step size
        **kwargs - additional arguments for the gradient functions
        """
        self.theta = theta0
        self.X = X0
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.kwargs = kwargs
    
    def iterate(self):
        """
        Iterates the algorithm once
        """
        raise NotImplementedError

class IPLA(Particle_Optimiser):
    """
    IPLA optimisation algorithm
    """
    def __init__(self, theta0, X0, grads_U, gamma=0.01, **kwargs):
        """
        Inputs:

        theta0 - initial parameter
        X0 - initial data. This is a vector of size N
        grads_U - tuple of functions (ave_grad_U_theta, grad_U_X) where ave_grad_U_theta is the average gradient of U wrt theta and grad_U_X is the gradient of U wrt X
        gamma - step size
        **kwargs - additional arguments for the gradient functions
        """

        super().__init__(theta0, X0, gamma, **kwargs)
        self.ave_grad_U_theta, self.grad_U_X = grads_U

    def grad_U_X_fn(self, theta, X):
        """
        Helper function to evaluate the gradient of U wrt X

        Inputs:
        theta - parameter
        X - data matrix

        Outputs:
        grad_U_X - gradient of U wrt X
        """

        return self.grad_U_X(theta, X, **self.kwargs)
    
    def ave_grad_U_theta_fn(self, theta, X):
        """
        Helper function to evaluate the average gradient of U wrt theta

        Inputs:
        theta - parameter
        X - data matrix
        """

        return self.ave_grad_U_theta(theta, X, **self.kwargs)
    
    def iterate(self):
        """
        Performs one iteration of the IPLA algorithm and updates the parameters

        Inputs:
        None

        Outputs:
        None
        """

        D, N = np.shape(self.X)
        theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X) + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None

class PGD(Particle_Optimiser):
    """
    PGD optimisation algorithm
    """
    def __init__(self, theta0, X0, grads_U, gamma=0.01, **kwargs):
        """
        Inputs:

        theta0 - initial parameter
        X0 - initial data. This is a vector of size N
        grads_U - tuple of functions (ave_grad_U_theta, grad_U_X) where ave_grad_U_theta is the average gradient of U wrt theta and grad_U_X is the gradient of U wrt X
        gamma - step size
        **kwargs - additional arguments for the gradient functions
        """

        super().__init__(theta0, X0, gamma, **kwargs)
        self.ave_grad_U_theta, self.grad_U_X = grads_U

    def grad_U_X_fn(self, theta, X):
        """
        Helper function to evaluate the gradient of U wrt X

        Inputs:
        theta - parameter
        X - data matrix

        Outputs:
        grad_U_X - gradient of U wrt X
        """

        return self.grad_U_X(theta, X, **self.kwargs)
    
    def ave_grad_U_theta_fn(self, theta, X):
        """
        Helper function to evaluate the average gradient of U wrt theta

        Inputs:
        theta - parameter
        X - data matrix

        Outputs:
        ave_grad_U_theta - average gradient of U wrt theta
        """

        return self.ave_grad_U_theta(theta, X, **self.kwargs)
    
    def iterate(self):
        """
        Performs one iteration of the PGD algorithm and updates the parameters
        """

        D, N = np.shape(self.X)
        theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None


class SNIS_IPLA(Particle_Optimiser):
    """
    SNIS-IPLA (Self-Normalising Importance Sampling IPLA) optimisation algorithm
    """

    def __init__(self, theta0, X0, U, grads_U, gamma=0.01, test=False, **kwargs):
        """
        Inputs:

        theta0 - initial parameter
        X0 - initial data. This is a vector of size N
        U - function to evaluate U
        grads_U - tuple of functions (grad_U_theta, grad_U_X) where grad_U_theta is the gradient of U wrt theta and grad_U_X is the gradient of U wrt X
        gamma - step size
        test - whether or not to use uniform 1/N weights to test the algorithm's functionality
        **kwargs - additional arguments for the gradient functions
        """

        super().__init__(theta0, X0, gamma, **kwargs)
        self.U = U
        self.grad_U_theta, self.grad_U_X = grads_U
        self.test=test
        self.ESS_arr = []

    def grad_U_X_fn(self, theta, X):
        """
        Helper function to evaluate the gradient of U wrt X

        Inputs:
        theta - parameter
        X - data matrix

        Outputs:
        grad_U_X - gradient of U wrt X
        """

        return self.grad_U_X(theta, X, **self.kwargs)
    
    def grad_U_theta_fn(self, theta, Xi):
        """
        Helper function to evaluate the gradient of U wrt theta

        Inputs:
        theta - parameter
        Xi - data point

        Outputs:
        grad_U_theta - gradient of U wrt theta
        """

        return self.grad_U_theta(theta, Xi, **self.kwargs)
    
    def U_fn(self, theta, X):
        """
        Helper function to evaluate U

        Inputs:
        theta - parameter
        X - data matrix

        Outputs:
        U - U
        """

        return self.U(theta, X, **self.kwargs)
    
    def iterate(self):
        """
        Performs one iteration of the SNIS-IPLA algorithm and updates the parameters
        """

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

class SVGD_EM(Particle_Optimiser):
    """
    SVGD-EM (Stein Variational Gradient Descent Expectation Maximisation) optimisation algorithm
    """
    def __init__(self, theta0, X0, grads_U, kernel, grad_kernel_x2, gamma=0.01, noise=True, **kwargs):
        """
        Inputs:

        theta0 - initial parameter
        X0 - initial data. This is a vector of size N
        grads_U - tuple of functions (ave_grad_U_theta, grad_U_X) where ave_grad_U_theta is the average gradient of U wrt theta and grad_U_X is the gradient of U wrt X
        kernel - kernel function
        grad_kernel_x2 - gradient of kernel wrt x2
        gamma - step size
        noise - whether or not to add noise to the algorithm
        **kwargs - additional arguments for the gradient functions
        """

        super().__init__(theta0, X0, gamma, **kwargs)
        self.kernel = kernel
        self.iter = 0
        self.noise = noise
        self.j_iter = []
        self.grad_kernel_x2 = grad_kernel_x2
        self.ave_grad_U_theta, self.grad_U_X = grads_U

    def kernel_fn(self, x1, x2):
        """
        Helper function to evaluate the kernel

        Inputs:
        x1 - first input
        x2 - second input

        Outputs:
        kernel - kernel
        """

        return self.kernel(x1, x2)
    
    def grad_kernel_x2_fn(self, x1, x2):
        """
        Helper function to evaluate the gradient of the kernel wrt x2

        Inputs:
        x1 - first input
        x2 - second input

        Outputs:
        grad_kernel_x2 - gradient of kernel wrt x2
        """

        return self.grad_kernel_x2(x1, x2)
    
    def grad_U_X_fn(self, theta, Xi):
        """
        Helper function to evaluate the gradient of U wrt X

        Inputs:
        theta - parameter
        Xi - data point

        Outputs:
        grad_U_X - gradient of U wrt X
        """

        return self.grad_U_X(theta, Xi, **self.kwargs)
    
    def ave_grad_U_theta_fn(self, theta, X):
        """
        Average gradient of U wrt theta

        Inputs:
        theta - parameter
        X - data matrix

        Outputs:
        ave_grad_U_theta - average gradient of U wrt theta
        """

        return self.ave_grad_U_theta(theta, X, **self.kwargs)
    
    def mass_matrix(self, X):
        """
        Helper function to evaluate the block mass matrix

        Inputs:
        X - data matrix

        Outputs:
        mass_m - block mass matrix
        """

        D, N = np.shape(X)
        #mass_matrix_components = [[self.kernel_fn(X[:,i], X[:,j])*np.eye(D) for i in range(N)] for j in range(N)]
        Xbar = X.reshape((-1,1))
        mass_matrix_components = [[self.kernel_fn(Xbar[i], Xbar[j])*np.eye(D) for i in range(N)] for j in range(N)]
        mass_m = (1/N) * np.bmat(mass_matrix_components)
        return mass_m
    
    def iterate(self):
        """
        Performs one iteration of the SVGD-EM algorithm and updates the parameters
        """
        D, N = np.shape(self.X)
        if self.noise:
            theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X) + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        else:
            theta_next = self.theta - self.gamma * self.ave_grad_U_theta_fn(self.theta, self.X)
        update_term = np.zeros((D, N)) 
        if self.noise:
            mass_m = self.mass_matrix(self.X)
            jitter, root_mass_matrix = min_jitter_chol(mass_m)
            self.j_iter.append(jitter)
            root_mass_matrix = np.sqrt(2) * root_mass_matrix
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
