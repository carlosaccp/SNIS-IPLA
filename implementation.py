import numpy as np

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
        theta_next = self.theta - self.gamma*np.sum([SNIS_weights[i] * self.grad_U_theta_fn(self.theta, self.X[:,i]) for i in range(N)]) + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None
    
class fast_SNIS_IPLA:
    def __init__(self, theta0, X0, p, grad_p, grad_U_X, gamma=0.01, test=False, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.theta = theta0
        self.X = X0
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.grad_U_X = grad_U_X
        self.kwargs = kwargs
        self.p = p
        self.grad_theta_p = grad_p
        self.test=test

    def p_fn(self, theta, x):
        return self.p(theta, x, **self.kwargs)
    
    def grad_theta_p_fn(self, theta, x):
        return self.grad_theta_p(theta, x, **self.kwargs)

    def grad_U_X_fn(self, theta, x):
        return self.grad_U_X(theta, x, **self.kwargs)
    
    def grad_p_fn(self, theta, x):
        return self.grad_theta_p(theta, x, **self.kwargs)
    
    def iterate(self):
        D, N = np.shape(self.X)
        update_term = np.sum([self.grad_theta_p_fn(self.theta, self.X[:,k]) for k in range(N)])/np.sum([self.p_fn(self.theta, self.X[:,k]) for k in range(N)])
        theta_next = self.theta + self.gamma * update_term + np.sqrt(2*self.gamma/N) * np.random.normal(size=1)
        X_next = self.X - self.gamma * self.grad_U_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None