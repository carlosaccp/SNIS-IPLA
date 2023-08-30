import numpy as np

class IPLA:
    def __init__(self, theta0, X0, grads_u, gamma=0.01, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.theta = theta0
        self.X = X0
        self.N = X0.shape[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.grad_u_theta, self.grad_u_X = grads_u
        self.kwargs = kwargs

    def grad_u_X_fn(self, theta, x):
        return self.grad_u_X(theta, x, **self.kwargs)
    
    def grad_u_theta_fn(self, theta, x):
        return self.grad_u_theta(theta, x, **self.kwargs)
    
    def iterate(self):
        N = self.N
        theta_next = self.theta + self.gamma/N * np.sum([self.grad_u_theta_fn(self.theta, self.X[:,i]) for i in range(N)]) + np.sqrt(2*self.gamma/N) * np.random.normal(size=self.theta.shape)  
        X_next = np.zeros_like(self.X)
        for i in range(N):
            X_next[:,i] = self.X[:,i] + self.gamma * self.grad_u_X_fn(theta_next, self.X[:,i]) + np.sqrt(2*self.gamma) * np.random.normal(size=self.X[:,i].shape)
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None
