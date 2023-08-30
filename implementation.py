import numpy as np

class IPLA:
    def __init__(self, theta0, X0, grads_u, gamma=0.01, **kwargs):
        # theta0: initial parameter 
        # X0: initial data. This is a vector of size N
        self.theta = theta0
        self.X = X0
        self.N = np.shape(X0)[1]
        self.thetas = [self.theta]
        self.Xs = [self.X]
        self.gamma = gamma
        self.ave_grad_u_theta, self.grad_u_X = grads_u
        self.kwargs = kwargs

    def grad_u_X_fn(self, theta, x):
        return self.grad_u_X(theta, x, **self.kwargs)
    
    def ave_grad_u_theta_fn(self, theta, x):
        return self.ave_grad_u_theta(theta, x, **self.kwargs)
    
    def iterate(self):
        D, N = np.shape(self.X)
        theta_next = self.theta - self.gamma * self.ave_grad_u_theta_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=1)
        X_next = self.X - self.gamma * self.grad_u_X_fn(self.theta, self.X) + np.sqrt(2*self.gamma) * np.random.normal(size=(D, N))
        self.thetas.append(theta_next)
        self.Xs.append(X_next)
        self.theta = theta_next
        self.X = X_next
        return None
