import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GaussianProcess:
    def __init__(self, l=1.0, sigma_f=1.0, std=0.05, domain_range=(-4, 4), d=100):
        self.l = l
        self.sigma_f = sigma_f
        self.std = std #maybe turn this into a child of Bayesian Optimization so it can inherit the below two variables
        self.domain = np.linspace(domain_range[0], domain_range[1], d).reshape(-1, 1)

    def rbf_kernel(self, x1, x2):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma_f**2 * np.exp(-0.5 * sqdist / self.l**2)
    
    def matern_kernel(self, x1, x2, l):

        return (1 + (np.sqrt(5)/l )* self.euclidian(x1,x2) + (5/3*l)*self.euclidian(x1,x2)**2)*np.exp((-np.sqrt(5)/l)*self.euclidian(x1,x2))
    
    def euclidian(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def posterior_update(self, X, Y, domain, std):
        K = self.rbf_kernel(X, X) + (std**2) * np.eye(len(X))
        K_s = self.rbf_kernel(X, domain)
        K_ss = self.rbf_kernel(domain, domain)
        posterior_mean = K_s.T.dot(np.linalg.inv(K)).dot(Y)
        posterior_cov = K_ss - K_s.T.dot(np.linalg.inv(K)).dot(K_s)
        return posterior_mean, posterior_cov


class BayesianOptimization:
    """
    Class for performing Bayesian Optimization.

    Attributes:
        std (float): Standard deviation for noise in the objective function.

    Methods:
        call_function(x, function='sin'): Evaluates the objective function at a given point.
        initialize_query(n=10, d=100): Initializes the query points and the domain.
        upper_confidence_bound(mean, variance, kappa=2): Computes the upper confidence bound.
        run_iterations(T, plot=False): Runs the Bayesian Optimization iterations.
        plot_iteration(iteration, domain, mean, variance, X, Y, X_new, index): Plots the iteration results.
        getDomain(): Returns the domain of the optimization problem.
    """

    def __init__(self, std=0.05):
        self.std = std

    def call_function(self, x, function='sin'):
        """
        Evaluates the objective function at a given point.

        Args:
            x (float): The input value.
            function (str): The name of the objective function. Default is 'sin'.

        Returns:
            float: The value of the objective function at the given point.
        
        Raises:
            ValueError: If an invalid function argument is provided.
        """
        if function == 'sin':
            return np.sin(x) + np.random.normal(0, self.std)
        elif function == 'levy':
            return np.sin(3*np.pi*x)**2 + (x-1)**2 * (1 + np.sin(3*np.pi*x)**2)
        else:
            raise ValueError("Invalid function argument. Please choose 'sin' or 'levy'.")

    def initialize_query(self, n=10, d=100):
        """
        Initializes the query points and the domain.

        Args:
            n (int): The number of initial query points. Default is 10.
            d (int): The number of points in the domain. Default is 100.

        Returns:
            tuple: A tuple containing the initialized query points, the corresponding function values, and the domain.
        """
        X = np.random.uniform(-3, 3, size=(n, 1))
        Y = self.call_function(X)
        self.domain = np.linspace(-4, 4, d).reshape(-1, 1)
        return X, Y, self.domain

    def upper_confidence_bound(self, mean, variance, kappa=2):
        """
        Computes the upper confidence bound.

        Args:
            mean (ndarray): The mean values of the Gaussian process.
            variance (ndarray): The variance values of the Gaussian process.
            kappa (float): The exploration parameter. Default is 2.

        Returns:
            ndarray: The upper confidence bound values.
        """
        return mean.flatten() + kappa * np.sqrt(np.diag(variance))

    def run_iterations(self, T, plot=False):
        """
        Runs the Bayesian Optimization iterations.

        Args:
            T (int): The number of iterations to run.
            plot (bool): Whether to plot the iteration results. Default is False.

        Returns:
            tuple: A tuple containing the query points, the corresponding function values, the maximum function value, and the index of the maximum value.
        """
        X, Y, domain = self.initialize_query()
        gp = GaussianProcess()
        max_value = float('-inf')
        max_index = None
        for i in range(T):
            mean, variance = gp.posterior_update(X, Y, domain, self.std)
            ucb_values = self.upper_confidence_bound(mean, variance)
            index = np.argmax(ucb_values)
            X_new = domain[index]
            Y_new = self.call_function(X_new).flatten()
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            if Y_new > max_value:
                max_value = Y_new
                max_index = index
            if plot:
                self.plot_iteration(i+1, domain, mean, variance, X, Y, X_new, index)
        return X, Y, max_value, max_index

    def plot_iteration(self, iteration, domain, mean, variance, X, Y, X_new, index):
        """
        Plots the iteration results.

        Args:
            iteration (int): The iteration number.
            domain (ndarray): The domain of the optimization problem.
            mean (ndarray): The mean values of the Gaussian process.
            variance (ndarray): The variance values of the Gaussian process.
            X (ndarray): The query points.
            Y (ndarray): The corresponding function values.
            X_new (float): The next query point.
            index (int): The index of the next query point.
        """
        plt.figure(figsize=(12,6))
        plt.plot(domain, self.call_function(domain), 'y:', label='True Function')
        plt.plot(domain, mean, 'k', lw=2, zorder=9, label='GP Mean')
        plt.fill_between(domain.flatten(), mean.flatten() - 2 * np.sqrt(np.diag(variance)), mean.flatten() + 2 * np.sqrt(np.diag(variance)), alpha=0.2, color='k')
        plt.scatter(X, Y, c='r', zorder=10, edgecolors=(0, 0, 0), label='Queried Points')
        plt.scatter(X_new, mean[index], c='r', marker='*', s=100, label='Next Query Point')
        plt.xlabel("X")
        plt.ylabel("Objective Function")
        plt.title(f'Iteration {iteration}')
        plt.legend()
        plt.grid(True)
        # plt.show()

    def getDomain(self):
        """
        Returns the domain of the optimization problem.

        Returns:
            ndarray: The domain of the optimization problem.
        """
        return self.domain

if __name__ == "__main__":
    bo = BayesianOptimization(std=0.05)
    _, _, max_value, max_index = bo.run_iterations(T=5, plot=True)
    plt.show()
    domain = bo.getDomain()
    print(f"Maximum Value: {max_value}")
    print(f"Maximum Index: {domain[max_index]}")
    
    def f(x):
        return np.sin(x)
    def f2(x):

        return np.sin(3*np.pi*x)**2 + (x-1)**2 * (1 + np.sin(3*np.pi*x)**2)
    
    opt = minimize(lambda x: -f(x),x0=0, bounds=[(-5, 5)])

    print(f"True Maximum Value: {f(opt.x)}")
    print(f"True Maximum Index: {opt.x}")



