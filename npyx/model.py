from npyx.utils import repr_string

import numpy as np
from scipy.linalg import hankel

import sklearn.linear_model as sk_lin
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

class Linear_closed_form:
    """
    Closed form solver for linear or ridge regression.
    Setting lambda_ to 0 is equivalent
    to a multilinear regression without l2 regularisation.
    """
    def __init__(self, lambda_=1, memory_efficient=False):
        self.lambda_ = lambda_
        self.memory_efficient = memory_efficient
        
    def __repr__(self):
        return f"Linear or Ridge regression model with attributes and methods:\n\t{repr_string(self)}"
    
    def fit(self, X, y):

        I = np.eye(X.shape[1])
        if self.memory_efficient:
            # Solving for X.T @ X + self.lambda_ * I = X.T @ y,
            # prevents from having to do the actual memory-intensive matrix inversion
            self.coef_ = np.linalg.solve(X.T @ X + self.lambda_ * I, X.T @ y)
        else:
            self.coef_ = np.linalg.inv(X.T @ X + self.lambda_ * I) @ X.T @ y
        
    def predict(self, X):
        return (X @ self.coef_).squeeze()
    
class GLM:
    """
    Generalized Linear Model. Simple wrapper for key scikit-learn linear models
    as well as home-baked fast closed-form solvers (set use_closed_form_solution to True).
    ___________________________________________________________________________________________
    Arguments:
        - distribution: pick amongst ['gaussian', 'poisson', 'bernouilli'].
                        Assumed distribution of response variable y.
        - l2_penalty: [0-1], weight applied to l2 regularisation term in cost fuction.
                      See "Model selection procedure" below for more info.
                      If set to 1 or 0 and distribution is gaussian
                          -> Ridge or Lasso. respectively.
                      If 0 < l2_penalty < 1 and distribution is gaussian
                          -> elastic net and l1_penalty = 1-l2_penalty.
        - global_reg_penalty: float, global regularization penalty
                    See "Model selection procedure" below for more info.
                    - if l2_penalty==1 -> ridge regression: penalty on l2, usually called alpha
                    - if l2_penalty==0 lasso: penalty on l1, usually called alpha
                    - if 0 < l2_penalty < 1 -> elastic net: shared penalty
                       between l1 and l2 regularization, usually called lambda.
                       Actually, corresponds to variance of gaussian distribution
                       fit to data (sigma^2).
        - normalize_X: bool, whether to normalize the input data column-wise
                       (predictors/features dimension) before fitting the model.
        - use_closed_form_solution: bool, whether to compute closed form solution
                                    instead of scikit learn. Only valid for
                                    regular linear regression or ridge regression -
                                    in other cases, scikit-learn will be used.
        - memory_efficient: bool, whether to use memory efficient closed_form_solution
                            by approximating matrix inversion with np.solve.
                            Note: solving with np.solve is twice as fast for some reason!!
    ___________________________________________________________________________________________
    Useful methods:
        - fit(X, y, **kwargs):
        
             fits the model parameters and stores them in .beta.
             If use_closed_form_solution is False, simply calls the scikit learn .fit function.
             Any additional parameter can be passed to .fit as kwargs.
             Arguments:
                - X: (n_observations, n_predictors) array.
                - y: (n_observations,) array.
                
        - get_fit_params():
            Returns:
                - the model fit parameters .beta.
                
        - predict(X, evaluate_prediction = True):
            predicts y_pred from some array of predictors X. In essence, that is X @ self.beta.
            Arguments:
                - X: (n_observations, n_predictors) array.
            Returns:
                - y_predicted:  (n_observations,) array. Model prediction.
                
        - evaluate(y_true, y_predicted): computes, prints and returns model metrics.
            Arguments:
                - y_true: (n_observations,) array. Actual data.
                - y_predicted: (n_observations,) array. Model prediction.
            Returns:
                 - mean square error,
                 - r2 and
                 - explained variance
    ___________________________________________________________________________________________
    Useful attributes:
        - .beta: (n_predictors,) array. Model fit parameters once .fit(X, y) has been called.
        - .model: object, scikit-learn or custom model object (in particular has .fit, .predict, .coef_ methods/attributes).
        - .description: str, description of picked model given arguments.
        - .eval: dict, contains 'mse', 'r2' and explained variance 'ev' of model evaluation once .evaluate(y_true, y_predicted) has been called.
    ___________________________________________________________________________________________
    Model selection procedure:
        if distribution == "gaussian":
            if global_reg_penalty == 0:
                This is regular linear regression (ordinary least square)
            else:
                if l2_penalty == 1:
                    This is Ridge regression (pure l2 regularisation)
                elif l2 penalty == 0:
                    This is Lasso regression (pure l1 regularisation)
                elif 0 < l2_penalty < 1:
                    This is ElasticNet (l1_penalty = 1 - l2_penalty)
        elif "bernoulli":
            This is logistic regression.
                if global_reg_penalty == 0:
                    no regularization penalty (LogisticRegression(penalty=None))
                else:
                    global_reg_penalty is applied as
                    LogisticRegression(C = 1/global_reg_penalty)
                    if l2_penalty == 1:
                        pure l2 penalty (LogisticRegression(penalty='l2'))
                    elif l2 penalty == 0:
                        pure l1 penalty (LogisticRegression(penalty='l1'))
                    elif 0 < l2_penalty < 1:
                        both l1 and l2 penalty (LogisticRegression(penalty='elasticnet'))
                        The actual value of l2_penalty is ignored in this case.
        elif "poisson":
            This is a poisson GLM, with/out regularisation.
            scikit-learn PoissonRegressor is sctrictly equivalent to
            TweedieRegressor(power=1, link='log').
            Using regularizers with Poisson GLMs is very new,
            can only be approximated with 'cyclical gradient descent'.
            This is implemented in the R package glmnet,
            and in the python package PyGLMnet. Not worth implementing.
    """
    def __init__(self,
                 distribution = "gaussian",
                 l2_penalty = 1,
                 global_reg_penalty = 1,
                 normalize_X=True,
                 use_closed_form_solution=True,
                 memory_efficient = True,
                 **kwargs):
        
        self.distribution_options = ['gaussian', 'poisson', 'bernouilli']
        assert distribution in self.distribution_options
        self.distribution = distribution
        
        assert global_reg_penalty>=0, "global_reg_penalty should be positive!"
        self.global_reg_penalty = global_reg_penalty
        
        assert 0<=l2_penalty and l2_penalty<=1, "l2_penalty should be between [0-1]!"
        self.l2_penalty = l2_penalty
        self.l1_penalty = 1 - self.l2_penalty
        
        self.use_closed_form_solution = use_closed_form_solution
        self.memory_efficient = memory_efficient
        
        if self.distribution == "gaussian":
            if self.global_reg_penalty == 0:
                if self.use_closed_form_solution:
                    self.description = ("Closed form regular Linear Regression "
                                       "(Gaussian GLM, no regularization)")
                    self.model = \
                    Linear_closed_form(lambda_=self.global_reg_penalty, # that is 0
                                       memory_efficient=self.memory_efficient)
                else:
                    self.description = ("Regular Linear Regression"
                                        "(Gaussian GLM, no regularization)")
                    self.model = \
                    sk_lin.LinearRegression(fit_intercept=False,
                                            **kwargs)
            else:
                if self.l2_penalty == 1:
                    if self.use_closed_form_solution:
                        self.description = ("Closed form Ridge Regression "
                                            "(Gaussian GLM, pure l2 regularization)")
                        self.model = \
                        Linear_closed_form(lambda_=self.global_reg_penalty, # > 0
                                           memory_efficient=self.memory_efficient)
                    else:
                        self.description = ("Ridge Regression "
                                            "(Gaussian GLM, pure l2 regularization)")
                        self.model = \
                        sk_lin.Ridge(alpha=self.global_reg_penalty,
                                     fit_intercept=False,
                                     **kwargs)
                elif self.l2_penalty == 0:
                    self.description = ("Lasso Regression "
                                        "(Gaussian GLM, pure l1 regularization)")
                    self.model = \
                    sk_lin.Lasso(alpha=self.global_reg_penalty,
                                 fit_intercept=False,
                                 **kwargs)
                elif 0 < self.l2_penalty < 1:
                    self.description = ("Elastic Net Regression "
                                        "(Gaussian GLM, balanced l1 and l2 regularization "
                                        "where l1 = 1-l2 penalty)")
                    self.model = \
                    sk_lin.ElasticNet(alpha=self.global_reg_penalty,
                                      l1_ratio=self.l1_penalty,
                                      fit_intercept=False,
                                      **kwargs)
        elif self.distribution == "bernoulli":
                if global_reg_penalty == 0:
                    self.description = ("Logistic Regression "
                                        "(Bernoulli GLM), no regularization")
                    self.model = \
                    sk_lin.LogisticRegression(penalty=None,
                                              fit_intercept=False,
                                              **kwargs)
                else:
                    if self.l2_penalty == 1:
                        self.description = ("Logistic Regression "
                                            "(Bernoulli GLM), l2 regularization")
                        self.model = \
                        sk_lin.LogisticRegression(penalty='l2',
                                                  C = 1/self.global_reg_penalty,
                                                  fit_intercept=False,
                                                  **kwargs)
                    elif self.l2_penalty == 0:
                        self.description = ("Logistic Regression "
                                            "(Bernoulli GLM), l1 regularization")
                        self.model = \
                        sk_lin.LogisticRegression(penalty='l1',
                                                  C = 1/self.global_reg_penalty,
                                                  fit_intercept=False,
                                                  **kwargs)
                    elif 0 < self.l2_penalty < 1:
                        self.description = ("Logistic Regression (Bernoulli GLM), "
                                            "l1+l2 (elastic net) regularization")
                        # The actual value of l2_penalty is ignored in this case.
                        self.model = \
                        sk_lin.LogisticRegression(penalty='elasticnet',
                                                  C = 1/self.global_reg_penalty,
                                                  fit_intercept=False,
                                                  **kwargs)
        elif self.distribution == "poisson":
            self.description = "Poisson GLM, no regularization"
            # scikit-learn PoissonRegressor is
            # sctrictly equivalent to TweedieRegressor(power=1, link='log')
            self.model = sk_lin.TweedieRegressor(power=1,
                                                 link='log',
                                                 **kwargs)
            # Note: Using regularizers with Poisson GLMs is very new,
            # can only be approximated with 'cyclical gradient descent'.
            # This is implemented in the R package glmnet,
            # and in the python package PyGLMnet. Not worth implementing.

        self.normalize_X = normalize_X
        self.is_fit = False
        self.is_normalized = False
        self.eval = {} 
    
    def __repr__(self):
        return f"{self.description} with attributes:\n\t{repr_string(self)}"
    
    def assert_fit(self):
        assert self.is_fit, "You must first call .fit(X,y)."

    def scale(self, X, axis=0):
        m = X.mean(axis)
        s = X.std(axis)
        s[s==0] = 1 # cases where all values are the same
        X_scaled = (X - m)/s
        return X_scaled
        
    def fit(self, X, y, verbose=True):
        
        X = X.copy().squeeze()
        y = y.copy().squeeze()
        assert X.shape[0] == y.shape[0],\
          f"Dependant variable y of shape {y.shape} but provided design matrix X has {X.shape[0]} rows only!"
        
        if self.normalize_X:
            X = self.scale(X)
        
        if verbose: print(f"\nFitting a {self.description}...\n")
        self.model.fit(X, y)
        
        self.beta = self.model.coef_.squeeze()

        self.is_fit = True
    
    def predict(self, X):
        
        self.assert_fit()
        assert X.shape[1] == self.beta.shape[0],\
          f"{len(self.beta)} fit parameters but provided design matrix X is of shape {X.shape}!"
        
        return self.model.predict(X)
    
    def get_fit_params(self):
        
        self.assert_fit()
        
        return self.beta
    
    def evaluate(self, y, y_predicted, verbose=False):
        # Evaluate model performance
        y, y_predicted = y.squeeze(), y_predicted.squeeze()
        self.eval['mse'] = mean_squared_error(y, y_predicted)
        if verbose: print("Mean Squared Error:", self.eval['mse'])
        # R-squared
        self.eval['r2'] = r2_score(y, y_predicted)
        if verbose: print("R-squared:", self.eval['r2'] )
        # Explained variance
        self.eval['ev'] = explained_variance_score(y, y_predicted)
        if verbose: print("Explained variance:", self.eval['ev'])
        
        return self.eval['mse'], self.eval['r2'], self.eval['ev']


def generate_design_matrix(X, b, predictor_window):
    """
    Arguments:
        - X: input time series (n_bins, n_neurons,). T=0 is top of matrix.
        - b: float, size of bins in X
        - predictor_window: [t_past, t_future], by how much to shift X columns
                          in the past and the future. Will round to the closest multiple of 'b'
                          (must be same unit a 'b').
    Returns:
        - X_design, deisng matrix with time-shifted time series according to 'predictor_window'.
    """
    n_bins, n_neurons = X.shape
    
    assert predictor_window[0]<=0, "predictor_window[0] must be negative!"
    shifts_past = int(-predictor_window[0]//b)
    shifts_future = int(predictor_window[1]//b)

    X_3d = np.zeros((n_bins, shifts_past+1+shifts_future, n_neurons)).astype(np.float64)
    for i in range(n_neurons):

        binned_t = X[:,i]
        padded_t = np.hstack((np.zeros(shifts_past), binned_t))

        hankel_padded_t = hankel(padded_t, np.zeros(shifts_past+1+shifts_future))

        X_3d[:,:,i] = hankel_padded_t[:-shifts_past, :]
    
    # reshape such that neurons are concatenated
    X_no_offest = np.reshape(X_3d, (n_bins,-1), order='F')
    
    # now, add a bias term
    X_design = np.hstack((np.ones((n_bins,1)), X_no_offest))
    
    return X_design