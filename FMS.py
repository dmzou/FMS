import numpy as np
from sklearn.utils.extmath import randomized_svd

class FMS:

    def __init__(self, n_components=2, robustness_power=1, max_iter=10000, tau=1e-8, epsilon=1e-8):
        self.n_components = n_components
        self.robustness_power = robustness_power
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
         
    def fit(self, data):
        iter = 0
        self.mean = np.mean( data , axis=0 )
        data =  data - self.mean  # centralize
        self.data = normalize( data )
        U , Sigma , VT = randomized_svd( self.data, 
                                         n_components=self.n_components,
                                         random_state=None )
        V = VT.T
        diff_c = np.inf
        c_prev = np.inf
        while diff_c > 1e-7 and iter < self.max_iter:
            # proj data onto orth. complement
            C = self.data.T - np.matmul( V , np.matmul( V.T , self.data.T ) )
            scale = np.reshape( np.sqrt( np.sum( C * C , axis=0 ) ) ** ( 2 - self.robustness_power ), (1,-1) )
            Y = self.data * np.matlib.repmat( np.minimum( scale.T ** (-0.5) , 1/self.epsilon ) , 1 , np.shape(self.data)[1] )
             
            c = np.sum( scale )
            diff_c = c_prev - c
            c_prev = c
             
            U , Sigma , VT = randomized_svd( Y, 
                                             n_components=self.n_components,
                                             random_state=285714 )
            V = VT.T
            iter += 1
        self.V = V #V: matrix consisting of principal components 
        print( "FMS fitted with {:d} steps!".format( iter ) )
         
    def transform(self, data):
        data = data - self.mean
        return np.matmul( data , self.V )
         
    def fit_transform(self, data):
        self.fit(data)
        data = data - self.mean
        return np.matmul( data , self.V )
