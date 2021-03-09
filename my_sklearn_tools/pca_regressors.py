import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Memory
from tempfile import mkdtemp

__all__ = ['PCARegressionCV', 'PCARegression'] 

class PCARegressionCV():
    
    def __init__(self, 
                scale=False,
                cv=None, 
                n_jobs=None, 
                cache_dir=True,  
                verbose=0):
        
        self.scale = scale
        self.cv = cv
        self.n_jobs=n_jobs
        self.cache_dir = cache_dir
        self.verbose = verbose
        
    
    
    def build(self, reg_type):
        
        if type(reg_type) is str:
            try:
                reg = self._get_regressor(reg_type)
            except:
                msg = "Regression type is not valid"
                ValueError(msg)
            
        if self.cache_dir:
            tmpfolder = mkdtemp()
            memory = Memory(location=tmpfolder, self.verbose)
        else:
            memory=None
        
        if self.scale:
            pip = make_pipeline(VarianceThreshold(), StandardScaler(), PCA(), reg, 
                                memory=memory)
        else:
            pip = make_pipeline(VarianceThreshold(), PCA(), reg, 
                                memory=memory)
        
    
        param_grid = self._get_param_grid(reg_type)
        grid = GridSearchCV(pip, 
                            param_grid, 
                            cv =self.cv, 
                            n_jobs=self.n_jobs,
                            scoring="neg_mean_squared_error")
                    
        return grid
    
    def _get_regressor(self, 
                       reg_type):
    
        regression_types = {'lasso': linear_model.Lasso(max_iter=1e6), 
                            'ridge': linear_model.Ridge(),
                            'elasticnet':linear_model.ElasticNet(max_iter=1e6),
                            #'lars': TODO,
                            #'lassolars': TODO
                            }
        reg = regression_types[reg_type]
        return reg

    def _get_param_grid(self, reg_type):
        
        alphas =  10**np.linspace(-4, 2, 100)
        l1_ratio = [0.25, 0.5, 0.75]
        param_grids = {
            'lasso': {'lasso__alpha': alphas},
            'ridge': {'ridge__alpha': alphas},
            'elasticnet':  {'elasticnet__alpha': alphas, 
                            'elasticnet__l1_ratio': l1_ratio},
                       #'lars': TODO,
                       #'lassolars': TODO
                       # }
        }
                       
        param_grid = param_grids[reg_type]
        return param_grid


class PCARegression():
    
    def __init__(self,
                 scale=False, 
                 cache_dir=False, 
                 verbose=0):
        self.scale = scale
        self.cache_dir = cache_dir
        self.verbose = verbose
    
    
    def build(self, reg_type):
        
        if type(reg_type) is str:
            try:
                reg = self._get_regressor(reg_type)
            except:
                msg = "Regression type is not valid"
                ValueError(msg)
        
             
        if self.cache_dir:
            tmpfolder = mkdtemp()
            memory = Memory(location=tmpfolder, self.verbose)
        else:
            memory=None
            
        if self.scale:
            pip = make_pipeline(VarianceThreshold(), StandardScaler(), PCA(), reg, 
                                memory=memory)
        else:
            pip = make_pipeline(VarianceThreshold(), PCA(), reg, 
                                memory=memory)       
    
        return pip
    def _get_regressor(self, reg_type):
    
        regression_types = {'ols': linear_model.LinearRegression(),
                            'lasso': linear_model.Lasso(max_iter=1e6), 
                            'ridge': linear_model.Ridge(),
                            'elasticnet':linear_model.ElasticNet(max_iter=1e6),
                            #'lars': TODO,
                            #'lassolars': TODO
                            }
        reg = regression_types[reg_type]
        return reg
