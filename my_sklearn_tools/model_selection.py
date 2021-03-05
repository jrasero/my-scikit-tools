import numpy as np
from sklearn.model_selection import StratifiedKFold

class StratifiedKFoldReg(StratifiedKFold):
    
    """
    
    This class generate cross-validation partitions
    for regression setups, such that these partitions
    resemble the original sample distribution of the 
    target variable.
    
    """
    
    def split(self, X, y, groups=None):
        
        n_samples = len(y)

        # This little correction is to guarantee that 
        # each bin has at least n_split points within it

        if n_samples % self.n_splits==0:
            num = int(n_samples/self.n_splits)
        else:
            num = int(n_samples/(1 + self.n_splits))

        q = np.quantile(y, np.linspace(0, 1, num, endpoint=False))
        y_bin = np.digitize(y, q)

        return super().split(X, y_bin, groups)