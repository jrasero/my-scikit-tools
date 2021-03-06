# my-scikit-tools

Being tired of defining the same lines of code every time that I use specific things,
I decided to implement these in this package, so that I just need to 
import the particular tools that I need.

Up to now, the tools implemented are:

  - A module for principal component regression. This module includes an object that simply builds either a Pipeline or a GridSearchCV with a pipeline that concatenates a Variance Threshold feature selection, Scaling, PCA and a  linear regressor with regularisation (None, L1, L2 or ElasticNet). Alternatively, there are also coded LassoPCR, RidgePCR and ElasticnetPCR objects that in principle should be much faster than using GridSearchCV.
  - A stratifiedKFold object for regression.
  - A module for connectivity-based predictive modelling.

Under continuously development.

# How to install

```
git clone https://github.com/jrasero/my-scikit-tools.git
cd my-scikit-tools
pip install -U .
```
