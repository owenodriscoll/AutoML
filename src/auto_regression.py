from .automl import AutomatedML
from .regressors import regressor_selector
from dataclasses import dataclass
from typing import Callable, List
from sklearn.metrics import median_absolute_error, r2_score


@dataclass
class AutomatedRegression(AutomatedML):
    """
    Automated regression, child class of AutomatedML
    
    Available regressors: 
    --------------------
    'dummy', 'lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 'adaboost', 
    'gradientboost', 'histgradientboost', 'knn', 'sgd', 'bagging', 'svr', 'elasticnet'
    
    """
    __doc__ += AutomatedML.__doc__
    
    metric_optimise: Callable = r2_score
    models_to_optimize: List[str] = None
    models_to_assess: List[str] = None
    _ml_objective: str = 'regression'
    
    def __post_init__(self):
        
        # -- run delayed initialization of parent method, this among others, 
        # creates the directory for files to be saved
        super().__post_init__()
        
        if self.metric_assess is None:
            self.metric_assess: List[Callable] =  [median_absolute_error, r2_score]

        if self.models_to_optimize is None: 
            self.models_to_optimize: List[str] = ['lassolars', 'bayesianridge', 'histgradientboost']
            
        if self.models_to_assess is None:
            self.models_to_assess: List[str] = self.models_to_optimize
            
        self._models_optimize: List[Callable] = regressor_selector(regressor_names=self.models_to_optimize,
                                                        random_state=self.random_state)
        self._models_assess: List[Callable] = regressor_selector(regressor_names=self.models_to_assess,
                                                      random_state=self.random_state)
        