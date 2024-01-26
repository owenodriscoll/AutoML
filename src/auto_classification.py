from .automl import AutomatedML
from .classifiers import classifier_selector
from dataclasses import dataclass
from typing import Callable, List
from sklearn.metrics import accuracy_score, precision_score


@dataclass
class AutomatedClassification(AutomatedML):
    """
    Automated classification, child class of AutomatedML
    
    Available classifiers: 
    --------------------
    'dummy', 'lightgbm', 'xgboost', 'catboost', 'adaboost', 'gradientboost', 
    'histgradientboost', 'knn', 'sgd', 'bagging', 'svc'
    
    """
    __doc__ += AutomatedML.__doc__
    
    metric_optimise: Callable = accuracy_score
    models_to_optimize: List[str] = None
    models_to_assess: List[str] = None
    _ml_objective: str = 'classification'
    
    def __post_init__(self):
        
        # -- run delayed initialization of parent method, this among others, 
        # splits the data in training and testing such that n_classes from 
        # training data can be used for
        self._stratify = self.y
        super().__post_init__()
        
        if self.metric_assess is None:
            precision_score_macro = [lambda y_true, y_pred: precision_score(y_true, y_pred, average = 'macro')]
            self.metric_assess: List[Callable] = [accuracy_score, precision_score_macro]

        if self.models_to_optimize is None: 
            self.models_to_optimize: List[str] = ['svc', 'sgd', 'histgradientboost']
            
        if self.models_to_assess is None:
            self.models_to_assess: List[str] = self.models_to_optimize
            
        # -- number of classes is a necessary initialization param for several classifiers
        n_classes = len(set(self.y_train)) + 1
        self._models_optimize: List[Callable] = classifier_selector(classifier_names=self.models_to_optimize,
                                                        random_state=self.random_state,  n_classes = n_classes)
        self._models_assess: List[Callable] = classifier_selector(classifier_names=self.models_to_assess,
                                                      random_state=self.random_state,  n_classes = n_classes)
        