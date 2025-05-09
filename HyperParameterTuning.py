"""This file deifnes the possible hyperparameters that are being investigated when HyperParameterTuning is set to True in 'RunModel.py'."""
from sklearn import svm
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer


modelClassifierDic = {
    'LR':{
        'model':LogisticRegression(),
        'params':{
            'penalty': ['l2'],
            'C': [0.1,0.5,1,2,5],
            'solver':["lbfgs"]#,'newton-cholesky']
            #Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.
        }
    },
    'DTC':{
        'model':DecisionTreeClassifier(),
        'params':{
            'splitter': ['best','random'],
            'max_depth':[2,4,6,8,10,None]
        }
    },
    'LRBagged':{
        'model':BaggingClassifier(estimator=LogisticRegression(solver = 'lbfgs',penalty='l2',C=1)), # These values should be changed depending on the HPtuning results for LR  
        'params':{
            'n_estimators':[2,3,4,5,8,10],
            'max_samples':[0.5,0.6,0.7,0.8],
            'oob_score':[False], # OOB score has below neglegible influence on the results 
            'warm_start':[False]
        }
    },
    'DTCBagged':{
        'model':BaggingClassifier(estimator=DecisionTreeClassifier(splitter='random',max_depth=2)), # DTC should be changed depending on the HPtuning results for DTC  
        'params':{
            'n_estimators':[2,3,5,8,10],
            'max_samples':[0.5,0.6,0.7,0.8],
            'oob_score':[False], # OOB score has below neglegible influence on the results 
            'warm_start':[True,False]
        }
    },
    'RFC':{ # Mere dybdegående RF 
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[2,3,4,5,6,7,8,9,10,20,40], # Number of trees
            'max_samples':[0.5,0.6,0.7,0.8,0.9],
            'max_depth':[3,4,5,6,7,8,9,10,20,None],
            'warm_start':[False,True], 
            'criterion':['gini','entropy','log_loss']
        }    
    },
    'XGBC':{
        'model':xgb.XGBClassifier(),
        'params':{
            'n_estimators':[2,3,4,5,6,8,10],
            'max_depth':[1,2,3,5,7,10], 
            'learning_rate':[0.4,0.5,0.6,0.7,0.8], 
            'grow_policy':['depthwise','lossguide'] # Tree growing policy. depthwise: Favors splitting at nodes closest to the node,lossguide: Favors splitting at nodes with highest loss change.
        }
    }
}
modelClassifierDic_fast = {
    'RFC':{ # Brugt til test af DS level 
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[2,3,4,5,8,10], # Number of trees
            'max_samples':[0.6,0.7,0.8],
            'max_depth':[3,4,5,8,10],
            'warm_start':[True], 
            'criterion':['gini','entropy','log_loss']
        }
    }
}

modelRegressorDic = {
    # 'DTR':{
    #     'model':DecisionTreeRegressor(),
    #     'params':{
    #         'splitter': ['best','random'],
    #         'max_depth':[None,1,2,3,4,5,6,7,8,9,10,15,20]
    #     }
    # },
    # 'LinReg':{
    #     'model':LinearRegression(),
    #     'params':{'copy_X':[True]}
    # },
    'RFR':{
        'model':RandomForestRegressor(),
        'params':{
            'n_estimators':[1,2,3,4,5,10,20,40,100], # Number of trees
            'max_samples':[0.5,0.6,0.7,0.8,0.9,1.0],
            'max_depth':[3,4,5,8,10,15,20,None],
            'warm_start':[True], 
            'criterion':['squared_error', 'absolute_error']#, 'friedman_mse', 'poisson']
        }
    }
    # 'XGBR':{
    #     'model':xgb.XGBRegressor(),
    #     'params':{
    #         'n_estimators':[1,2,3,4,5,6,7,8,9,10,20,40,100],
    #         'max_depth':[1,2,3,4,5,6,7,8,9,10,20,40], 
    #         'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #         # 'eval_metric':[recall_score,None],
    #         # 'booster':['gbtree','gblinear','dart'],
    #         # 'grow_policy':['depthwise','lossguide'] # Tree growing policy. depthwise: Favors splitting at nodes closest to the node,lossguide: Favors splitting at nodes with highest loss change.
    #     }
    # }
}
modelRegressorDic_fast = {
    'RFR':{
        'model':RandomForestRegressor(),
        'params':{
            'n_estimators':[2,3,4,5,8,10], # Number of trees
            'max_samples':[0.6,0.7,0.8],
            'max_depth':[3,4,5,8,10],
            'warm_start':[True], 
            'criterion':['squared_error', 'absolute_error']#, 'friedman_mse', 'poisson']
        }
    }
}

