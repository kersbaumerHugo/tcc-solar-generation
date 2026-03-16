# src/models/trainer.py
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.logger import logger

class ModelTrainer:
    """Responsável pelo treinamento de modelos de ML"""
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        min_samples: int = 10000
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.min_samples = min_samples
        self.tscv = TimeSeriesSplit()
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = "GERACAO"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados em treino/teste
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        
        logger.info(f"Dividindo dados: {len(X)} amostras, test_size={self.test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            shuffle=False  # Importante para séries temporais!
        )
        
        # Tratamento final de NaN
        if "UMIDADE" in X_train.columns:
            mean_umidade = X_train["UMIDADE"].mean()
            X_train["UMIDADE"].fillna(mean_umidade, inplace=True)
            X_test["UMIDADE"].fillna(mean_umidade, inplace=True)
        
        return X_train, X_test, y_train, y_test
    
    def train_decision_tree(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        grid_params: Dict = None
    ) -> Tuple[DecisionTreeRegressor, Dict]:
        """
        Treina Decision Tree com GridSearchCV
        
        Returns:
            (best_model, cv_results)
        """
        if grid_params is None:
            grid_params = {
                'max_features': [1.0],
                'max_depth': [5, 7, 10, 12, 15, 20],
                'criterion': ['squared_error'],
                'random_state': [self.random_state]
            }
        
        logger.info("Treinando Decision Tree com GridSearchCV...")
        
        dt_cv = GridSearchCV(
            estimator=DecisionTreeRegressor(),
            param_grid=grid_params,
            cv=self.tscv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1  # Usa todos os cores
        )
        
        dt_cv.fit(X_train, y_train)
        
        logger.info(f"Melhor score (CV): {dt_cv.best_score_:.4f}")
        logger.info(f"Melhores parâmetros: {dt_cv.best_params_}")
        
        return dt_cv.best_estimator_, dt_cv.cv_results_
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        grid_params: Dict = None
    ) -> Tuple[RandomForestRegressor, Dict]:
        """
        Treina Random Forest com GridSearchCV
        
        Returns:
            (best_model, cv_results)
        """
        if grid_params is None:
            grid_params = {
                'n_estimators': [1, 2, 5, 10, 15, 20],
                'max_features': [1.0],
                'max_depth': [5, 7, 10, 12, 15, 20],
                'criterion': ['squared_error'],
                'random_state': [self.random_state]
            }
        
        logger.info("Treinando Random Forest com GridSearchCV...")
        
        rf_cv = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=grid_params,
            cv=self.tscv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1
        )
        
        rf_cv.fit(X_train, y_train)
        
        logger.info(f"Melhor score (CV): {rf_cv.best_score_:.4f}")
        logger.info(f"Melhores parâmetros: {rf_cv.best_params_}")
        
        return rf_cv.best_estimator_, rf_cv.cv_results_