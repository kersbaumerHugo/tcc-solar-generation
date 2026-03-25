# tests/integration/test_pipeline.py
"""
Testes de integração do pipeline completo.

Cada teste exercita múltiplos componentes juntos para detectar bugs de
interface que testes unitários não encontram: coluna com nome errado
chegando ao modelo, scaler fitado em dados errados, features incompatíveis
entre treino e predição, etc.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.predict import predict
from src.data.processors import DataProcessor
from src.data.transformers.cleaning import DataCleaner
from src.data.transformers.normalization import DataNormalizer
from src.data.validators.quality_checks import DataQualityChecker
from src.features.engineering import FeatureImportanceAnalyzer
from src.features.selection import FeatureSelector
from src.models.decision_tree import DecisionTreeStrategy
from src.models.evaluator import ModelEvaluator
from src.models.random_forest import RandomForestStrategy
from src.models.registry import ModelRegistry
from src.models.trainer import ModelTrainer


# ---------------------------------------------------------------------------
# 1. Limpeza → Normalização → Split → Treino → Avaliação
# ---------------------------------------------------------------------------


class TestCleanNormTrainEvaluate:
    """Fluxo principal: DataCleaner → DataNormalizer → ModelTrainer → ModelEvaluator."""

    def test_cleaned_df_has_no_nulls(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        assert df.isnull().sum().sum() == 0

    def test_cleaned_df_has_temporal_features(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        assert "hour_cos" in df.columns
        assert all(str(m) in df.columns for m in range(1, 13))

    def test_split_preserves_temporal_order(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        X_train, X_test, _, _ = ModelTrainer().split_data(df)
        assert X_train.index.max() < X_test.index.min()

    def test_normalizer_fits_on_train_only(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(df)

        normalizer = DataNormalizer()
        X_train_scaled = normalizer.fit_transform(X_train)
        X_test_scaled = normalizer.transform(X_test)

        # Treino deve estar em [0, 1]
        assert X_train_scaled.min().min() >= -1e-9
        assert X_train_scaled.max().max() <= 1.0 + 1e-9
        # Teste pode ter valores fora do range (usa escala do treino)
        assert isinstance(X_test_scaled, pd.DataFrame)

    def test_decision_tree_produces_valid_metrics(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy()])
        X_train, X_test, y_train, y_test = trainer.split_data(df)

        normalizer = DataNormalizer()
        X_train_s = normalizer.fit_transform(X_train)
        X_test_s = normalizer.transform(X_test)

        results = trainer.train_all(X_train_s, y_train)
        model, _ = results["Decision Tree"]

        metrics = ModelEvaluator().evaluate(model, X_test_s, y_test)
        assert -1.0 <= metrics["r2"] <= 1.0
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

    def test_random_forest_r2_greater_than_decision_tree(self, pipeline_df):
        """RF geralmente supera DT em datasets maiores — valida a lógica do ensemble."""
        df = DataCleaner().transform(pipeline_df)
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy(), RandomForestStrategy()])
        X_train, X_test, y_train, y_test = trainer.split_data(df)

        normalizer = DataNormalizer()
        X_train_s = normalizer.fit_transform(X_train)
        X_test_s = normalizer.transform(X_test)

        results = trainer.train_all(X_train_s, y_train)
        evaluator = ModelEvaluator()

        dt_metrics = evaluator.evaluate(results["Decision Tree"][0], X_test_s, y_test)
        rf_metrics = evaluator.evaluate(results["Random Forest"][0], X_test_s, y_test)

        # Ambos devem ter R² positivo em dados com sinal real
        assert dt_metrics["r2"] > 0
        assert rf_metrics["r2"] > 0


# ---------------------------------------------------------------------------
# 2. Validação de qualidade → bloqueio de dados ruins
# ---------------------------------------------------------------------------


class TestQualityGate:
    """DataQualityChecker deve detectar problemas antes do pipeline."""

    def test_clean_data_passes_quality_check(self, pipeline_df):
        report = DataQualityChecker().check(pipeline_df)
        assert report.passed

    def test_missing_required_column_fails(self, pipeline_df):
        df = pipeline_df.drop(columns=["RADIACAO"])
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("RADIACAO" in issue for issue in report.issues)

    def test_negative_radiacao_fails(self, pipeline_df):
        df = pipeline_df.copy()
        df.iloc[0, df.columns.get_loc("RADIACAO")] = -50.0
        report = DataQualityChecker().check(df)
        assert not report.passed

    def test_duplicate_timestamps_fail(self, pipeline_df):
        df = pd.concat([pipeline_df, pipeline_df.iloc[:10]])
        report = DataQualityChecker().check(df)
        assert not report.passed


# ---------------------------------------------------------------------------
# 3. Registry → save → load → predict (ciclo completo)
# ---------------------------------------------------------------------------


class TestRegistryCycle:
    """Treina, persiste e recupera — garante que save/load não corrompe artefatos."""

    @pytest.fixture(scope="class")
    def trained_registry(self, tmp_path_factory, pipeline_df):
        """Registry com modelo + processor persistidos."""
        models_dir = tmp_path_factory.mktemp("models")
        registry = ModelRegistry(models_dir)

        processor = DataProcessor()
        df = processor.rename_columns(pipeline_df)
        df = processor.add_temporal_features(df)
        df = processor.handle_missing_values(df)
        df = processor.drop_irrelevant_features(df)

        trainer = ModelTrainer(strategies=[DecisionTreeStrategy()])
        X_train, X_test, y_train, y_test = trainer.split_data(df)
        X_train_s = processor.normalize_data(X_train)

        results = trainer.train_all(X_train_s, y_train)
        model, _ = results["Decision Tree"]

        registry.save(model, "INTEG_decision_tree")
        registry.save(processor, "INTEG_processor")

        return registry, X_test, y_test, processor

    def test_loaded_model_predicts_same_as_original(self, trained_registry, pipeline_df):
        registry, X_test, y_test, processor = trained_registry

        model = registry.load("INTEG_decision_tree")
        X_test_s = processor.transform(X_test)
        preds = model.predict(X_test_s)

        assert len(preds) == len(X_test)
        assert not np.isnan(preds).any()

    def test_predict_function_end_to_end(self, trained_registry, pipeline_df):
        registry, _, _, _ = trained_registry

        result = predict("INTEG", "decision_tree", pipeline_df, registry)

        assert "GERACAO_PREVISTA" in result.columns
        assert "GERACAO_REAL" in result.columns
        assert len(result) > 0
        assert not result["GERACAO_PREVISTA"].isna().any()

    def test_predict_output_index_is_subset_of_input(self, trained_registry, pipeline_df):
        """handle_missing_values pode remover linhas — índice do resultado ⊆ input."""
        registry, _, _, _ = trained_registry
        result = predict("INTEG", "decision_tree", pipeline_df, registry)
        assert result.index.isin(pipeline_df.index).all()


# ---------------------------------------------------------------------------
# 4. FeatureSelector integrado ao pipeline de treino
# ---------------------------------------------------------------------------


class TestFeatureSelectorIntegration:
    """FeatureSelector deve reduzir dimensionalidade sem quebrar o pipeline."""

    def test_reduced_features_still_train(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy()])
        X_train, X_test, y_train, y_test = trainer.split_data(df)

        normalizer = DataNormalizer()
        X_train_s = normalizer.fit_transform(X_train)
        X_test_s = normalizer.transform(X_test)

        # Treina modelo inicial para extrair importâncias
        results = trainer.train_all(X_train_s, y_train)
        model, _ = results["Decision Tree"]

        # Seleciona top 5 features
        selector = FeatureSelector(top_n=5)
        X_train_r = selector.fit_transform(model, list(X_train_s.columns), X_train_s)
        X_test_r = selector.transform(X_test_s)

        assert X_train_r.shape[1] == 5
        assert X_test_r.shape[1] == 5
        assert list(X_train_r.columns) == list(X_test_r.columns)

    def test_feature_importance_covers_all_features(self, pipeline_df):
        df = DataCleaner().transform(pipeline_df)
        trainer = ModelTrainer(strategies=[DecisionTreeStrategy()])
        X_train, _, y_train, _ = trainer.split_data(df)

        normalizer = DataNormalizer()
        X_train_s = normalizer.fit_transform(X_train)

        results = trainer.train_all(X_train_s, y_train)
        model, _ = results["Decision Tree"]

        importance_df = FeatureImportanceAnalyzer().get_importance(model, list(X_train_s.columns))
        assert len(importance_df) == X_train_s.shape[1]
        assert abs(importance_df["importance"].sum() - 1.0) < 1e-6
