# tests/test_processors.py
import numpy as np
import pandas as pd
import pytest

from src.config.settings import Config
from src.data.processors import DataProcessor

# sample_df, sample_df_with_nulls e raw_df vêm do conftest.py


class TestRenameColumns:
    def test_renames_geracao_column(self, raw_df):
        processor = DataProcessor()
        result = processor.rename_columns(raw_df)
        assert "GERACAO" in result.columns

    def test_renames_radiacao_column(self, raw_df):
        processor = DataProcessor()
        result = processor.rename_columns(raw_df)
        assert "RADIACAO" in result.columns

    def test_uses_config_mapping(self, raw_df):
        """rename_columns deve usar Config.COLUMN_MAPPING — sem mapeamento local."""
        processor = DataProcessor()
        result = processor.rename_columns(raw_df)
        for original, renamed in Config.COLUMN_MAPPING.items():
            if original in raw_df.columns:
                assert renamed in result.columns


class TestAddTemporalFeatures:
    def test_adds_all_12_month_columns(self, sample_df):
        processor = DataProcessor()
        result = processor.add_temporal_features(sample_df)
        for month in range(1, 13):
            assert str(month) in result.columns

    def test_adds_hour_cos(self, sample_df):
        processor = DataProcessor()
        result = processor.add_temporal_features(sample_df)
        assert "hour_cos" in result.columns

    def test_hour_cos_range(self, sample_df):
        processor = DataProcessor()
        result = processor.add_temporal_features(sample_df)
        assert result["hour_cos"].between(-1.0, 1.0).all()

    def test_month_columns_are_binary(self, sample_df):
        """Colunas de mês devem conter apenas 0 e 1."""
        processor = DataProcessor()
        result = processor.add_temporal_features(sample_df)
        for month in range(1, 13):
            unique_vals = set(result[str(month)].unique())
            assert unique_vals <= {0, 1}

    def test_single_month_still_produces_12_columns(self):
        """Garante que o fix do LabelBinarizer funciona: mesmo com 1 mês, 12 colunas."""
        index = pd.date_range("2023-06-01", periods=10, freq="h")
        df = pd.DataFrame(
            {"GERACAO": [1.0] * 10, "RADIACAO": [500.0] * 10},
            index=index,
        )
        processor = DataProcessor()
        result = processor.add_temporal_features(df)
        for month in range(1, 13):
            assert str(month) in result.columns

    def test_drops_intermediate_mes_column(self, sample_df):
        processor = DataProcessor()
        result = processor.add_temporal_features(sample_df)
        assert "Mes" not in result.columns

    def test_drops_intermediate_hour_column(self, sample_df):
        processor = DataProcessor()
        result = processor.add_temporal_features(sample_df)
        assert "Hour" not in result.columns


class TestHandleMissingValues:
    def test_drops_rows_without_radiacao(self, sample_df_with_nulls):
        processor = DataProcessor()
        result = processor.handle_missing_values(sample_df_with_nulls)
        assert result["RADIACAO"].isna().sum() == 0

    def test_fills_temperatura_with_mean(self, sample_df_with_nulls):
        processor = DataProcessor()
        result = processor.handle_missing_values(sample_df_with_nulls)
        assert result["TEMPERATURA"].isna().sum() == 0

    def test_fills_umidade_with_mean(self, sample_df_with_nulls):
        """Umidade agora é tratada aqui, não mais em trainer.split_data()."""
        processor = DataProcessor()
        result = processor.handle_missing_values(sample_df_with_nulls)
        assert result["UMIDADE"].isna().sum() == 0

    def test_preserves_rows_with_full_data(self, sample_df):
        original_len = len(sample_df)
        processor = DataProcessor()
        result = processor.handle_missing_values(sample_df)
        assert len(result) == original_len


class TestDropIrrelevantFeatures:
    def test_drops_configured_columns(self):
        """Deve remover todas as colunas em Config.FEATURES_TO_DROP."""
        from src.config.settings import Config

        index = pd.date_range("2023-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "GERACAO": [1.0] * 5,
                "RADIACAO": [500.0] * 5,
                "VEL_VENTO": [3.0] * 5,
                "COORD_N": [0.0] * 5,
            },
            index=index,
        )
        processor = DataProcessor()
        result = processor.drop_irrelevant_features(df)

        for col in Config.FEATURES_TO_DROP:
            assert col not in result.columns, f"'{col}' deveria ter sido removida"

    def test_preserves_target_and_feature_columns(self):
        index = pd.date_range("2023-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {"GERACAO": [1.0] * 5, "RADIACAO": [500.0] * 5, "VEL_VENTO": [3.0] * 5},
            index=index,
        )
        processor = DataProcessor()
        result = processor.drop_irrelevant_features(df)

        assert "GERACAO" in result.columns
        assert "RADIACAO" in result.columns

    def test_does_not_fail_if_column_absent(self):
        """errors='ignore': não falha se uma coluna de FEATURES_TO_DROP não existir."""
        index = pd.date_range("2023-01-01", periods=5, freq="h")
        df = pd.DataFrame({"GERACAO": [1.0] * 5, "RADIACAO": [500.0] * 5}, index=index)
        processor = DataProcessor()
        # Nenhuma coluna de FEATURES_TO_DROP existe aqui — não deve levantar exceção
        result = processor.drop_irrelevant_features(df)
        assert list(result.columns) == ["GERACAO", "RADIACAO"]


class TestNormalizeData:
    def test_output_range_is_0_to_1(self, sample_df):
        processor = DataProcessor()
        result = processor.normalize_data(sample_df)
        assert result.min().min() >= -1e-10
        assert result.max().max() <= 1.0 + 1e-10

    def test_preserves_column_names(self, sample_df):
        processor = DataProcessor()
        result = processor.normalize_data(sample_df)
        assert list(result.columns) == list(sample_df.columns)

    def test_preserves_index(self, sample_df):
        processor = DataProcessor()
        result = processor.normalize_data(sample_df)
        assert result.index.equals(sample_df.index)


class TestTransform:
    def test_transform_uses_train_scale(self, sample_df):
        """
        Garante que transform() aplica a escala do treino, não recalcula.

        Importância: o conjunto de teste não pode influenciar os parâmetros
        do scaler — isso seria data leakage.
        """
        train = sample_df.iloc[:15]
        test = sample_df.iloc[15:]

        processor = DataProcessor()
        processor.normalize_data(train)  # fit no treino
        result = processor.transform(test)  # transform no teste

        assert list(result.columns) == list(test.columns)
        assert result.index.equals(test.index)

    def test_transform_preserves_column_names(self, sample_df):
        train = sample_df.iloc[:15]
        test = sample_df.iloc[15:]
        processor = DataProcessor()
        processor.normalize_data(train)
        result = processor.transform(test)
        assert list(result.columns) == list(test.columns)

    def test_transform_raises_before_fit(self, sample_df):
        """transform() sem normalize_data() prévia deve falhar explicitamente."""
        processor = DataProcessor()
        with pytest.raises(Exception):
            processor.transform(sample_df)
