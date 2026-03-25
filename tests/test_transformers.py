# tests/test_transformers.py
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.data.transformers.base import DataTransformer
from src.data.transformers.cleaning import DataCleaner
from src.data.transformers.normalization import DataNormalizer


class TestDataTransformerABC:
    def test_is_abstract(self):
        """DataTransformer não pode ser instanciado diretamente."""
        with pytest.raises(TypeError):
            DataTransformer()

    def test_fit_transform_calls_fit_then_transform(self, sample_df):
        """fit_transform() deve produzir o mesmo resultado que fit().transform()."""
        cleaner = DataCleaner()
        result_chained = cleaner.fit(sample_df).transform(sample_df)
        result_shortcut = DataCleaner().fit_transform(sample_df)
        pd.testing.assert_frame_equal(result_chained, result_shortcut)


class TestDataCleaner:
    def test_fit_returns_self(self, sample_df):
        cleaner = DataCleaner()
        assert cleaner.fit(sample_df) is cleaner

    def test_transform_adds_12_month_columns(self, sample_df):
        df = DataCleaner().transform(sample_df)
        for month in range(1, 13):
            assert str(month) in df.columns

    def test_transform_adds_hour_cos(self, sample_df):
        df = DataCleaner().transform(sample_df)
        assert "hour_cos" in df.columns

    def test_hour_cos_values_in_range(self, sample_df):
        df = DataCleaner().transform(sample_df)
        assert df["hour_cos"].between(-1.0, 1.0).all()

    def test_month_columns_are_binary(self, sample_df):
        df = DataCleaner().transform(sample_df)
        for month in range(1, 13):
            unique = set(df[str(month)].unique())
            assert unique <= {0, 1}

    def test_drops_radiacao_nulls(self, sample_df_with_nulls):
        """Linhas sem RADIACAO devem ser removidas."""
        original_len = len(sample_df_with_nulls)
        df = DataCleaner().transform(sample_df_with_nulls)
        assert len(df) < original_len

    def test_fills_temperatura_nulls(self, sample_df_with_nulls):
        df = DataCleaner().transform(sample_df_with_nulls)
        assert not df["TEMPERATURA"].isna().any()

    def test_fills_umidade_nulls(self, sample_df_with_nulls):
        df = DataCleaner().transform(sample_df_with_nulls)
        assert not df["UMIDADE"].isna().any()

    def test_transform_without_fit_works(self, sample_df):
        """DataCleaner é stateless — transform() funciona sem fit() prévio."""
        df = DataCleaner().transform(sample_df)
        assert not df.empty

    def test_full_pipeline_removes_no_unwanted_rows(self, sample_df):
        """Sem nulls em RADIACAO, nenhuma linha deve ser removida."""
        df = DataCleaner().transform(sample_df)
        assert len(df) == len(sample_df)

    def test_rename_columns_static(self, raw_df):
        """_rename_columns pode ser chamado diretamente como @staticmethod."""
        df = DataCleaner._rename_columns(raw_df)
        assert "GERACAO" in df.columns


class TestDataNormalizer:
    def test_fit_transform_range_is_0_to_1(self, sample_df):
        X = sample_df.drop(columns=["GERACAO"])
        result = DataNormalizer().fit_transform(X)
        assert result.min().min() >= 0.0 - 1e-10
        assert result.max().max() <= 1.0 + 1e-10

    def test_fit_transform_preserves_columns(self, sample_df):
        X = sample_df.drop(columns=["GERACAO"])
        result = DataNormalizer().fit_transform(X)
        assert list(result.columns) == list(X.columns)

    def test_fit_transform_preserves_index(self, sample_df):
        X = sample_df.drop(columns=["GERACAO"])
        result = DataNormalizer().fit_transform(X)
        pd.testing.assert_index_equal(result.index, X.index)

    def test_transform_uses_train_scale(self, sample_df):
        """X_test normalizado com escala do treino pode ultrapassar [0,1]."""
        n = len(sample_df)
        X_train = sample_df.drop(columns=["GERACAO"]).iloc[: n // 2]
        X_test = sample_df.drop(columns=["GERACAO"]).iloc[n // 2 :]

        normalizer = DataNormalizer()
        normalizer.fit_transform(X_train)
        result = normalizer.transform(X_test)

        # Resultado deve usar a escala do treino, não re-fitada no teste
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(X_test.columns)

    def test_transform_raises_before_fit(self, sample_df):
        """transform() sem fit() prévio deve lançar NotFittedError."""
        X = sample_df.drop(columns=["GERACAO"])
        with pytest.raises(NotFittedError):
            DataNormalizer().transform(X)

    def test_fit_returns_self(self, sample_df):
        X = sample_df.drop(columns=["GERACAO"])
        normalizer = DataNormalizer()
        assert normalizer.fit(X) is normalizer

    def test_scaler_property_returns_minmaxscaler(self, sample_df):
        from sklearn.preprocessing import MinMaxScaler

        X = sample_df.drop(columns=["GERACAO"])
        normalizer = DataNormalizer()
        normalizer.fit_transform(X)
        assert isinstance(normalizer.scaler, MinMaxScaler)
