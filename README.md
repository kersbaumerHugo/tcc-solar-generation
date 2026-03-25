# TCC Solar Generation

Previsão de geração de energia fotovoltaica usando Machine Learning.
Trabalho de Conclusão de Curso — Engenharia Mecânica, UFRJ.

**Autor:** Hugo Kersbaumer Knupp

---

## Objetivo

Prever a geração horária de usinas fotovoltaicas (MWh) a partir de dados climáticos do INMET,
utilizando Decision Trees e Random Forests com validação temporal (TimeSeriesSplit).

---

## Arquitetura

```
src/
├── config/          # Settings centralizados (Config class)
├── data/
│   ├── extractors/  # SolarGenerationExtractor, ClimateDataExtractor
│   ├── transformers/ # DataCleaner, DataNormalizer (ABC DataTransformer)
│   ├── validators/  # DataQualityChecker + QualityReport
│   └── processors.py # Facade: orquestra limpeza + normalização
├── features/
│   ├── engineering.py  # FeatureImportanceAnalyzer
│   └── selection.py    # FeatureSelector (top_n + threshold)
├── models/
│   ├── base.py          # ModelStrategy ABC (Strategy + OCP)
│   ├── decision_tree.py # DecisionTreeStrategy
│   ├── random_forest.py # RandomForestStrategy
│   ├── trainer.py       # ModelTrainer (split + train_all)
│   ├── evaluator.py     # ModelEvaluator (R², RMSE, MAE)
│   └── registry.py      # ModelRegistry (joblib save/load)
├── visualization/
│   ├── geographic.py    # Mapas de usinas/estações (Basemap opcional)
│   └── performance.py   # Gráficos de predição, importância, métricas
└── utils/
    ├── logger.py        # Logging estruturado
    └── io.py            # Utilitários de I/O

scripts/
├── run_training.py     # Treina e salva modelos + processor
├── run_evaluation.py   # Avalia modelos salvos
├── run_preprocessing.py
└── predict.py          # Prediz geração para novos dados
```

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/kersbaumerHugo/tcc-solar-generation.git
cd tcc-solar-generation

# Instale em modo editável (recomendado para desenvolvimento)
pip install -e ".[dev]"
```

---

## Como usar

### Treinar modelos

```bash
python scripts/run_training.py
```

Lê os dados processados em `data/full/`, treina Decision Tree e Random Forest
com GridSearchCV + TimeSeriesSplit, salva em `models/`.

### Avaliar modelos

```bash
python scripts/run_evaluation.py
```

Carrega os modelos salvos e imprime métricas (R², RMSE, MAE) por usina.

### Predizer geração

```python
from scripts.predict import predict
from src.models.registry import ModelRegistry
from pathlib import Path
import pandas as pd

registry = ModelRegistry(Path("models"))
df = pd.read_parquet("data/processed/minha_usina.parquet")
result = predict("MINHA_USINA", "random_forest", df, registry)
print(result[["GERACAO_PREVISTA", "GERACAO_REAL"]])
```

---

## Testes

```bash
# Todos os testes (unitários + integração)
pytest

# Só integração
pytest tests/integration/ -v

# Com cobertura
pytest --cov=src --cov-report=term-missing
```

**Cobertura atual:** 229 testes passando (unitários + integração).

---

## Features do modelo

| Feature | Descrição |
|---------|-----------|
| `RADIACAO` | Radiação solar (Kj/m²) |
| `TEMPERATURA` | Temperatura do ar (°C) |
| `UMIDADE` | Umidade relativa (%) |
| `hour_sin`, `hour_cos` | Hora do dia (codificação cíclica) |
| `1`..`12` | Mês do ano (one-hot encoding) |

**Target:** `GERACAO` — geração no centro de gravidade (MWh)

---

## Princípios de design

- **SRP** — cada classe tem uma responsabilidade
- **OCP** — novos modelos via `ModelStrategy` sem alterar `ModelTrainer`
- **DIP** — `ModelTrainer` depende da ABC, não de implementações concretas
- **Sem data leakage** — `DataNormalizer` fita apenas em `X_train`
- **Validação de dados** — `DataQualityChecker` bloqueia dados inválidos antes do pipeline
