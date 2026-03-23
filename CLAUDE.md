# 🤖 CLAUDE - CONFIGURAÇÃO DO PROJETO

Este arquivo configura o comportamento do Claude AI como **mentor sênior de arquitetura** para este projeto.

---

## 📊 SOBRE O PROJETO

### Identificação
- **Nome:** TCC Solar Generation - Previsão de Geração Fotovoltaica
- **Tipo:** Trabalho de Conclusão de Curso (Engenharia Mecânica - UFRJ)
- **Autor:** Hugo Kersbaumer Knupp
- **Status:** Em refatoração (código acadêmico → produção)

### Objetivo
Prever geração de energia solar fotovoltaica usando **Machine Learning** (Decision Trees e Random Forests) com base em dados climáticos do INMET.

### Stack Tecnológica
```python
# Data Manipulation
pandas >= 2.0.0
numpy >= 1.24.0
openpyxl >= 3.1.0

# Machine Learning
scikit-learn >= 1.3.0
- DecisionTreeRegressor
- RandomForestRegressor
- GridSearchCV
- TimeSeriesSplit

# Visualization
matplotlib >= 3.7.0
basemap

# Geolocation
geopy >= 2.3.0

#🏗️ ESTRUTURA ATUAL (LEGADO)
##Pipeline de Execução

projeto/
├── gerador_csv_geracao.py       # ~200 linhas - Processa dados de geração
├── gerador_dados_climaticos.py  # ~150 linhas - Processa dados INMET
├── merge_geracao_clima.py       # ~250 linhas - Merge temporal dos dados
├── modelo_final.py              # ~400 linhas - GridSearch + validação
├── mapa.py                      # ~100 linhas - Visualizações geográficas
├── comparacao_incidenciaxhoras.py # Script de análise exploratória
└── README.md

##Fluxo de Dados
###Input: Excel com geração solar + CSVs do INMET
###ETL: Limpeza, merge temporal, feature engineering
###ML: GridSearchCV com TimeSeriesSplit
###Output: Modelos treinados + visualizações

###Features do Modelo:
. Target: Geração de energia (MWh)
. Features principais:
. RADIACAO (Kj/m²)
. TEMPERATURA (°C)
. UMIDADE (%)
. Hour (transformação cosseno)
. Month (one-hot encoding)

##❌ PROBLEMAS IDENTIFICADOS
. Arquitetura
. Código monolítico (funções com 100+ linhas)
. Lógica de negócio misturada com I/O
. Violação de Single Responsibility Principle
. Paths hardcoded (cwd+r'\\geracao_csv\\')
. Sem separação de responsabilidades (ETL/Train/Evaluate)

##Qualidade de Código:
. Sem type hints
. Sem docstrings (ou inadequadas)
. print() ao invés de logging estruturado
. Nomes de variáveis pouco descritivos (df_full, cwd)
. Comentários em português misturados com inglês
. Magic numbers sem constantes
. Engenharia de Software
. Sem testes unitários
. Sem validação de dados
. Sem tratamento robusto de exceções
. Sem versionamento de dados
. Sem rastreabilidade (data lineage)
. Difícil de escalar
##Performance
. Carrega datasets completos na memória
. Sem chunking para grandes arquivos
. Operações não vetorizadas
. Sem cache de resultados intermediários

#✅ ARQUITETURA ALVO
##Estrutura Moderna
tcc-solar-generation/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configurações centralizadas
│   │   └── constants.py         # Constantes do projeto
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extractors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # DataExtractor ABC
│   │   │   ├── generation.py    # SolarGenerationExtractor
│   │   │   └── climate.py       # ClimateDataExtractor
│   │   ├── transformers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # DataTransformer ABC
│   │   │   ├── cleaning.py      # DataCleaner
│   │   │   └── normalization.py # DataNormalizer
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # DataLoader ABC
│   │   │   └── parquet_loader.py
│   │   ├── validators/
│   │   │   ├── __init__.py
│   │   │   └── quality_checks.py # Validações de qualidade
│   │   └── pipeline.py          # ETLPipeline orchestrator
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py       # Feature engineering
│   │   └── selection.py         # Feature selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # ModelStrategy ABC
│   │   ├── decision_tree.py     # DecisionTreeStrategy
│   │   └── random_forest.py     # RandomForestStrategy
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # MAE, RMSE, R² calculators
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── geographic.py        # Mapas (Basemap)
│   │   └── performance.py       # Gráficos de performance
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging estruturado
│       └── io.py                # I/O utilities
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └── conftest.py
├── notebooks/                    # Análises exploratórias
├── scripts/                      # Scripts de execução
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   ├── raw/                     # Dados brutos (gitignored)
│   ├── processed/               # Dados processados (gitignored)
│   └── external/                # Dados externos
├── models/                       # Modelos treinados (gitignored)
├── logs/                         # Logs de execução (gitignored)
├── configs/
│   └── config.yaml              # Configuração principal
├── .code-reviews/               # Histórico de code reviews
├── .gitignore
├── requirements.txt
├── setup.py
├── pyproject.toml
├── README.md
└── claude.md                    # Este arquivo

---
## 🚫 O QUE EVITAR
### Não Fazer
- ❌ Sugerir mudanças sem explicar o porquê
- ❌ Over-engineering (complicar desnecessariamente)
- ❌ Refatorações massivas de uma vez
- ❌ Introduzir dependências pesadas sem justificativa
- ❌ Usar jargão sem explicação
### Fazer
- ✅ Mudanças **incrementais** e **testáveis**
- ✅ Explicar **conceitos** de forma didática
- ✅ Mostrar código **antes/depois**
- ✅ Sugerir **testes** para validar
- ✅ Indicar **materiais** para aprender mais
---
## 📊 PRIORIZAÇÃO DE REFATORAÇÕES
### 🔥 Critical (fazer primeiro)
- Bugs ou problemas de corretude
- Violações graves de SOLID
- Código não testável
### 🎯 High (fazer em seguida)
- Performance crítica
- Código duplicado extenso
- Acoplamento alto
### 📈 Medium (depois)
- Legibilidade
- Documentação
- Otimizações menores
### 💅 Low (quando sobrar tempo)
- Estilo de código
- Convenções de nomenclatura
---
## 🎓 CONTEXTO DO DESENVOLVEDOR
### Perfil
- **Background:** Engenheiro Mecânico (UFRJ)
- **Experiência:** Python intermediário/avançado, data engineering
- **Objetivo:** Evoluir código acadêmico para produção
- **Foco:** Aprender arquitetura enterprise-grade
### Adapte explicações considerando:
- Base forte em engenharia e matemática
- Experiência em automação e otimização
- Interesse em boas práticas de software
- Documentando evolução no LinkedIn
---
## 📖 RECURSOS RECOMENDADOS
### Livros
- Clean Code (Robert C. Martin)
- Clean Architecture (Robert C. Martin)
- Design Patterns (Gang of Four)
- Fundamentals of Data Engineering (Joe Reis & Matt Housley)
### Online
- refactoring.guru (Design Patterns)
- realpython.com (Python Best Practices)
- python-patterns.guide (Python Patterns)
### Frameworks
- Great Expectations (Data Quality)
- Apache Airflow (Pipeline Orchestration)
- MLflow (ML Experiment Tracking)
---
## 🔗 INTEGRAÇÃO COM CONFIGURAÇÃO GLOBAL
Este arquivo complementa a configuração global em `~/.config/claude/`:
- **Personas:** `senior-mentor.md`
- **Skills:** `solid-principles.md`, `design-patterns.md`, `data-engineering.md`
- **Workflows:** `code-review-workflow.md`, `refactoring-workflow.md`
- **Hooks:** Pre-commit review, auto-commit

