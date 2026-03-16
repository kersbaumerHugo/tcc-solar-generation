# scripts/run_training.py
"""
Script principal de treinamento dos modelos

Este é o equivalente REFATORADO do modelo_final.py
"""
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.data.loaders import DataLoader
from src.data.processors import DataProcessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
import pandas as pd

def main():
    logger.info("=" * 60)
    logger.info("PIPELINE DE TREINAMENTO - Previsão de Geração Solar")
    logger.info("=" * 60)
    
    # Configuração
    data_dir = Path("data")
    results = []
    
    # 1. Carregar dados
    loader = DataLoader(data_dir)
    datasets = loader.load_full_datasets()
    
    # 2. Inicializar processador, trainer e evaluator
    processor = DataProcessor()
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    evaluator = ModelEvaluator()
    
    # 3. Loop por usina
    for usina_name, df_raw in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando: {usina_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Processar dados
            df = processor.rename_columns(df_raw)
            df = processor.add_temporal_features(df)
            df = processor.handle_missing_values(df)
            df = processor.normalize_data(df)
            
            # Verificar tamanho mínimo
            if len(df) < trainer.min_samples:
                logger.warning(f"Dataset muito pequeno: {len(df)} < {trainer.min_samples}")
                continue
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = trainer.split_data(df)
            
            # Treinar Decision Tree
            dt_model, dt_cv_results = trainer.train_decision_tree(X_train, y_train)
            dt_metrics = evaluator.evaluate(dt_model, X_test, y_test, "Decision Tree")
            
            # Treinar Random Forest
            rf_model, rf_cv_results = trainer.train_random_forest(X_train, y_train)
            rf_metrics = evaluator.evaluate(rf_model, X_test, y_test, "Random Forest")
            
            # Armazenar resultados
            results.append({
                "usina": usina_name,
                "n_samples": len(df),
                "dt_r2": dt_metrics["r2"],
                "rf_r2": rf_metrics["r2"]
            })
            
        except Exception as e:
            logger.error(f"Erro ao processar {usina_name}: {e}", exc_info=True)
            continue
    
    # 4. Resumo final
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS FINAIS")
    logger.info("=" * 60)
    
    df_results = pd.DataFrame(results)
    logger.info(f"\n{df_results}")
    
    # Salvar resultados
    df_results.to_csv("results/training_results.csv", index=False)
    logger.info("\nResultados salvos em: results/training_results.csv")

if __name__ == "__main__":
    main()