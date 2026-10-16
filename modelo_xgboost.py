"""
Módulo de Implementação e Avaliação do Modelo XGBoost (Gradient Boosting Machine)
Desafio Millai / Klike - Modelo Preditivo de Qualidade de Criativos (klike_score)

Este arquivo:
1. Conecta-se ao pipeline de preparação dos dados (`pipeline_preparacao.py`);
2. Instancia e treina o estimador XGBoost Regressor (XGBRegressor);
3. Disponibiliza uma função dedicada e linhas isoladas para avaliação no conjunto de teste;
4. Compara valores reais vs. valores previstos pelo modelo.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Importação dos dados preparados e do pipeline modular
from pipeline_preparacao import (
    X_test_prep,
    X_train_prep,
    feature_names,
    y_test,
    y_train,
)


def instanciar_modelo_xgboost() -> xgb.XGBRegressor:
    """
    Instancia o estimador XGBoost Regressor com hiperparâmetros calibrados
    para o tamanho e características do dataset de criativos.
    """
    modelo = xgb.XGBRegressor(
        n_estimators=100,      # Número de árvores sequenciais
        learning_rate=0.08,    # Taxa de aprendizado (passo do gradiente)
        max_depth=4,           # Profundidade máxima por árvore (evita overfitting)
        subsample=0.85,        # Fração de amostras por árvore
        colsample_bytree=0.85, # Fração de features por árvore
        random_state=42,       # Semente para reprodutibilidade
        objective='reg:squarederror'
    )
    return modelo


def treinar_modelo(modelo: xgb.XGBRegressor, X_train: np.ndarray, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Treina o modelo XGBoost estritamente nos dados de treinamento.
    """
    modelo.fit(X_train, y_train)
    return modelo


def testar_modelo(modelo: xgb.XGBRegressor, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """
    Avalia o modelo no conjunto de teste e calcula as principais métricas de regressão:
    - R² (Capacidade explicativa da variância)
    - RMSE (Penaliza erros grandes)
    - MAE (Erro médio em pontos de score)
    """
    # Linha de predição no conjunto de teste
    y_pred = modelo.predict(X_test)

    metricas = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'y_pred': y_pred
    }

    return metricas


def exibir_tabela_comparativa(y_test: pd.Series, y_pred: np.ndarray, n_amostras: int = 10) -> pd.DataFrame:
    """
    Gera um DataFrame comparando lado a lado os valores reais e as predições.
    """
    df_comparativo = pd.DataFrame({
        'klike_score Real': y_test.values[:n_amostras],
        'klike_score Previsto': np.round(y_pred[:n_amostras], 2),
        'Erro Absoluto (Pontos)': np.round(np.abs(y_test.values[:n_amostras] - y_pred[:n_amostras]), 2)
    })
    return df_comparativo


# ==============================================================================
# EXECUÇÃO DO TREINAMENTO E TESTE
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 TREINAMENTO DO MODELO XGBOOST (GBM)")
    print("=" * 60)

    # 1. Instanciação e Treinamento
    modelo_xgb = instanciar_modelo_xgboost()
    treinar_modelo(modelo_xgb, X_train_prep, y_train)
    print("✅ Modelo XGBoost treinado com sucesso no conjunto de treino (400 amostras)!")

    # 2. LINHA SEPARADA DE TESTE COM O CONJUNTO DE TESTE (100 amostras não vistas)
    print("\n" + "=" * 60)
    print("🎯 TESTANDO O MODELO COM O CONJUNTO DE TESTE")
    print("=" * 60)
    
    # Execução da predição e cálculo das métricas de teste
    resultados = testar_modelo(modelo_xgb, X_test_prep, y_test)

    print(f"📊 Coeficiente de Determinação (R²): {resultados['R²']:.4f}  (Explica ~{resultados['R²']*100:.1f}% da variância)")
    print(f"📏 Raiz do Erro Quadrático Médio (RMSE): {resultados['RMSE']:.2f} pontos")
    print(f"📉 Erro Médio Absoluto (MAE)           : {resultados['MAE']:.2f} pontos")

    # 3. Comparativo de Amostras Reais vs Previstas
    print("\n🔍 Comparação das 10 Primeiras Amostras do Conjunto de Teste:")
    tabela_teste = exibir_tabela_comparativa(y_test, resultados['y_pred'], n_amostras=10)
    print(tabela_teste.to_string(index=True))

    # 4. Importância das Features no XGBoost
    importancias = pd.DataFrame({
        'Feature': feature_names,
        'Importância': modelo_xgb.feature_importances_
    }).sort_values(by='Importância', ascending=False)

    print("\n🏆 Top 5 Features Mais Determinantes para o XGBoost:")
    print(importancias.head(5).to_string(index=False))
