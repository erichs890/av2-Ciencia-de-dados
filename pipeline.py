import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import mean_squared_log_error


def prever_precos(caminho_arquivo_teste):
    """
    Funcao obrigatoria para o corretor automatico.
    Le o arquivo de teste, aplica o pre-processamento (embutido no modelo
    via ColumnTransformer + TransformedTargetRegressor) e retorna as predicoes
    em dolares.

    Parametros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.

    Retorna:
    np.array: As predicoes de precos (nao negativas, em dolares).
    """
    # 1. Leitura dos dados de teste
    df_teste = pd.read_csv(caminho_arquivo_teste)

    # 2. Remove a coluna Id (nao e feature) e SalePrice (caso venha por engano)
    if "Id" in df_teste.columns:
        df_teste = df_teste.drop(columns=["Id"])
    if "SalePrice" in df_teste.columns:
        df_teste = df_teste.drop(columns=["SalePrice"])

    # 3. MSSubClass e codigo categorico (consistente com o treino)
    if "MSSubClass" in df_teste.columns:
        df_teste["MSSubClass"] = df_teste["MSSubClass"].astype(str)

    # 4. Carregamento do modelo (pipeline completo: preproc + regressor + log/expm1)
    caminho_modelo = "modelo_baseline.joblib"
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"O arquivo do modelo '{caminho_modelo}' nao foi encontrado na raiz do projeto."
        )
    modelo = joblib.load(caminho_modelo)

    # 5. Predicao. O ColumnTransformer interno seleciona/imputa/codifica as colunas
    #    corretas; o TransformedTargetRegressor ja inverte o log1p e devolve $.
    predicoes = modelo.predict(df_teste)

    # 6. Pos-processamento: garante valores >= 0 (RMSLE nao admite negativos)
    predicoes_finais = np.clip(predicoes, a_min=0, a_max=None)

    return predicoes_finais


if __name__ == "__main__":
    # Bloco de teste local para o aluno
    arquivo_teste_exemplo = "teste_publico.csv"

    print("--- Executando Validacao Local do Pipeline ---")

    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' nao encontrado.")
    else:
        try:
            resultados = prever_precos(arquivo_teste_exemplo)

            print("\n[OK] Pipeline rodou corretamente.")
            print("-" * 30)
            print("Primeiras 5 predicoes:")
            print(resultados[:5])
            print("-" * 30)

            df_val = pd.read_csv(arquivo_teste_exemplo)
            if "SalePrice" in df_val.columns:
                y_true = df_val["SalePrice"]
                rmsle = np.sqrt(mean_squared_log_error(y_true, resultados))
                print(f"Metrica RMSLE Local: {rmsle:.5f}")
            else:
                print("[Nota] Coluna 'SalePrice' nao encontrada. RMSLE pulado.")

        except Exception as e:
            print(f"\n[ERRO] no pipeline: {e}")
