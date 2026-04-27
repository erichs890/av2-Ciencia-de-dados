"""
TREINAMENTO E VALIDAÇÃO — Competição House Prices (Ames).
Métrica oficial: RMSLE.

Estrutura:
1. Carrega treino.csv
2. Define ColumnTransformer (ordinal + one-hot + numérico)
3. Compara Ridge / RandomForest / HistGradientBoosting via CV (MAE em $ e RMSLE)
4. Faz RandomizedSearch no campeão
5. Salva 'modelo_baseline.joblib' (nome exigido pelo pipeline.py do template)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_log_error


# =========================================================
# 1) CARGA
# =========================================================
df = pd.read_csv("treino.csv")
y = df["SalePrice"].astype(float)
X = df.drop(columns=["Id", "SalePrice"])

# MSSubClass é numérica no CSV mas é semanticamente categórica
X["MSSubClass"] = X["MSSubClass"].astype(str)


# =========================================================
# 2) DEFINIÇÃO DOS GRUPOS DE COLUNAS
# =========================================================

# 2a) Categóricas onde NaN = ausência da característica.
#     Preenchemos com 'None' antes do encoder.
NONE_CATEGORICAL = [
    "Alley", "MasVnrType",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
]

# 2b) Numéricas onde NaN = ausência → preencher com 0
ZERO_NUMERIC = [
    "GarageYrBlt", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
]

# 2c) Ordinais — ORDEM EXPLÍCITA (do pior ao melhor).
#     Inclui 'None' como categoria mais baixa onde aplicável.
ORDINAL_MAPS = {
    "ExterQual":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual":      ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond":      ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure":  ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1":  ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2":  ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual":   ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish":  ["None", "Unf", "RFn", "Fin"],
    "GarageQual":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC":        ["None", "Fa", "TA", "Gd", "Ex"],
    "Fence":         ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional":    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LotShape":      ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope":     ["Sev", "Mod", "Gtl"],
    "PavedDrive":    ["N", "P", "Y"],
    "Street":        ["Grvl", "Pave"],
    "Alley":         ["None", "Grvl", "Pave"],
    "Utilities":     ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "CentralAir":    ["N", "Y"],
}
ORDINAL_COLS = list(ORDINAL_MAPS.keys())
ORDINAL_CATEGORIES = [ORDINAL_MAPS[c] for c in ORDINAL_COLS]

# 2d) Nominais (one-hot). MSSubClass entra aqui (já convertido p/ string).
NOMINAL_COLS = [
    "MSSubClass", "MSZoning", "LandContour", "LotConfig", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating",
    "Electrical", "GarageType", "MiscFeature", "SaleType", "SaleCondition",
]

# 2e) Numéricas contínuas — todo o resto numérico (e LotFrontage, que tratamos com mediana)
ALL_NUMERIC = X.select_dtypes(include=[np.number]).columns.tolist()
NUMERIC_COLS = [c for c in ALL_NUMERIC if c not in ZERO_NUMERIC]
# LotFrontage entra aqui — mediana global é boa o suficiente; bairro seria marginal.


# =========================================================
# 3) PIPELINES POR GRUPO
# =========================================================

# Para preencher NaN com 'None' antes de cair no encoder ordinal
none_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_pipe = Pipeline([
    ("imputer", none_imputer),
    ("ordinal", OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

nominal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

zero_numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler()),
])

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categóricas estruturais ('None' → ordinal? não — são nominais misturadas)
# Algumas delas (FireplaceQu, BsmtQual, etc.) já estão em ORDINAL_COLS.
# As que sobram em NONE_CATEGORICAL e NÃO são ordinais nem nominais já listadas
# já são tratadas pelo nominal_pipe via 'most_frequent', mas o NaN aqui tem
# significado. Então: para essas, queremos 'None' literal, não a moda.
remaining_none_cols = [c for c in NONE_CATEGORICAL
                       if c not in ORDINAL_COLS and c not in NOMINAL_COLS]
# (vazio na prática — todas já estão cobertas; deixamos como salvaguarda)

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ordinal_pipe, ORDINAL_COLS),
        ("nom", nominal_pipe, NOMINAL_COLS),
        ("zero_num", zero_numeric_pipe, ZERO_NUMERIC),
        ("num", numeric_pipe, NUMERIC_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)


# =========================================================
# 4) MODELOS — TODOS ENVELOPADOS EM TransformedTargetRegressor (log1p)
# =========================================================
def wrap_log(model):
    """Treina em log1p(SalePrice) e retorna predição em $ via expm1."""
    return TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
    )

modelos = {
    "Ridge":           wrap_log(Ridge(alpha=10.0, random_state=42)),
    "RandomForest":    wrap_log(RandomForestRegressor(
                            n_estimators=400, max_features="sqrt",
                            min_samples_leaf=2, n_jobs=-1, random_state=42)),
    "HistGradBoost":   wrap_log(HistGradientBoostingRegressor(
                            max_iter=600, learning_rate=0.05,
                            max_leaf_nodes=31, min_samples_leaf=20,
                            l2_regularization=1.0, random_state=42)),
}


# =========================================================
# 5) MÉTRICAS
# =========================================================
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)  # RMSLE não admite negativos
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))

scorer_rmsle = make_scorer(rmsle, greater_is_better=False)
scorer_mae   = make_scorer(mean_absolute_error, greater_is_better=False)

cv = KFold(n_splits=5, shuffle=True, random_state=42)


# =========================================================
# 6) CROSS-VALIDATION COMPARATIVO
# =========================================================
print("=" * 70)
print("BASELINE DO PROFESSOR (Linear Regression):")
print("  RMSLE: 0.17543  |  MAE: $22,186.99  |  R²: 0.83046")
print("=" * 70)
print()
print("Cross-Validation (5-fold) — métricas em $ (MAE) e RMSLE:")
print("-" * 70)

resultados = {}
for nome, pipe_modelo in modelos.items():
    full_pipe = Pipeline([("prep", preprocessor), ("model", pipe_modelo)])

    rmsle_scores = -cross_val_score(full_pipe, X, y, cv=cv,
                                    scoring=scorer_rmsle, n_jobs=-1)
    mae_scores   = -cross_val_score(full_pipe, X, y, cv=cv,
                                    scoring=scorer_mae, n_jobs=-1)

    resultados[nome] = {
        "rmsle_mean": rmsle_scores.mean(),
        "rmsle_std":  rmsle_scores.std(),
        "mae_mean":   mae_scores.mean(),
        "mae_std":    mae_scores.std(),
    }

    print(f"{nome:>15} | RMSLE = {rmsle_scores.mean():.5f} ± {rmsle_scores.std():.5f} "
          f"| MAE = ${mae_scores.mean():>10,.2f} ± ${mae_scores.std():,.2f}")

print("-" * 70)
melhor = min(resultados, key=lambda k: resultados[k]["rmsle_mean"])
print(f"\n>>> Melhor modelo (menor RMSLE): {melhor}\n")


# =========================================================
# 7) TUNING NO CAMPEÃO
# =========================================================
print("Iniciando RandomizedSearchCV no campeão...")

if melhor == "HistGradBoost":
    param_dist = {
        "model__regressor__learning_rate":   [0.03, 0.05, 0.07, 0.1],
        "model__regressor__max_iter":        [400, 600, 900, 1200],
        "model__regressor__max_leaf_nodes":  [15, 31, 63],
        "model__regressor__min_samples_leaf":[10, 20, 30, 50],
        "model__regressor__l2_regularization":[0.0, 0.5, 1.0, 2.0],
    }
    base = wrap_log(HistGradientBoostingRegressor(random_state=42))
elif melhor == "RandomForest":
    param_dist = {
        "model__regressor__n_estimators":     [300, 500, 800],
        "model__regressor__max_features":     ["sqrt", 0.3, 0.5],
        "model__regressor__min_samples_leaf": [1, 2, 4],
        "model__regressor__max_depth":        [None, 20, 30],
    }
    base = wrap_log(RandomForestRegressor(n_jobs=-1, random_state=42))
else:  # Ridge
    param_dist = {"model__regressor__alpha": [0.1, 1.0, 5.0, 10.0, 30.0, 100.0]}
    base = wrap_log(Ridge(random_state=42))

full_pipe_tune = Pipeline([("prep", preprocessor), ("model", base)])

search = RandomizedSearchCV(
    full_pipe_tune, param_distributions=param_dist,
    n_iter=20, cv=cv, scoring=scorer_rmsle, n_jobs=-1, random_state=42,
)
search.fit(X, y)

print(f"Melhores parâmetros: {search.best_params_}")
print(f"RMSLE pós-tuning (CV): {-search.best_score_:.5f}")

# Métricas finais do modelo tunado (CV)
final_rmsle = -cross_val_score(search.best_estimator_, X, y, cv=cv,
                               scoring=scorer_rmsle, n_jobs=-1)
final_mae   = -cross_val_score(search.best_estimator_, X, y, cv=cv,
                               scoring=scorer_mae, n_jobs=-1)
print(f"\n=== MODELO FINAL (após tuning) ===")
print(f"RMSLE (CV 5-fold): {final_rmsle.mean():.5f} ± {final_rmsle.std():.5f}")
print(f"MAE   (CV 5-fold): ${final_mae.mean():,.2f} ± ${final_mae.std():,.2f}")
print(f"\nBaseline professor: RMSLE 0.17543 | MAE $22,186.99")
print(f"Melhoria RMSLE: {(0.17543 - final_rmsle.mean())/0.17543*100:.1f}%")
print(f"Melhoria MAE:   {(22186.99 - final_mae.mean())/22186.99*100:.1f}%")


# =========================================================
# 8) FIT FINAL EM TODO O TREINO E SAVE
# =========================================================
modelo_final = search.best_estimator_
modelo_final.fit(X, y)

joblib.dump(modelo_final, "modelo_baseline.joblib")
print("\n[OK] Modelo salvo como 'modelo_baseline.joblib' (nome exigido pelo pipeline.py).")
