"""
TREINAMENTO E VALIDAÇÃO — Competição House Prices (Ames).
Métrica oficial: RMSLE.

Estrutura:
1. Carrega treino.csv
2. Define ColumnTransformer (ordinal + one-hot + numérico) garantindo StandardScaler nas numéricas.
3. Compara KNN / SVR / RandomForest / GradientBoosting via CV (Tempo, MAE em $ e RMSLE)
4. Faz RandomizedSearchCV no campeão
5. Salva 'modelo_baseline.joblib'
"""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
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

# 2d) Nominais (one-hot).
NOMINAL_COLS = [
    "MSSubClass", "MSZoning", "LandContour", "LotConfig", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating",
    "Electrical", "GarageType", "MiscFeature", "SaleType", "SaleCondition",
]

# 2e) Numéricas contínuas
ALL_NUMERIC = X.select_dtypes(include=[np.number]).columns.tolist()
NUMERIC_COLS = [c for c in ALL_NUMERIC if c not in ZERO_NUMERIC]


# =========================================================
# 3) PIPELINES POR GRUPO (Com StandardScaler nas numéricas)
# =========================================================

none_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_pipe = Pipeline([
    ("imputer", none_imputer),
    ("ordinal", OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

# Importante: handle_unknown='ignore'
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
# 4) MODELOS (TransformedTargetRegressor)
# =========================================================
def wrap_log(model):
    """Envolve regressor com log1p para treinamento e expm1 para predição."""
    return TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
    )

modelos = {
    "KNN":             wrap_log(KNeighborsRegressor()),
    "SVR":             wrap_log(SVR(kernel='rbf')),
    "RandomForest":    wrap_log(RandomForestRegressor(random_state=42, n_jobs=-1)),
    "GradientBoost":   wrap_log(GradientBoostingRegressor(random_state=42)),
}


# =========================================================
# 5) MÉTRICAS
# =========================================================
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)  # Previne valores negativos pro log
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))

scorer_rmsle = make_scorer(rmsle, greater_is_better=False)
scorer_mae   = make_scorer(mean_absolute_error, greater_is_better=False)

scoring = {"rmsle": scorer_rmsle, "mae": scorer_mae}
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# =========================================================
# 6) CROSS-VALIDATION COMPARATIVO
# =========================================================
print("=" * 80)
print("Cross-Validation (5-fold) — Comparando Algoritmos (Métricas em $ e RMSLE):")
print("-" * 80)

resultados = {}
for nome, pipe_modelo in modelos.items():
    full_pipe = Pipeline([("prep", preprocessor), ("model", pipe_modelo)])
    
    start_time = time.time()
    cv_results = cross_validate(full_pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    elapsed_time = time.time() - start_time
    
    rmsle_scores = -cv_results["test_rmsle"]
    mae_scores   = -cv_results["test_mae"]

    resultados[nome] = {
        "rmsle_mean": rmsle_scores.mean(),
        "rmsle_std":  rmsle_scores.std(),
        "mae_mean":   mae_scores.mean(),
        "mae_std":    mae_scores.std(),
        "time":       elapsed_time
    }

    print(f"{nome:>15} | Tempo (CV): {elapsed_time:>5.2f}s | RMSLE = {rmsle_scores.mean():.5f} ± {rmsle_scores.std():.5f} | MAE = ${mae_scores.mean():>10,.2f}")

print("-" * 80)
melhor = min(resultados, key=lambda k: resultados[k]["rmsle_mean"])
print(f"\n>>> Melhor modelo base (menor RMSLE): {melhor}\n")


# =========================================================
# 7) TUNING NO CAMPEÃO
# =========================================================
print("Iniciando RandomizedSearchCV no modelo campeão...")

if melhor == "KNN":
    param_dist = {
        "model__regressor__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__regressor__weights":     ['uniform', 'distance'],
        "model__regressor__p":           [1, 2],
    }
    base = wrap_log(KNeighborsRegressor())
elif melhor == "SVR":
    param_dist = {
        "model__regressor__C":       [0.1, 1, 10, 50, 100, 500, 1000],
        "model__regressor__gamma":   ['scale', 'auto', 0.01, 0.1],
        "model__regressor__epsilon": [0.01, 0.05, 0.1, 0.2],
    }
    base = wrap_log(SVR(kernel='rbf'))
elif melhor == "RandomForest":
    param_dist = {
        "model__regressor__n_estimators":     [100, 300, 500, 800],
        "model__regressor__max_depth":        [None, 10, 20, 30],
        "model__regressor__min_samples_leaf": [1, 2, 4],
    }
    base = wrap_log(RandomForestRegressor(random_state=42, n_jobs=-1))
else: # GradientBoost
    param_dist = {
        "model__regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__regressor__n_estimators":  [100, 300, 500, 800],
        "model__regressor__max_depth":     [3, 4, 5, 7],
    }
    base = wrap_log(GradientBoostingRegressor(random_state=42))

full_pipe_tune = Pipeline([("prep", preprocessor), ("model", base)])

search = RandomizedSearchCV(
    full_pipe_tune, param_distributions=param_dist,
    n_iter=20, cv=cv, scoring=scorer_rmsle, n_jobs=-1, random_state=42,
    error_score="raise"
)

# Mede o tempo de tuning
t0_tune = time.time()
search.fit(X, y)
t_tune = time.time() - t0_tune

print(f"Tempo de tuning: {t_tune:.2f}s")
print(f"Melhores parâmetros: {search.best_params_}")
print(f"RMSLE pós-tuning (CV): {-search.best_score_:.5f}")

# Métricas finais do modelo tunado via cross_validate (apenas para exibição final)
cv_final = cross_validate(search.best_estimator_, X, y, cv=cv, scoring=scoring, n_jobs=-1)
final_rmsle = -cv_final["test_rmsle"]
final_mae   = -cv_final["test_mae"]

print(f"\n=== MODELO FINAL (após tuning) ===")
print(f"RMSLE (CV 5-fold): {final_rmsle.mean():.5f} ± {final_rmsle.std():.5f}")
print(f"MAE   (CV 5-fold): ${final_mae.mean():,.2f} ± ${final_mae.std():,.2f}")


# =========================================================
# 8) FIT FINAL EM TODO O TREINO E EXPORTAÇÃO
# =========================================================
modelo_final = search.best_estimator_
joblib.dump(modelo_final, "modelo_baseline.joblib")
print("\n[OK] Modelo selecionado com menor RMSLE salvo como 'modelo_baseline.joblib'.")
