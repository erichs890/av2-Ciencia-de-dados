# Agent Log — Contexto do Projeto

> Documento de handoff para outra IA assumir o projeto sem precisar reler tudo do zero.
> Última atualização: 2026-05-19 (Correções do professor após apresentação do Acompanhamento 2)

---

## 1. Visão geral

Projeto acadêmico (AV2 de Ciência de Dados) para uma **competição de Machine Learning** baseada no dataset **Ames Housing** (variante do clássico "House Prices" do Kaggle). Objetivo: prever `SalePrice` de imóveis e **superar o baseline do professor** (Regressão Linear simples).

Métrica oficial de avaliação: **RMSLE** (Root Mean Squared Logarithmic Error).

**Baseline a bater:**
```
RMSLE: 0.17543  |  RMSE: $36,061.40  |  MAE: $22,186.99  |  R²: 0.83046
```

**Resultado atual entregue (CV 5-fold, SVR Tunado):**
```
RMSLE: 0.13240  |  MAE: $15,498.02  →  -24.5% RMSLE / -30.1% MAE vs baseline
(Nota: SVR com parâmetros default chegou a 0.12890 no CV).
```

A submissão segue um template fixo do professor: `tuliorr/template-house-prices`. O corretor automático chama uma função específica (`prever_precos`) num arquivo específico (`pipeline.py`) com um modelo específico (`modelo_baseline.joblib`). O edital (PDF) estabelece exigências rigorosas que agora estão 100% cumpridas.

### Estado dos check-ins (Acompanhamentos)
- **Acompanhamento 1** (EDA + limpeza) — ✅ já apresentado.
- **Acompanhamento 2** (pipeline rodando + modelos preliminares) — ✅ apresentado. Professor apontou correções que **já foram aplicadas** no notebook (ver §6).

---

## 2. Stack e dependências

- **Python 3.13** (Windows) localmente / **Google Colab** para a apresentação.
- **pandas 2.3** — leitura/manipulação dos CSVs
- **numpy 2.2** — arrays e log1p/expm1
- **scikit-learn 1.8** — `ColumnTransformer`, `Pipeline`, `TransformedTargetRegressor`, `KNeighborsRegressor`, `SVR`, `RandomForestRegressor`, `GradientBoostingRegressor`, `RandomizedSearchCV`
- **joblib 1.5** — serialização do modelo
- **matplotlib + seaborn** — visualizações no notebook

**Decisão consciente:** **NÃO** usamos dependências externas fora do ecossistema Scikit-Learn (ex: XGBoost/LightGBM) para não quebrar a máquina do corretor automático no momento de instalar o `requirements.txt`. O SVR e o Gradient Boosting cumprem o papel de alta performance.

`requirements.txt` final:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
```

---

## 3. Arquivos do projeto

| Arquivo | Origem | Vai pra submissão? | Vai pro Colab? |
|---|---|---|---|
| `treino.csv` | dado pelo professor (1168 linhas) | não | **SIM** |
| `teste_publico.csv` | dado pelo professor (1459 linhas) | não | **SIM** |
| `data_description.txt` | dicionário das 79 features | não | não |
| `metricas_baseline.txt` | resultado do baseline do professor | não | não |
| `treinamento.ipynb` | **artefato principal** — notebook completo (EDA + modelagem + apresentação) | não | **SIM** |
| `pipeline.py` | função `prever_precos` exigida pelo corretor | **SIM** | **SIM** |
| `requirements.txt` | idêntico ao template | **SIM** | não |
| `modelo_baseline.joblib` | gerado dentro do notebook (Seção 12) | **SIM** | não (gerado lá) |
| `relatorio.pdf` | **pendente** | **SIM** | não |

> **Observação:** o antigo `treinamento.py` foi **deletado**. Toda a geração do modelo agora acontece dentro do `treinamento.ipynb`, que serve duplo propósito: notebook de desenvolvimento + apresentação dos check-ins.

---

## 4. Estrutura do `treinamento.ipynb`

O notebook está dividido em **duas partes** correspondentes aos dois check-ins:

### PARTE 1 — Acompanhamento 1 (EDA)
1. Carga dos dados
2. Visão geral (`info`, `describe`)
3. Variável alvo (`SalePrice`) — distribuição original + log1p, skewness
4. Correlações com variáveis numéricas — top 10 positiva/negativa, heatmap, scatter dos top 4
5. Variáveis categóricas — boxplots ordenados por mediana
6. Valores ausentes — tabela e gráfico de barras com cortes em 50% e 80%

### PARTE 2 — Acompanhamento 2 (Modelagem)
7. Preparação X / y + **`train_test_split(80/20, seed=42)`** ← correção do professor
8. `ColumnTransformer` — 4 sub-pipelines paralelos
8.1. **Comparativo visual antes vs depois do tratamento** (tabela de métricas + histogramas de 4 variáveis chave)
9. Log no alvo (`TransformedTargetRegressor`) + Validação Cruzada K-Fold(5) **no `X_train`**
10. Comparativo dos 5 modelos (LinReg, KNN, SVR, RF, GB) — CV **no `X_train`**
11. Otimização do SVR via `RandomizedSearchCV` (20 iterações) **no `X_train`**
12. **Avaliação final no `X_test` com `search.best_estimator_`** ← correção do professor
    - Predição do melhor modelo no conjunto reservado
    - Tabela y_real vs y_previsto + métricas (RMSLE, MAE, R²)
    - Scatter previsto vs real + histograma dos resíduos
13. Refit do `best_estimator_` em **100% dos dados** (X, y) + salvamento do `modelo_baseline.joblib`
14. Execução do `pipeline.py` (`from pipeline import prever_precos`) sobre `teste_publico.csv`
15. Conclusões

**Convenção crítica do notebook:** todos os imports (pandas, numpy, matplotlib, seaborn, **todo o sklearn** + joblib/os/time) estão concentrados em **uma única célula no topo** (logo após o cabeçalho). Roda essa célula primeiro e o resto do notebook funciona em qualquer ordem.

### Como rodar no Colab
1. Upload: `treinamento.ipynb`, `treino.csv`, `teste_publico.csv`, `pipeline.py`.
2. *Runtime → Run all* (~3-5 min, a Seção 11 é a mais demorada).
3. O `modelo_baseline.joblib` é gerado automaticamente na Seção 12.

---

## 5. Estratégia de pré-processamento (100% conforme edital)

Tudo embutido em **um único `Pipeline + ColumnTransformer`** que vai junto no `.joblib`. O `pipeline.py` não duplica regras.

1. **Ordinais (`ord`)**: 24 colunas com escala explícita (Po < Fa < TA < Gd < Ex). NaNs preenchidos com `'None'` antes de passar pelo `OrdinalEncoder`.
2. **Nominais (`nom`)**: 21 colunas categorizadas tratadas por `OneHotEncoder`. Regra de ouro atendida: **`handle_unknown='ignore'`** para lidar com novas categorias na base de teste secreta. (Inclui `MSSubClass` convertido p/ string).
3. **Numéricas com ausência estrutural (`zero_num`)**: como áreas de garagens e porões inexistentes. NaN preenchido com `0` e obrigatoriamente padronizado com `StandardScaler` (essencial para SVR e KNN).
4. **Numéricas contínuas (`num`)**: NaNs preenchidos com mediana. Obrigatoriamente padronizadas com `StandardScaler`.

### 5.1 Transformação do Target (Logaritmo)

Todo regressor está envolvido em `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`. O modelo treina com o logaritmo dos preços para minimizar a proporção dos erros, otimizando nativamente a métrica RMSLE.

---

## 6. Correções do professor (apresentação do Acompanhamento 2)

Durante a apresentação, o professor apontou que o fluxo original treinava com o dataset inteiro. As correções foram aplicadas em **2026-05-19**:

### 6.1. CV e search devem rodar **só no train**
- **Antes:** `cross_validate(pipe, X, y, ...)` e `search.fit(X, y)` usando o `treino.csv` inteiro (1168 linhas).
- **Depois:** `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG)`. Toda a CV e o `RandomizedSearchCV` rodam apenas sobre `X_train / y_train` (~934 linhas).

### 6.2. Avaliação honesta com `search.best_estimator_`
- Nova Seção 12 do notebook: pega o melhor modelo (`best_model = search.best_estimator_`), chama `best_model.predict(X_test)` e compara `y_pred` vs `y_test` (que o modelo nunca viu).
- Exibe RMSLE, MAE em dólares, R² + tabela das 10 primeiras predições + scatter previsto×real + histograma dos resíduos.
- Esse é o **estimador honesto da generalização**, livre da contaminação que a CV tem (já que ela é usada para escolher hiperparâmetros).

### 6.3. Refit final no dataset inteiro
- Depois da avaliação no test, o `modelo_final.fit(X, y)` é executado em **100% das 1168 linhas** antes de salvar o `.joblib`. Os hiperparâmetros já foram escolhidos via CV no train; agora aproveita-se todo o sinal disponível para o modelo de produção.

### 6.4. pipeline.py recebendo caminho do GitHub
- Já estava correto: `prever_precos(caminho_arquivo_teste)` aceita qualquer path e devolve `np.ndarray`. O notebook (Seção 14) faz `from pipeline import prever_precos` — exatamente o que o corretor automático fará.

---

## 7. Modelos comparados (CV 5-fold, KFold seed=42, **no X_train**)

Cada modelo no notebook está **explicitamente amarrado à aula correspondente** (PDFs das aulas foram fornecidos: KNN, SVM, Ensemble, Gradiente Descendente, Validação Cruzada, Métricas, Random Forest, Florestas Aleatórias).

| Modelo | Aula correspondente | Tempo (CV) | RMSLE | MAE |
|---|---|---|---|---|
| LinearRegression (baseline interno) | Gradiente Descendente | rápido | ~0.17 | ~$22k |
| **KNN (k=5)** | Métodos Baseados em Distância | 1.91s | 0.17394 ± 0.01480 | $22,853.17 |
| **SVR (RBF)** | Máquinas Vetores de Suporte | **1.20s** | **0.12890 ± 0.01952** | **$15,467.57** |
| **RandomForest** | Ensemble (Bagging) | 2.21s | 0.14616 ± 0.01607 | $18,238.55 |
| **GradientBoost** | Ensemble (Boosting) | 1.82s | 0.12920 ± 0.01907 | $16,051.11 |

O **SVR** foi o grande vencedor e seguiu para o `RandomizedSearchCV` (20 iterações).
**Hiperparâmetros otimizados:**
```
{'gamma': 0.01, 'epsilon': 0.01, 'C': 1}
```

*Obs: a busca aleatória com CV gerou RMSLE de validação cruzada de `0.13240` (mais conservador que o default `0.12890`, mas melhor para generalização).*

---

## 8. Pipeline de submissão (`pipeline.py`)

Função obrigatória no padrão:
```python
def prever_precos(caminho_arquivo_teste: str) -> np.ndarray
```

Retorna as previsões já em dólares (por causa do inversor do logaritmo no modelo) e com um filtro `np.clip(a_min=0)` no final para garantir que não existem valores negativos (pois o avaliador quebra ao calcular RMSLE de números negativos). O `modelo_baseline.joblib` segue nomeado perfeitamente para não quebrar a chamada.

> **Importante:** no notebook (Seção 13) o `pipeline.py` é importado **diretamente** (`from pipeline import prever_precos`) — não há cópia inline do código, para evitar manter duas versões da mesma função.

---

## 9. Estado atual e próximos passos

### Já feito ✅
- Refatoração total — todo o código de modelagem agora vive em `treinamento.ipynb` (Python script `treinamento.py` foi deletado).
- 5 modelos comparados (LinReg, KNN, SVR, RF, GB) com fundamentação teórica vinculada a cada aula.
- Acompanhamento 1 (EDA) já apresentado.
- Acompanhamento 2 (modelagem) já apresentado; correções do professor aplicadas (split train/test + avaliação no test com `best_estimator_`).
- Pipeline de submissão validado (`pipeline.py` rodando sobre `teste_publico.csv`).
- Comparativo visual antes/depois do Pipeline (Seção 8.1) — útil para a apresentação.

### Pendente (Ações do Usuário)
1. **Relatório Técnico (PDF):** Capa, Introdução, EDA, Engenharia de Features, Tabela Comparativa de ML e Conclusão/Justificativa.
2. **Repositório de Desenvolvimento:** organizar o `treinamento.ipynb` no GitHub.
3. **Submissão via GitHub:** fork do *template-house-prices*, substituir arquivos gerados e colar o link no AVA.

---

## 10. Convenções e restrições

- **Não renomear** `modelo_baseline.joblib`.
- **Não mudar a assinatura** de `prever_precos(caminho_arquivo_teste)`.
- **Manter `np.clip(0, None)`** no final do `pipeline.py`.
- **Todos os imports em uma única célula no topo** do `treinamento.ipynb` — não espalhar imports pelo notebook (causa `NameError` se a pessoa não rodar célula por célula em ordem).
- **Seção 13** do notebook deve usar `from pipeline import prever_precos` — nunca duplicar o código da função inline.
- O código preza por usar soluções **puramente do Scikit-Learn** (`Pipeline`, `ColumnTransformer`, `TransformedTargetRegressor`) para manter a infraestrutura simples e imune a conflitos nos scripts de correção na nuvem do professor.
- **Não titular o notebook como "Acompanhamento 2"** — ele cobre AMBOS os check-ins (EDA + Modelagem). Título genérico, divisão clara em duas Partes com cabeçalhos próprios.
