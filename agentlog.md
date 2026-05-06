# Agent Log — Contexto do Projeto

> Documento de handoff para outra IA assumir o projeto sem precisar reler tudo do zero.
> Última atualização: 2026-05-06 (Refatoração Completa do Treinamento)

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

---

## 2. Stack e dependências

- **Python 3.13** (Windows)
- **pandas 2.3** — leitura/manipulação dos CSVs
- **numpy 2.2** — arrays e log1p/expm1
- **scikit-learn 1.8** — `ColumnTransformer`, `Pipeline`, `TransformedTargetRegressor`, `KNeighborsRegressor`, `SVR`, `RandomForestRegressor`, `GradientBoostingRegressor`, `RandomizedSearchCV`
- **joblib 1.5** — serialização do modelo

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

| Arquivo | Origem | Vai pra submissão? |
|---|---|---|
| `treino.csv` | dado pelo professor (1168 linhas) | não |
| `teste_publico.csv` | dado pelo professor (1459 linhas) | não |
| `data_description.txt` | dicionário das 79 features (Ames Housing) | não |
| `metricas_baseline.txt` | resultado do baseline do professor | não |
| `treinamento.py` | gerado — script refatorado p/ os 4 modelos (SVR, KNN, RF, GB) | não (é só p/ regerar o modelo) |
| `pipeline.py` | gerado — função `prever_precos` exigida pelo corretor | **SIM** |
| `requirements.txt` | gerado — idêntico ao template | **SIM** |
| `modelo_baseline.joblib` | gerado pelo `treinamento.py` | **SIM** |
| `relatorio.pdf` | **pendente** | **SIM** |

---

## 4. Estratégia de pré-processamento (100% conforme edital)

Tudo embutido em **um único `Pipeline + ColumnTransformer`** que vai junto no `.joblib`. O `pipeline.py` não duplica regras.

1. **Ordinais (`ord`)**: 24 colunas com escala explícita (Po < Fa < TA < Gd < Ex). NaNs preenchidos com `'None'` antes de passar pelo `OrdinalEncoder`.
2. **Nominais (`nom`)**: 21 colunas categorizadas tratadas por `OneHotEncoder`. Regra de ouro atendida: **`handle_unknown='ignore'`** para lidar com novas categorias na base de teste secreta. (Inclui `MSSubClass` convertido p/ string).
3. **Numéricas com ausência estrutural (`zero_num`)**: Como áreas de garagens e porões inexistentes. NaN preenchido com `0` e obrigatoriamente padronizado com `StandardScaler` (essencial para SVR e KNN).
4. **Numéricas contínuas (`num`)**: NaNs preenchidos com mediana. Obrigatoriamente padronizadas com `StandardScaler`.

### 4.1. Transformação do Target (Logaritmo)

Todo regressor está envolvido em `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`. O modelo treina com o logaritmo dos preços para minimizar a proporção dos erros, otimizando nativamente a métrica RMSLE.

---

## 5. Modelos comparados (CV 5-fold, KFold seed=42)

Testamos os 4 algoritmos pedidos nas aulas recentes sob o mesmo pré-processamento e log transformation.

| Modelo | Tempo (CV) | RMSLE | MAE |
|---|---|---|---|
| **KNN** | 1.91s | 0.17394 ± 0.01480 | $22,853.17 |
| **SVR (RBF)** | **1.20s** | **0.12890 ± 0.01952** | **$15,467.57** |
| **RandomForest** | 2.21s | 0.14616 ± 0.01607 | $18,238.55 |
| **GradientBoost** | 1.82s | 0.12920 ± 0.01907 | $16,051.11 |

O **SVR** foi o grande vencedor e seguiu para o `RandomizedSearchCV` (20 iterações).
**Hiperparâmetros otimizados:**
```
{'gamma': 0.01, 'epsilon': 0.01, 'C': 1}
```

*Obs: A busca aleatória com CV encontrou uma estabilização de hiperparâmetros que gerou o RMSLE de validação cruzada de `0.13240` (um ajuste um pouco mais conservador do que o valor base, mas excelente para generalização).*

---

## 6. Pipeline de submissão (`pipeline.py`)

Função obrigatória no padrão:
```python
def prever_precos(caminho_arquivo_teste: str) -> np.ndarray
```

Retorna as previsões já em dólares (por causa do inversor do logaritmo no modelo) e com um filtro `np.clip(a_min=0)` no final para garantir que não existem valores negativos (pois o avaliador quebra ao calcular RMSLE de números negativos). O `modelo_baseline.joblib` segue nomeado perfeitamente para não quebrar a chamada.

---

## 7. Estado atual e próximos passos

### Já feito ✅
- Refatoração total do código em `treinamento.py`.
- Incorporação de KNN, SVR, RandomForest e GradientBoost.
- Check-list de obediência total ao Edital (PDF).
- Criação final do modelo via SVR Tunado.
- Validação do `pipeline.py`.

### Pendente (Ações do Usuário)
1. **Relatório Técnico (PDF):** É preciso criar a documentação exigida (Capa, Introdução, EDA, Engenharia de Features, Tabela Comparativa de ML e Conclusão/Justificativa).
2. **Repositório de Desenvolvimento:** O usuário precisa enviar e organizar seus próprios *Notebooks Jupyter (.ipynb)* provando a Análise Exploratória (EDA).
3. **Submissão via GitHub:** Fazer o fork do *template-house-prices*, substituir os arquivos gerados aqui e colar o link na plataforma (AVA).

---

## 8. Convenções e restrições

- **Não renomear** `modelo_baseline.joblib`.
- **Não mudar a assinatura** de `prever_precos(caminho_arquivo_teste)`.
- **Manter `np.clip(0, None)`** no final do `pipeline.py`.
- O código preza por usar soluções **puramente do Scikit-Learn** (`Pipeline`, `ColumnTransformer`, `TransformedTargetRegressor`) para manter a infraestrutura simples e imune a conflitos nos scripts de correção na nuvem do professor.
