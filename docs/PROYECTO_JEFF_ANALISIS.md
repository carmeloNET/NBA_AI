# Documentación Técnica del Proyecto NBA AI - Jeff

## 1. Modelos de Predicción Utilizados

El proyecto utiliza **3 modelos de Machine Learning** más un modelo base:

| Modelo | Algoritmo | MAE (2024-2025) | Archivo |
|--------|-----------|------------------|---------|
| **Baseline** | Fórmula PPG simple | - | No requiere modelo |
| **Linear** | Ridge Regression | 11.2 | `ridge_v0.4_mae11.2.joblib` |
| **Tree** | XGBoost | 10.1 | `xgboost_v0.4_mae10.1.joblib` |
| **MLP** | PyTorch Neural Network | 11.1 | `mlp_v0.4_mae11.1.pth` |
| **Ensemble** | Promedio ponderado (30% Linear + 40% Tree + 30% MLP) | - | No implementado completamente |

---

## 2. Arquitectura del Modelo de Machine Learning

### 2.1 Ridge Regression (Linear)
- **Hiperparámetros probados**: α ∈ {1.0, 10.0, 100.0}
- **Selección**: Se elige el mejor α basado en MAE en datos de validación
- **Preprocesamiento**: StandardScaler para normalizar features
- **Objetivo**: Predecir simultáneamente `home_score` y `away_score`

### 2.2 XGBoost (Tree)
- **Hiperparámetros fijos** (best practices de la literatura):
  - `n_estimators`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `gamma`: 0.1
  - `reg_lambda`: 1.0
- **Arquitectura**: MultiOutputRegressor (2 regresores, uno por cada objetivo)

### 2.3 MLP (PyTorch)
- **Arquitectura**: Input → 64 → ReLU → Dropout(0.2) → 32 → ReLU → Dropout(0.2) → 2
- **Normalización**: Features escalados + Targets normalizados (media=0, std=1)
- **Early stopping**: patience=25 epochs
- **Optimizador**: Adam con lr=0.001
- **Batch size**: 32
- **Épocas máx**: 200

---

## 3. Features (Campos) Utilizados para Predicciones

El sistema genera **43 features** por partido, organizados en 4 categorías:

### 3.1 Features Básicos (16 features)
Para cada equipo (home y away):
- `Home_Win_Pct` / `Away_Win_Pct` - Porcentaje de victorias
- `Home_PPG` / `Away_PPG` - Puntos por partido
- `Home_OPP_PPG` / `Away_OPP_PPG` - Puntos permitidos por partido
- `Home_Net_PPG` / `Away_Net_PPG` - Diferencia neta de puntos

### 3.2 Features Diferenciales (4 features)
- `Win_Pct_Diff` - Diferencia de porcentaje de victoria
- `PPG_Diff` - Diferencia de puntos por partido
- `OPP_PPG_Diff` - Diferencia de puntos permitidos
- `Net_PPG_Diff` - Diferencia neta

### 3.3 Features Home/Away (20 features)
Rendimiento específico como local vs visitante:
- `Home_Win_Pct_Home` - Victorias como local
- `Home_PPG_Home` - Puntos como local
- `Away_Win_Pct_Away` - Victorias como visitante
- `Away_PPG_Away` - Puntos como visitante

### 3.4 Features con Decimiento Temporal (20 features)
Similar a los básicos pero con ponderación exponencial (half_life=10 días):
- `Time_Decay_Home_Win_Pct`
- `Time_Decay_Home_PPG`
- etc.

### 3.5 Features de Descanso y Temporada (7 features)
- `Day_of_Season` - Día de la temporada
- `Home_Rest_Days` / `Away_Rest_Days` - Días de descanso
- `Home_Game_Freq` / `Away_Game_Freq` - Frecuencia de partidos
- `Rest_Days_Diff` - Diferencia de días de descanso
- `Game_Freq_Diff` - Diferencia de frecuencia

---

## 4. Proceso de Entrenamiento

### 4.1 Datos de Entrenamiento
- **Temporada de entrenamiento**: 2023-2024
- **Temporada de test**: 2024-2025
- **Cantidad**: ~1,300 juegos por temporada

### 4.2 Pipeline de Entrenamiento
```
1. Cargar datos featurizados desde la BD
2. Eliminar valores NaN
3. Separar features de targets (home_score, away_score)
4. Entrenar modelo con datos de entrenamiento
5. Evaluar en datos de test
6. Guardar modelo con versionado semántico
```

### 4.3 Métricas de Evaluación
- **MAE (Mean Absolute Error)**: Error absoluto medio por partido
- **Score MAE**: Promedio del MAE para home y away
- El mejor modelo (XGBoost) tiene MAE de ~10.1 puntos

### 4.4 Comando de Entrenamiento
```bash
# Entrenar todos los modelos
python -m src.model_training.train --model_type all --train_season 2023-2024 --test_season 2024-2025

# Entrenar un modelo específico
python -m src.model_training.train --model_type Tree --train_season 2023-2024 --test_season 2024-2025
```

---

## 5. Obtención de Datos

### 5.1 Fuentes de Datos

| Fuente | API/Scraping | Datos Obtenidos |
|--------|--------------|-----------------|
| **NBA API** | stats.nba.com | Schedule, PBP, Boxscores, Players |
| **NBA CDN** | cdn.nba.com | Play-by-play en tiempo real |
| **NBA Official** | PDFs injury reports | Lesiones diarias |
| **ESPN API** | site.api.espn.com | Líneas de apuestas |
| **Covers.com** | Web scraping | Líneas de apuestas históricas |

### 5.2 Pipeline de ETL

```
Schedule → Players → Injuries → Betting → PBP → GameStates → Boxscores → Features → Predictions
```

**Etapas detalladas**:
1. **Schedule**: Calendario de partidos desde NBA API
2. **Players**: Referencia de jugadores
3. **Injuries**: Scraping de PDFs de lesiones (NBA Official)
4. **Betting**: Líneas de apuestas (ESPN + Covers)
5. **PBP**: Play-by-play de cada partido
6. **GameStates**: Parseo de PBP en estados discretos
7. **Boxscores**: Estadísticas tradicionales
8. **Features**: Generación de features desde estados previos
9. **Predictions**: Ejecución de modelos predictivos

### 5.3 Volumen de Datos
- **Base de datos actual**: 1,302 juegos (2025-2026)
- **Base de datos dev**: 4,098 juegos (3 temporadas)
- **Base de datos full**: 37,366 juegos (27 temporadas)
- **PbP por juego**: ~492 jugadas promedio

---

## 6. Mercados de Apuestas Calculados

### 6.1 Líneas de Apuestas Recolectadas

El proyecto recopila **3 tipos de líneas** (opening, current, closing):

| Campo | Descripción |
|-------|-------------|
| `espn_opening_spread` | Spread de apertura (ESPN) |
| `espn_opening_total` | Total de apertura (ESPN) |
| `espn_opening_home_moneyline` | Moneyline home apertura |
| `espn_current_spread` | Spread actual (pre-partido) |
| `espn_current_total` | Total actual |
| `espn_closing_spread` | Spread de cierre (post-partido) |
| `espn_closing_total` | Total de cierre |
| `covers_closing_spread` | Spread de cierre (Covers) |
| `covers_closing_total` | Total de cierre (Covers) |

### 6.2 Resultados de Apuestas
- `spread_result`: 'W' (cubrió), 'L' (no cubrió), 'P' (push)
- `ou_result`: 'O' (over), 'U' (under), 'P' (push)

### 6.3 Predicciones del Modelo

El modelo predice:
- `pred_home_score`: Puntuación predicha del equipo local
- `pred_away_score`: Puntuación predicha del equipo visitante
- `pred_home_win_pct`: Probabilidad de victoria del local

### 6.4 Cómo Usar las Predicciones para Apuestas

**Spread**: Si `pred_home_score - pred_away_score > spread`, el home cubre

**Over/Under**: Si `pred_home_score + pred_away_score > total`, el over paga

**Nota**: El modelo NO calcula directamente el spread o total optimal. Solo predice puntuaciones. El usuario debe comparar con las líneas de apuestas.

---

## 7. Estructura de la Base de Datos

### Tablas Principales

| Tabla | Descripción |
|-------|-------------|
| `Games` | Calendario maestro (game_id, equipos, status) |
| `PbP_Logs` | Play-by-play raw JSON |
| `GameStates` | Estados parseados (~492 por juego) |
| `PlayerBox` | Estadísticas de jugador por juego |
| `TeamBox` | Estadísticas de equipo por juego |
| `Features` | 43 features por juego |
| `Predictions` | Predicciones de todos los modelos |
| `Betting` | Líneas de apuestas (ESPN + Covers) |
| `InjuryReports` | Lesiones oficiales NBA |
| `Players` | Referencia de jugadores |
| `Teams` | Referencia de equipos |

---

## 8. Uso del Sistema de Predicciones

### 8.1 Predicciones Pre-Juego
```python
from src.predictions.prediction_engines.tree_predictor import TreePredictor

predictor = TreePredictor(model_paths=["models/xgboost_v0.4_mae10.1.joblib"])
predictions = predictor.make_pre_game_predictions(["0022500981"])
```

### 8.2 Predicciones En-Vivo
El sistema mezcla la predicción pre-juego con el score actual:
```python
blend_factor = (time_remaining / total_time) ** 2
current_prediction = pre_game_score * blend_factor + current_score * (1 - blend_factor)
```

### 8.3 API Web
```bash
# Obtener juegos con predicciones
GET /api/games?date=2026-03-16&predictor=Tree

# Obtener juego específico
GET /api/games?game_ids=0022500981&predictor=Tree
```

---

## 9. Limitaciones y Mejoras Futuras

### Limitaciones Actuales
- **MAE de ~10 puntos**: Error promedio de 10 puntos por equipo
- **No considera lesiones**: Los features de lesiones no están integrados
- **No considera lineup específico**: Usa stats agregados del equipo
- **Solo predice score**: No predice spread/total directamente

### Mejoras Potenciales
- Integrar features de lesiones en el modelo
- Añadir datos de starters/lineup
- Entrenar con más temporadas
- Explorar modelos de deep learning más complejos
- Calibrar predicciones para mercados de apuestas específicos
