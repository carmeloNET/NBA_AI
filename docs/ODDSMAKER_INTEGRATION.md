# Análisis: ¿NBA AI sirve para un Proyecto de Oddsmaker?

## Respuesta Corta

**SÍ, el proyecto proporciona las bases**, pero **NO está diseñado específicamente para oddsmakería**. Te explico exactamente qué puedes usar y qué necesitarías agregar.

---

## 1. Lo que NBA AI YA te ofrece

### ✅ Lo que puedes aprovechar

| Componente | Utilidad para Oddsmaker |
|------------|------------------------|
| **Modelos ML** (XGBoost, Ridge, MLP) | Predicen puntuaciones de partidos |
| **Features** (43 variables) | Stats históricos de equipos para entrenar tus propios modelos |
| **Pipeline de datos** | Automatiza obtención de datos NBA desde NBA API |
| **Predicción de score** | `pred_home_score`, `pred_away_score` |
| **Probabilidad implícita** | `pred_home_win_pct` calculada con función logística |
| **Datos de líneas** | Recopila líneas de ESPN y Covers |

### ❌ Lo que NO tiene (y necesitarías)

| Lo que falta | Por qué es importante |
|--------------|----------------------|
| **Cuotas sin vig** | No calcula "fair odds" o probabilidades "reales" |
| **Modelos para mercados** | No hay modelos para predecir spread, total, moneyline directamente |
| **Calibración de probabilidades** | Las probabilidades no están calibradas a eventos reales |
| **Márgenes/Vig** | No hay lógica para añadir overround |
| **Mercados adicionales** | No predice props, futures, parlays |

---

## 2. Arquitectura Recomendada para tu Oddsmaker

Basado en NBA AI, así podrías estructurar tu proyecto:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TU PROYECTO ODDSMAKER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐ │
│  │  NBA AI (existing)│    │  NUEVO: Módulo de Cuotas        │ │
│  ├──────────────────┤    ├──────────────────────────────────┤ │
│  │                  │    │  1. Fair Odds Calculator          │ │
│  │  • Datos NBA     │    │     └─ Quitar vig (reverse)      │ │
│  │  • Features      │    │                                  │ │
│  │  • Predicciones  │────▶  2. Vig Adder                   │ │
│  │    (scores)      │    │     └─ Añadir juice             │ │
│  │                  │    │                                  │ │
│  │  • Líneas (ESPN) │    │  3. Market Converters            │ │
│  │                  │    │     ├─ Score → Spread            │ │
│  └──────────────────┘    │     ├─ Score → Total            │ │
│                          │     └─ Score → Moneyline         │ │
│                          │                                  │ │
│                          │  4. Odds Comparator             │ │
│                          │     └─ Tu cuota vs mercado       │ │
│                          │                                  │ │
│                          └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementación: Cómo generar cuotas base

### 3.1 Paso 1: Obtener predicción de score

```python
from src.predictions.prediction_engines.tree_predictor import TreePredictor

predictor = TreePredictor(model_paths=["models/xgboost_v0.4_mae10.1.joblib"])
predictions = predictor.make_pre_game_predictions(["0022500981"])

pred_home_score = predictions["0022500981"]["pred_home_score"]  # ej: 115.5
pred_away_score = predictions["0022500981"]["pred_away_score"]  # ej: 112.3
```

### 3.2 Paso 2: Calcular probabilidad desde score (sin vig)

NBA AI usa esta fórmula logística (en `prediction_utils.py`):

```python
import numpy as np

def score_to_probability(home_score, away_score):
    """
    Convierte scores predichos a probabilidad.
    IMPORTANTE: Esta probabilidad INCLUYE el sesgo del modelo,
    NO es una probabilidad "fair" o real.
    """
    base_a = -0.2504  # Intercept
    base_b = 0.1949   # Coeficiente
    score_diff = home_score - away_score
    
    # Probabilidad "bruta" del modelo
    prob = 1 / (1 + np.exp(-(base_a + base_b * score_diff)))
    return prob
```

**Problema**: Esta probabilidad está sesgada por el modelo y no representa cuotas reales.

### 3.3 Paso 3: Remover el Vig (Reverse Juice)

Para obtener cuotas "justas" o "fair odds", necesitas quitar el vig:

```python
def remove_vig(prob_home, prob_away):
    """
    Remueve el vig de las probabilidades.
    
    Args:
        prob_home: Probabilidad implícita del home (incluye vig)
        prob_away: Probabilidad implícita del away (incluye vig)
    
    Returns:
        tuple: (fair_prob_home, fair_prob_away) sin vig
    """
    # Sumar probabilidades
    total = prob_home + prob_away
    
    # Calcular el "overround" (vig total)
    overround = total - 1.0  # Si es 0.05, hay 5% de vig
    
    # Remover vig proporcionalmente
    fair_home = prob_home / total
    fair_away = prob_away / total
    
    return fair_home, fair_away

def prob_to_decimal_odds(probability):
    """
    Convierte probabilidad a cuotas decimales.
    fair_odds = 1 / probability
    """
    return 1 / probability

# Ejemplo:
prob_home = 0.55  # Del modelo (incluye vig)
prob_away = 0.50  # Del modelo

fair_home, fair_away = remove_vig(prob_home, prob_away)
print(f"Fair Home: {fair_home:.3f} → Cuota: {prob_to_decimal_odds(fair_home):.2f}")
print(f"Fair Away: {fair_away:.3f} → Cuota: {prob_to_decimal_odds(fair_away):.2f}")
```

### 3.4 Paso 4: Añadir Vig (Juice)

```python
def add_vig(fair_prob, vig_percentage=0.05):
    """
    Añade vig a una probabilidad "fair".
    
    Args:
        fair_prob: Probabilidad sin vig (0.0 - 1.0)
        vig_percentage: Porcentaje de vig a añadir (default 5%)
    
    Returns:
        tuple: (adj_prob, decimal_odds_con_vig)
    """
    # Ajustar probabilidad con vig
    adj_prob = fair_prob * (1 + vig_percentage)
    
    # Calcular cuota decimal
    decimal_odds = 1 / adj_prob
    
    return adj_prob, decimal_odds

# Ejemplo: Añadir 5% de vig
fair_home = 0.50
adj_prob, decimal_odds = add_vig(fair_home, vig_percentage=0.05)

print(f"Prob ajustada: {adj_prob:.3f}")
print(f"Cuota decimal: {decimal_odds:.2f}")
print(f"Implied probability: {1/decimal_odds:.3f}")  # 0.525 (5% más)
```

---

## 4. Generación de Cuotas para Diferentes Mercados

NBA AI solo predice **scores**, no mercados. Aquí te explico cómo convertir scores a mercados:

### 4.1 Moneyline

```python
def score_to_moneyline(home_score, away_score):
    """
    Convierte score predicho a moneyline.
    Basado en la diferencia de score.
    """
    import numpy as np
    
    # Calcular probabilidad desde score
    prob_home = score_to_probability(home_score, away_score)
    
    # Convertir a American Odds
    if prob_home >= 0.5:
        # Favorite
        ml = -(prob_home / (1 - prob_home)) * 100
    else:
        # Underdog
        ml = ((1 - prob_home) / prob_home) * 100
    
    return round(ml)

# Ejemplo
home_score = 115.5
away_score = 112.3
ml = score_to_moneyline(home_score, away_score)
print(f"Home ML: {ml}")  # ej: -135
```

### 4.2 Spread (Point Spread)

```python
def score_to_spread(home_score, away_score, home_favorite=True):
    """
    Estima el spread desde scores predichos.
    Spread = Score favorito - Score underdog
    """
    score_diff = home_score - away_score
    
    # Si home es favorito, spread negativo
    if home_favorite:
        spread = -abs(score_diff)
    else:
        spread = abs(score_diff)
    
    return round(spread, 1)

# Ejemplo
spread = score_to_spread(115.5, 112.3, home_favorite=True)
print(f"Spread: {spread}")  # ej: -3.2
```

### 4.3 Over/Under (Total)

```python
def score_to_total(home_score, away_score):
    """
    Estima el total desde scores predichos.
    Total = Home Score + Away Score
    """
    return round(home_score + away_score, 1)

# Ejemplo
total = score_to_total(115.5, 112.3)
print(f"Total: {total}")  # ej: 227.8
```

---

## 5. Estructura de Datos de Líneas en NBA AI

La tabla `Betting` ya tiene las líneas de mercado:

```sql
-- Tabla Betting en NBA AI
CREATE TABLE Betting (
    game_id TEXT PRIMARY KEY,
    
    -- Opening (ESPN)
    espn_opening_spread REAL,
    espn_opening_total REAL,
    espn_opening_home_moneyline INTEGER,
    espn_opening_away_moneyline INTEGER,
    
    -- Current (pre-game)
    espn_current_spread REAL,
    espn_current_total REAL,
    espn_current_home_moneyline INTEGER,
    
    -- Closing
    espn_closing_spread REAL,
    espn_closing_total REAL,
    
    -- Covers (backup)
    covers_closing_spread REAL,
    covers_closing_total REAL,
    
    -- Resultados (post-game)
    spread_result TEXT,  -- 'W', 'L', 'P'
    ou_result TEXT       -- 'O', 'U', 'P'
);
```

**Puedes usar estos datos para:**
- Comparar tus cuotas con el mercado
- Calibrar tus modelos contra líneas reales
- Backtest de estrategias

---

## 6. Roadmap: De NBA AI a tu Oddsmaker

### Fase 1: Usar como está (Mes 1)
- [x] Obtener predicciones de score
- [x] Usar líneas de mercado existentes
- [x] Comparar predicciones vs líneas

### Fase 2: Cuotas Base (Mes 2)
- [ ] Implementar `remove_vig()` 
- [ ] Calcular "fair odds" desde probabilidades del modelo
- [ ] Calibrar contra resultados históricos

### Fase 3: Tus Propios Modelos (Mes 3+)
- [ ] Entrenar modelos específicos para mercados (no solo scores)
- [ ] Features adicionales: lesiones, lineup, travel, etc.
- [ ] Implementar `add_vig()` configurable por mercado

### Fase 4: Mercados Avanzados (Mes 4+)
- [ ] Props (player points, rebounds, assists)
- [ ] Futures (championship, playoffs)
- [ ] Live/in-game betting
- [ ] Parlays y teasers

---

## 7. Limitaciones Actuales del Modelo

| Métrica | Valor | Implicación |
|---------|-------|-------------|
| **MAE** | ~10 puntos/equipo | Error significativo en scores |
| **Solo NBA** | - | No sirve para otros deportes |
| **No considera lesiones** | - | Impacto no modelado |
| **No considera lineup** | - | Starts/bench no incluido |
| **Predice scores, no mercados** | - | Requiere conversión manual |

---

## 8. Código Completo: Ejemplo de Uso

```python
"""
Ejemplo: Generar cuotas completas para un partido
"""
import numpy as np
from src.predictions.prediction_engines.tree_predictor import TreePredictor

class OddsMaker:
    def __init__(self, model_path, vig=0.05):
        self.predictor = TreePredictor(model_paths=[model_path])
        self.vig = vig
    
    def get_prediction(self, game_id):
        preds = self.predictor.make_pre_game_predictions([game_id])
        return preds[game_id]
    
    def score_to_prob(self, home_score, away_score):
        base_a, base_b = -0.2504, 0.1949
        score_diff = home_score - away_score
        return 1 / (1 + np.exp(-(base_a + base_b * score_diff)))
    
    def generate_odds(self, game_id):
        pred = self.get_prediction(game_id)
        home_score = pred["pred_home_score"]
        away_score = pred["pred_away_score"]
        
        # Probabilidades "crudas" del modelo
        prob_home = self.score_to_prob(home_score, away_score)
        prob_away = 1 - prob_home
        
        # Remover vig
        total = prob_home + prob_away
        fair_home = prob_home / total
        fair_away = prob_away / total
        
        # Añadir vig
        home_with_vig = fair_home * (1 + self.vig)
        away_with_vig = fair_away * (1 + self.vig)
        
        return {
            "game_id": game_id,
            "pred_home_score": round(home_score, 1),
            "pred_away_score": round(away_score, 1),
            
            # Fair odds (sin vig)
            "fair_home_prob": round(fair_home, 3),
            "fair_away_prob": round(fair_away, 3),
            "fair_home_decimal": round(1/fair_home, 2),
            "fair_away_decimal": round(1/fair_away, 2),
            
            # Con vig
            "home_decimal": round(1/home_with_vig, 2),
            "away_decimal": round(1/away_with_vig, 2),
            
            # Derived markets
            "total": round(home_score + away_score, 1),
            "spread": round(home_score - away_score, 1),
            
            # Moneyline (American)
            "home_ml": self._to_moneyline(home_with_vig),
            "away_ml": self._to_moneyline(away_with_vig),
        }
    
    def _to_moneyline(self, prob):
        if prob >= 0.5:
            return round(-(prob / (1 - prob)) * 100)
        else:
            return round(((1 - prob) / prob) * 100)

# Uso
oddsmaker = OddsMaker("models/xgboost_v0.4_mae10.1.joblib", vig=0.05)
odds = oddsmaker.generate_odds("0022500981")

print("=" * 50)
print("ODDSMAKER - Cuotas Generadas")
print("=" * 50)
print(f"Partido: {odds['pred_home_score']} vs {odds['pred_away_score']}")
print(f"Total (O/U): {odds['total']}")
print(f"Spread: {odds['spread']}")
print("-" * 50)
print(f"Home ML: {odds['home_ml']} (@ {odds['home_decimal']})")
print(f"Away ML: {odds['away_ml']} (@ {odds['away_decimal']})")
print("-" * 50)
print(f"Fair Home: {odds['fair_home_decimal']}")
print(f"Fair Away: {odds['fair_away_decimal']}")
```

---

## 9. Conclusión

| Pregunta | Respuesta |
|----------|-----------|
| ¿NBA AI sirve para oddsmakería? | **SÍ**, como base |
| ¿Puedo generar cuotas sin vig? | **SÍ**, con `remove_vig()` |
| ¿Puedo añadir mi propio vig? | **SÍ**, con `add_vig()` |
| ¿Tiene modelos para mercados? | **NO**, solo predice scores |
| ¿Puedo comparar con el mercado? | **SÍ**, tiene datos de ESPN/Covers |

**Recomendación**: Usa NBA AI como tu "motor de predicción de scores" y construye tu propia capa de "oddsmaking" encima.
