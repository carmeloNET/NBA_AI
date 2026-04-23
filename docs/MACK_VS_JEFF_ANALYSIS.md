# Comparación: NBA AI (Jeff) vs Andrew Mack (Oddsmaker)

## La Pregunta Fundamental

> "¿Machine Learning es para crear un modelo y para oddsmaker son fórmulas? ¿Andrew Mack usa ML también?"

**Respuesta corta**: Andrew Mack usa **estadística avanzada/fórmulas**, NO machine learning tradicional. Son enfoques **complementarios**, no competidores.

---

## Comparación de Enfoques

| Aspecto | NBA AI (Jeff) | Andrew Mack (Oddsmaker) |
|---------|---------------|------------------------|
| **Enfoque** | Machine Learning | Estadística/Fórmulas |
| **Predice** | Scores directamente | Proceso de generación |
| **Método** | XGBoost, Ridge, MLP | Poisson, Skellam, Bayes |
| **Features** | 43 variables genéricas | Four Factors + Pace + ORtg |
| **Input** | Features pre-calculados | Posesiones, eficiencia |
| **Output** | Score predicho | Probabilidades → Cuotas |
| **Calibración** | MAE del modelo | Brier Score |
| **Live** | Blend con score actual | Bayes Update Loop |

---

## Enfoque de Andrew Mack (Detalle)

### El proceso de generación del partido

Andrew Mack no predice puntos directamente. Modela el **proceso** que genera los puntos:

```
┌─────────────────────────────────────────────────────────────────┐
│              PROCESO DE GENERACIÓN (MACK)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CALCULAR PACE (Ritmo)                                      │
│     ↓                                                           │
│     Pace = Posesiones por 48 minutos                           │
│     (del equipo local y visitante)                             │
│                                                                 │
│  2. CALCULAR EFICIENCIAS                                        │
│     ↓                                                           │
│     ORtg = Puntos por 100 posesiones (Ofensivo)                │
│     DRtg = Puntos permitidos por 100 posesiones (Defensivo)    │
│                                                                 │
│  3. EXPECTED SCORE (FÓRMULA BASE)                              │
│     ↓                                                           │
│     Expected_Score = Pace × (ORtg × DRtg_oponente) / 100      │
│                                                                 │
│  4. DISTRIBUCIÓN ESTADÍSTICA                                    │
│     ↓                                                           │
│     Poisson → Over/Under                                        │
│     Skellam → Spread                                           │
│     Logit → Moneyline                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Las Fórmulas Clave

#### 1. Pace (Ritmo del partido)
```python
# Posesiones estimadas = (FGA - OREB + TOV + 0.44 * FTA)
# Pace = Posesiones / MinutosJugados * 48
pace_team = (fga - oreb + tov + 0.44 * fta) / minutes * 48
```

#### 2. Expected Score (Puntos esperados)
```python
# Fórmula de Dean Oliver
expected_score = expected_pace * (ortg_team * drtg_opponent) / league_avg_ortg / 100

# Simplificado:
# Expected Score A = (Pace_A + Pace_B) / 2 × (ORtg_A × DRtg_B) / 100
```

#### 3. Four Factors (Dean Oliver)

| Factor | Fórmula | Importancia |
|--------|---------|-------------|
| **eFG%** | (FG + 0.5×FG3) / FGA | Eficiencia de tiro |
| **TOV%** | TOV / (FGA + 0.44×FTA + TOV) | Cuidado del balón |
| **ORB%** | ORB / (ORB + DRB_opp) | Segundas oportunidades |
| **FT Rate** | FTA / FGA | Capacidad de llegar a línea |

### Distribuciones Estadísticas

#### Poisson (para Over/Under)
```python
from scipy.stats import poisson

# Lambda = Expected Score
lambda_home = 112.5
lambda_away = 108.2

# Probabilidad de Over 220
prob_over = 1 - poisson.cdf(220, lambda_home + lambda_away)
```

#### Skellam (para Spread)
```python
from scipy.stats import skellam

# Diferencia entre dos Poisson
prob_home_win = skellam.cdf(0, lambda_home, lambda_away)
prob_spread_cover = skellam.cdf(-3, lambda_home, lambda_away)  # Home -3
```

#### Distribución Normal (aproximación)
```python
from scipy.stats import norm

# Usando la diferencia de scores
score_diff = home_score - away_score
std_dev = 10.5  # Varianza típica en NBA

prob_home_win = norm.cdf(score_diff / std_dev)
```

---

## Los Dos Enfoques: ML vs Fórmulas

### ¿Por qué Andrew Mack NO usa ML?

1. **Interpretabilidad**: Las fórmulas te dicen EXACTAMENTE por qué un equipo es favorito
2. **Datos limitados**: Con ~82 partidos por equipo, ML puede sobreajustar
3. **Causalidad**: Las fórmulas capturan la física del juego (posesiones → puntos)
4. **Mercados**: Puedes derivar TODOS los mercados desde una base esperada

### ¿Cuándo usar ML?

| Escenario | Enfoque recomendado |
|-----------|---------------------|
| Pocos datos (< 1 temporada) | Fórmulas (Poisson) |
| Mucho datos (> 3 temporadas) | ML puede superar |
| Necesitas解释了 | Fórmulas |
| Buscar patrones complejos | ML |
| Mercados específicos | Fórmulas + ML complementario |

---

## Propuesta de Integración en NBA AI

### Opción 1: Enfoque Híbrido (Recomendado)

Combina lo mejor de ambos mundos:

```
┌─────────────────────────────────────────────────────────────────┐
│              NBA AI + MACK HYBRID                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CAPA 1: DATOS                                                 │
│  ├── NBA API → PBP, Boxscores                                  │
│  └── Calcular: Pace, ORtg, DRtg, Four Factors                 │
│                                                                 │
│  CAPA 2: PREDICCIÓN BASE                                       │
│  ├── Método A: Fórmula (Mack)                                  │
│  │    Expected_Score = f(Pace, ORtg, DRtg)                    │
│  │                                                               │
│  └── Método B: ML (Jeff)                                       │
│       XGBoost → pred_home_score, pred_away_score               │
│                                                                 │
│  CAPA 3: ENSAMBLE                                              │
│  └── blend(fórmula, ML) = predicción_final                     │
│                                                                 │
│  CAPA 4: MERCADOS                                              │
│  ├── Poisson → Over/Under                                       │
│  ├── Skellam → Spread                                           │
│  └── Logit → Moneyline                                         │
│                                                                 │
│  CAPA 5: ODDS                                                  │
│  ├── Fair Odds (sin vig)                                       │
│  └── + Vig (configurable)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Opción 2: Reemplazar ML con Fórmulas

Simplemente reemplazar el modelo XGBoost con fórmulas:

```python
def predict_with_mack_formula(home_team, away_team, db_path):
    """
    Predicción usando fórmulas de Andrew Mack.
    """
    # 1. Obtener stats de los equipos
    home_stats = get_team_advanced_stats(home_team, db_path)
    away_stats = get_team_advanced_stats(away_team, db_path)
    
    # 2. Calcular Pace promedio
    expected_pace = (home_stats['pace'] + away_stats['pace']) / 2
    
    # 3. Calcular Expected Scores
    league_avg_ortg = 112.0  # NBA 2024-25
    
    home_expected = expected_pace * (home_stats['ortg'] * away_stats['drtg']) / 100
    away_expected = expected_pace * (away_stats['ortg'] * home_stats['drtg']) / 100
    
    return {
        'pred_home_score': home_expected,
        'pred_away_score': away_expected
    }
```

---

## Implementación en NBA AI

### 1. Añadir Four Factors como Features

Los 43 features actuales de NBA AI son buenos, pero añade estos de Mack:

```python
FOUR_FACTORS_FEATURES = [
    'home_efg_pct',      # eFG% (eficiencia de tiro)
    'home_tov_pct',      # TOV% (pérdidas)
    'home_orb_pct',      # ORB% (rebotes ofensivos)
    'home_ft_rate',      # FT / FGA
    'away_efg_pct',
    'away_tov_pct',
    'away_orb_pct',
    'away_ft_rate',
    'home_pace',         # Ritmo
    'away_pace',
    'home_ortg',         # Offensive Rating
    'home_drtg',         # Defensive Rating
    'away_ortg',
    'away_drtg',
]
```

### 2. Nueva Función: Expected Score Calculator

```python
"""
Mack-inspired expected score calculator.
"""

def calculate_expected_score(home_team_stats, away_team_stats, league_avg_ortg=112.0):
    """
    Calcula el score esperado usando la fórmula de Dean Oliver.
    
    Args:
        home_team_stats: dict con 'pace', 'ortg', 'drtg'
        away_team_stats: dict con 'pace', 'ortg', 'drtg'
        league_avg_ortg: promedio de la liga (default 112.0)
    
    Returns:
        tuple: (home_expected, away_expected)
    """
    # Pace promedio del partido
    expected_pace = (home_team_stats['pace'] + away_team_stats['pace']) / 2
    
    # Expected Score usando la fórmula
    # Score = Pace × (ORtg × DRtg_opponent) / (LeagueAvg × 100)
    home_expected = expected_pace * (
        home_team_stats['ortg'] * away_team_stats['drtg']
    ) / league_avg_ortg / 100
    
    away_expected = expected_pace * (
        away_team_stats['ortg'] * home_team_stats['drtg']
    ) / league_avg_ortg / 100
    
    return home_expected, away_expected


def calculate_four_factors(stats_df, team_id):
    """
    Calcula los Four Factors desde boxscore stats.
    """
    team = stats_df[stats_df['team_id'] == team_id].iloc[0]
    
    efg_pct = (team['fgm'] + 0.5 * team['fg3m']) / team['fga'] if team['fga'] > 0 else 0
    
    poss = team['fga'] - team['oreb'] + team['tov'] + 0.44 * team['fta']
    tov_pct = team['tov'] / poss if poss > 0 else 0
    
    orb_pct = team['oreb'] / (team['oreb'] + team['dreb_opp']) if (team['oreb'] + team['dreb_opp']) > 0 else 0
    
    ft_rate = team['fta'] / team['fga'] if team['fga'] > 0 else 0
    
    return {
        'efg_pct': efg_pct,
        'tov_pct': tov_pct,
        'orb_pct': orb_pct,
        'ft_rate': ft_rate,
        'pace': team.get('pace', 100),
        'ortg': team.get('ortg', 110),
        'drtg': team.get('drtg', 110),
    }
```

### 3. Generación de Cuotas desde Expected Score

```python
from scipy.stats import poisson, skellam
import numpy as np

def expected_score_to_odds(home_score, away_score, vig=0.05):
    """
    Convierte expected scores a cuotas (fair odds).
    
    Args:
        home_score: Expected score del local
        away_score: Expected score del visitante
        vig: Porcentaje de vig (default 5%)
    
    Returns:
        dict con cuotas para todos los mercados
    """
    # Lambda para Poisson
    lambda_home = home_score
    lambda_away = away_score
    lambda_total = home_score + away_score
    
    # === MONEYLINE ===
    # Usando Skellam (diferencia de Poisson)
    prob_home_win = 1 - skellam.cdf(0, lambda_home, lambda_away)
    prob_away_win = skellam.cdf(0, lambda_home, lambda_away)
    
    # Remover vig
    total_prob = prob_home_win + prob_away_win
    fair_home_prob = prob_home_win / total_prob
    fair_away_prob = prob_away_win / total_prob
    
    # Añadir vig
    home_with_vig = fair_home_prob * (1 + vig)
    away_with_vig = fair_away_prob * (1 + vig)
    
    # Cuotas decimales
    home_decimal = 1 / home_with_vig
    away_decimal = 1 / away_with_vig
    
    # American odds
    home_ml = to_american(home_decimal)
    away_ml = to_american(away_decimal)
    
    # === OVER/UNDER ===
    # Mediana de Poisson = floor(lambda) o lambda
    total_median = lambda_total
    
    # Probabilidad de over specific total
    over_line = round(lambda_total)
    prob_over = 1 - poisson.cdf(over_line - 1, lambda_total)
    prob_under = poisson.cdf(over_line - 1, lambda_total)
    
    # === SPREAD ===
    # Diferencia esperada
    score_diff = home_score - away_score
    
    # Probabilidad de cubrir -X
    prob_cover = 1 - skellam.cdf(-score_diff, lambda_home, lambda_away)
    
    return {
        'moneyline': {
            'home': home_decimal,
            'away': away_decimal,
            'home_american': home_ml,
            'away_american': away_ml,
        },
        'total': {
            'line': over_line,
            'over': 1 / (prob_over * (1 + vig)),
            'under': 1 / (prob_under * (1 + vig)),
        },
        'spread': {
            'line': round(score_diff, 1),
            'home_prob': prob_cover,
        }
    }


def to_american(decimal_odds):
    """Convierte cuotas decimales a americanas."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))
```

---

## Resumen: ¿Cuál usar?

| Tu objetivo | Enfoque |
|-------------|---------|
| Explicar por qué un equipo gana | Fórmulas (Mack) |
| Maximizar precisión | ML (Jeff) + Fórmulas |
| Entender mercados | Fórmulas |
| Apuestas automatizadas | Híbrido |
| Pocos datos (temporada nueva) | Fórmulas |
| Mucha historia | ML puede ayudar |

**Mi recomendación**: Usa el enfoque de Mack como **base** y el ML de Jeff como **ajuste/refinamiento**. Los features de Mack (Pace, ORtg, Four Factors) son más interpretables y fundamentados en la física del juego.

---

## Referencias en tu Proyecto

- **Documentación Mack**: `C:\Users\AI_Agent\Documents\Analytics\docs\oddsmaker-model\`
- **Datos**: `C:\Users\AI_Agent\Documents\Analytics\bin\Debug\net10.0\data\nba\season-2025-26\`
- **Four Factors**: Dean Oliver's "Basketball on Paper"
- **Poisson/Skellam**: scipy.stats
