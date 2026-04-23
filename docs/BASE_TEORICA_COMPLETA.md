# Base Teórica Completa - Proyecto Oddsmaker

## 1. Libros de Referencia

### A. Conquering Risk (Elihu Feustel)
**Enfoque**: betting tradicional y gestión de riesgo

**Conceptos clave**:
- **Expected Value (EV)**: `EV = (prob_win × ganancia) - (prob_perdida × pérdida)`
- **Overround/Vig**: Margen de la casa = suma_probabilidades - 1
- **Fair Odds**: `1 / probabilidad`
- **Criterio de Kelly**: `f* = (bp - q) / b`
- **Half-Point Conversions**: Optimización de líneas
- **Teasers/Pleasers**: Apuestas derivadas del spread
- **Arbitrage/Middles**: Estrategias de cobertura
- **Derivatives**: Líneas derivadas de spread/total base

### B. Bayesian Sports Models (Andrew Mack)
**Enfoque**: Inferencia bayesiana para deportes

**Conceptos clave**:
- **Teorema de Bayes**: `P(B|A) = P(A|B) × P(B) / P(A)`
- **Prior**: Creencia inicial antes de ver evidencia
- **Likelihood**: Probabilidad de la evidencia dada la hipótesis
- **Posterior**: Creencia actualizada
- **Distribuciones**:
  - Poisson → eventos discretos (puntos, asistencias)
  - Normal/Gaussiana → variables continuas
  - Beta → probabilidades (win%)
  - Gamma/Erlang → tiempo entre eventos (live betting)
  - Binomial → sí/no
  - Negative Binomial → sobredispersión

---

## 2. Framework de Implementación (Fases)

### Fase 1: Arquitectura de Datos y Priors (Mack)
- **Priors**: Distribución inicial de fuerza del equipo
- **Fuentes**: Basketball-Reference Team Misc Stats (3-5 temporadas)
- **Tablas**: `dim_season_priors` con ORtg, DRtg, Pace, Four Factors

### Fase 2: Análisis Descriptivo (Michigan)
- **Rolling Averages**: Ventanas de 10-15 juegos
- **Desviación Estándar**: Consistencia del equipo
- **Z-Scores**: Normalización vs League Average
- **Fórmula**: `Z = (x - μ) / σ`

### Fase 3: Cruce de Contextos (Feustel)
- **SOS (Strength of Schedule)**: Dificultad del calendario
- **Back-to-Back**: Descuento de 1.5-2.5 puntos
- **Home Court Advantage**: ~3 puntos
- **Viajes y altitud**: Ajustes contextuales

### Fase 4: Motor de Probabilidad
| Mercado | Función | Referencia |
|--------|---------|-----------|
| Moneyline | Logística / Gaussiana | Feustel |
| Spread | Normal (Z-Score) | Feustel |
| Totals | Poisson | Mack |
| Player Props | Poisson | Mack |
| Milestones | Binomial | Mack |

### Fase 5: Oddsmaking
- **Fair Value**: `1 / probabilidad`
- **Overround**: Añadir margen (5-10%)
- **Kelly Criterion**: Tamaño de apuesta óptimo

---

## 3. Métricas y Validadores

### Métricas de Predicción
- **Brier Score**: Precisón de probabilidades
- **Z-Score Monitor**: Desviación vs league average
- **Confidence Interval**: Rango de error

### Validación de Datos
- Pace promedio = Team Misc Pace
- ORtg/DRtg debe coincidir con Basketball-Reference
- Suma de puntos = PTS total

---

## 4. Four Factors (Dean Oliver)

| Factor | Fórmula | Importancia |
|--------|---------|-------------|
| **eFG%** | (FG + 0.5×FG3) / FGA | Eficiencia de tiro |
| **TOV%** | TOV / (FGA + 0.44×FTA + TOV) | Cuidado del balón |
| **ORB%** | ORB / (ORB + DRB_opp) | Segundas oportunidades |
| **FT Rate** | FTA / FGA | Capacidad de llegar a línea |

---

## 5. Workflow de Andrew Mack

```
1. PRIORS (3-5 temporadas)
   └─ ORtg, DRtg, Pace, Four Factors

2. LIKELIHOOD (temporada actual)
   └─ Promedio acumulado

3. ROLLING (últimos 10-15 juegos)
   └─ Forma actual

4. EXPECTED SCORE
   └─ Pace × (ORtg × DRtg_opp) / 100

5. DISTRIBUCIONES
   └─ Poisson → O/U
   └─ Skellam → Spread
   └─ Logística → ML

6. LIVE (Bayes Update)
   └─ Posterior → Prior para siguiente play
```

---

## 6. Funciones Matemáticas Clave

### Expected Score (Mack)
```python
Expected_Score = Pace × (ORtg × DRtg_opponent) / League_Avg_ORtg / 100
```

### Poisson (Over/Under)
```python
P(X=k) = λ^k × e^(-λ) / k!
```

### Z-Score
```python
Z = (x - μ) / σ
```

### Fair Odds
```python
fair_decimal = 1 / probability
```

### Kelly Criterion
```python
f* = (b × p - q) / b
# donde q = 1 - p
```

---

## 7. Datos Necesarios

### Inputs del Modelo
- Posesiones (Poss)
- Puntos (PTS)
- FG, FGA, FG3, FG3A
- FTA, FTM
- OREB, DREB
- TOV
- Minutos jugados

### Calculados
- Pace = Poss / Min × 48
- ORtg = PTS / Poss × 100
- DRtg = PTS_opp / Poss × 100
- eFG% = (FG + 0.5×FG3) / FGA

---

## 8. Comparación: ML vs Fórmulas

| Aspecto | ML (Jeff) | Fórmulas (Mack/Feustel) |
|---------|-----------|-------------------------|
| **Predice** | Scores directamente | Proceso físico |
| **Interpretabilidad** | Baja | Alta |
| **Datos necesarios** | Muchos | Pocos |
| **Sobreajuste** | Sí | No |
| **Mercados derivados** | No | Sí |

---

## 9. Arquitectura Recomendada (Híbrido)

```
┌─────────────────────────────────────────────────────────┐
│                   CAPA 1: DATOS                        │
│  NBA API → Boxscores → Pace, ORtg, DRtg, Four F.   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              CAPA 2: ANÁLISIS ACTUAL                  │
│  Rolling 10-15 juegos, Z-Scores vs League Average     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              CAPA 3: AJUSTES CONTEXTUALES              │
│  Home advantage, B2B, Travel, SOS                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              CAPA 4: EXPECTED SCORE                    │
│  Pace × (ORtg × DRtg_opp) / 100                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              CAPA 5: PROBABILIDADES                    │
│  Poisson (O/U), Skellam (Spread), Logit (ML)         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              CAPA 6: CUOTAS                            │
│  Fair Odds = 1/P → + Vig → Comparar con mercado      │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Referencias

- **Dean Oliver**: "Basketball on Paper" - Four Factors
- **Andrew Mack**: Bayesian Sports Models - Inferencia bayesiana
- **Elihu Feustel**: "Conquering Risk" - Betting math
- **Basketball-Reference**: Datos y validación

---

## 11. Resumen de tu Base Teórica

| Componente | Fuente | Estado |
|------------|--------|--------|
| Priors (3-5 temporadas) | Mack | Por implementar |
| Rolling averages | Michigan | Por implementar |
| Four Factors | Oliver | Por implementar |
| Expected Score | Mack | Por implementar |
| Poisson O/U | Mack | Por implementar |
| Skellam Spread | Mack | Por implementar |
| Bayes Live | Mack | Por implementar |
| Kelly | Feustel | Por implementar |
| Vig/Overround | Feustel | Por implementar |

---

## 12. Próximo Paso

**Recomendación**: Implementar el enfoque de Mack (fórmulas estadísticas) como base, ya que:
1. Requiere menos datos para funcionar
2. Es más interpretable
3. Permite derivar todos los mercados desde una base
4. La industria lo usa como estándar

ML puede añadirse después como refinamiento opcional.
