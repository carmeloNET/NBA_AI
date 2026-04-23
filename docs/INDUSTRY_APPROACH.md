# Enfoque de la Industria: ¿Estadística, ML o Híbrido?

## Respuesta de la Industria

**La industria NO usa solo ML ni solo fórmulas**. Usa un **enfoque híbrido**:

| Tier | Enfoque | Uso en Industria |
|------|---------|-----------------|
| **1. Base** | Fórmulas estadísticas | Cuotas iniciales (pinacle, bet365) |
| **2. Ajuste** | Modelos estadísticos | Calibración de probabilidades |
| **3. Mejora** | ML (opcional) | Detectar patrones complejos |

---

## Por qué la Industria Usa Fórmulas como Base

### 1. **Interpretabilidad**
Un oddsmaker necesita explicar por qué una línea es así. Con fórmulas puedes decir:
> "El Lakers tiene ORtg 118 y el Celtics DRtg 110, con Pace 100 → Expected Score = 118×110/100 = 129.8"

Con ML solo dirías: "El modelo dice 129.8".

### 2. **Datos Limitados**
- Solo 82 partidos por equipo por temporada
- ML sobreajena con pocos datos
- Las fórmulas no sobreajenan porque tienen estructura fija

### 3. **Estabilidad**
Las fórmulas son estables año tras año. ML necesita reentrenamiento constante.

### 4. **Mercados Derivados**
Desde Expected Score puedes derivar:
- Moneyline (logística)
- Spread (Skellam)
- Over/Under (Poisson)
- Player Props (Poisson)

---

## Cuándo Usar ML (y por qué la mayoría no lo usa)

| Mito | Realidad |
|------|----------|
| "ML es más preciso" | Con 82 datos/equipo, ML supera a fórmulas solo marginalmente |
| "ML encuentra patrones complejos" | El mercado ya pricea esos patrones |
| "ML es el futuro" | Los mejores oddsmakers (Pinnacle) usan 80% estadísticas + 20% ML |

### Cuándo ML ayuda:
- **Mercados de jugadores** (props): Más datos por jugador
- **Live betting**: Patrones temporales
- **Features no lineales**: Interacciones complejas

---

## Arquitectura Recomendada (Industria Standard)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MOTOR DE ODDSMAKING (INDUSTRIA)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CAPA 1: DATOS BASE (FASE MACK)                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  PRIORS (3-5 temporadas)                                     │   │
│  │  ├── ORtg, DRtg, Pace                                       │   │
│  │  ├── Four Factors (eFG%, TOV%, ORB%, FT Rate)               │   │
│  │  └── Z-Scores vs League Average                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CAPA 2: ANÁLISIS ACTUAL (FASE MICHIGAN)                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ROLLING AVERAGES (últimos 10-15 juegos)                    │   │
│  │  ├── Promedio móvil                                         │   │
│  │  ├── Desviación estándar (varianza)                          │   │
│  │  └── Desviación vs Priors                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CAPA 3: AJUSTES CONTEXTUALES (FASE FEUSTEL)                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FACTORES EXTERNOS                                          │   │
│  │  ├── Home Court Advantage (~3 puntos)                       │   │
│  │  ├── Back-to-Back (-1.5 a -2.5 puntos)                     │   │
│  │  ├── Travel (viajes largos)                                 │   │
│  │  └── SOS (Strength of Schedule)                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CAPA 4: EXPECTED SCORE (FÓRMULA CORE)                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Expected_Score = Pace × (ORtg × DRtg_opp) / 100          │   │
│  │                                                               │   │
│  │  + Ajustes de Capas 2 y 3                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CAPA 5: PROBABILIDADES (DISTRIBUCIONES)                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MONEYLINE  → Distribución Logística / Normal              │   │
│  │  SPREAD     → Distribución Skellam                         │   │
│  │  OVER/UNDER → Distribución Poisson                         │   │
│  │  PROPS      → Distribución Poisson / Binomial              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CAPA 6: ML (OPCIONAL - MEJORA MARGINAL)                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  XGBoost/LightGBM                                          │   │
│  │  └── Input: Residual del modelo de fórmulas                │   │
│  │  └── Output: Ajuste de ±1-2% en probabilidades           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CAPA 7: CUOTAS FINALES                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FAIR ODDS = 1 / Probabilidad                              │   │
│  │  + OVERROUND (5-10%) → Cuotas finales                      │   │
│  │  → Comparar con mercado (Bet365/Pinnacle)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Resumen: Qué Usar y Cuándo

| Tu Objetivo | Enfoque | % Importancia |
|-------------|---------|---------------|
| **Crear líneas desde cero** | Fórmulas (Mack) | 70% |
| **Calibrar probabilidades** | Distribuciones (Poisson/Skellam) | 20% |
| **Ajustes finos** | ML (opcional) | 10% |
| **Validar contra mercado** | Comparar líneas | - |

---

## Mi Recomendación para NBA AI

**Implementá en este orden**:

1. **Fase 1**: Fórmulas de Mack (Expected Score)
   - Calcular Pace, ORtg, DRtg desde boxscores
   - Expected Score = Pace × (ORtg × DRtg_opp) / 100

2. **Fase 2**: Distribuciones
   - Poisson → Over/Under
   - Skellam → Spread
   - Logística → Moneyline

3. **Fase 3** (opcional): ML como refinamiento
   - Usar residuals del modelo de fórmulas
   - XGBoost para ajustar probabilidades

---

## Comparativa Final

| Enfoque | Ventajas | Desventajas |
|---------|----------|-------------|
| **Solo Fórmulas** | Estable, interpretable, funciona con pocos datos | No captura patrones complejos |
| **Solo ML** | Puede superar en ciertos casos | Sobreajuste, no interpretable, requiere muchos datos |
| **Híbrido (Recomendado)** | Lo mejor de ambos mundos | Más complejo de mantener |

**La mayoría de oddsmakers profesionales usan el enfoque híbrido**, pero con **80-90% peso en fórmulas** y 10-20% en ML para ajustes marginales.

---

## Referencias en tu Proyecto

- Documentación: `C:\Users\AI_Agent\Documents\Analytics\docs\oddsmaker-model\`
- Fase 1 (Mack): Priors, ORtg, DRtg, Four Factors
- Fase 2 (Michigan): Rolling averages, Z-Scores
- Fase 3 (Feustel): Ajustes contextuales (fatiga, home advantage)
- Fase 4: Distribuciones (Poisson, Skellam, Logística)
