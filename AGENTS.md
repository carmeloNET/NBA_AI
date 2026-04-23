# NBA AI - AGENTS.md

## Stack Tecnológico

### Lenguaje y Entorno
- **Python** 3.10+
- **Virtual Environment**: venv (creado por setup.py)

### Core
- **Flask** 3.1.2 - Web framework
- **SQLAlchemy** 2.0.44 - ORM
- **SQLite** - Base de datos
- **PyYAML** 6.0.3 - Configuración
- **python-dotenv** 1.2.1 - Variables de entorno
- **requests** 2.32.5 - HTTP client

### Datos y Procesamiento
- **pandas** 2.3.3 - Manipulación de datos
- **numpy** 2.3.5 - Computación numérica
- **pytz** 2025.2 - Timezones
- **tqdm** 4.67.1 - Progress bars
- **tzlocal** 5.3.1 - Timezone local

### Web Scraping
- **beautifulsoup4** 4.13.3 - HTML parsing
- **pdfplumber** 0.11.9 - PDF parsing (injury reports)

### Machine Learning
- **scikit-learn** 1.7.2 - ML algorithms (Ridge regression)
- **scipy** 1.16.3 - Scientific computing
- **xgboost** 3.1.2 - Gradient boosting
- **joblib** 1.5.2 - Model serialization
- **torch** 2.8.0+cpu (optional) - MLP predictor

### NBA Data
- **nba_api** 1.11.3 - NBA Stats API wrapper

### Testing
- **pytest** 9.0.1 - Test framework
- **pytest-cov** 7.0.0 - Code coverage

### Visualización
- **matplotlib** 3.9.1 - Gráficos
- **seaborn** 0.13.2 - Gráficos estadísticos

### Logging
- **python-json-logger** 4.0.0 - JSON logging

---

## Estructura del Proyecto

```
NBA_AI/
├── src/
│   ├── database_updater/     # ETL pipeline
│   │   ├── schedule.py       # Fetch schedules
│   │   ├── players.py       # Player reference data
│   │   ├── pbp.py           # Play-by-play data
│   │   ├── game_states.py   # Parse PBP → game states
│   │   ├── boxscores.py     # Box score stats
│   │   ├── betting.py       # Betting lines (ESPN, Covers)
│   │   ├── covers.py        # Covers.com scraping
│   │   └── database_update_manager.py  # Orchestrator
│   ├── predictions/
│   │   ├── prediction_engines/
│   │   │   ├── base_predictor.py    # Abstract base class
│   │   │   ├── baseline_predictor.py # PPG-based baseline
│   │   │   ├── linear_predictor.py   # Ridge regression
│   │   │   ├── tree_predictor.py     # XGBoost
│   │   │   ├── mlp_predictor.py      # PyTorch MLP
│   │   │   └── ensemble_predictor.py # Weighted ensemble
│   │   ├── features.py       # Feature engineering
│   │   └── prediction_manager.py
│   ├── model_training/
│   │   ├── train.py          # Model training
│   │   ├── models.py         # Model definitions
│   │   └── evaluation.py     # Metrics
│   ├── games_api/
│   │   ├── games.py          # Game data fetching
│   │   └── api.py            # REST endpoints
│   └── web_app/
│       ├── app.py            # Flask app
│       └── game_data_processor.py
├── tests/                    # pytest test suite
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
└── start_app.py              # Entry point
```

---

## Pipeline de Datos

```
Schedule → Players → Injuries → Betting → PBP → GameStates → Boxscores → Features → Predictions
```

### Flujo Completo:
1. **Schedule**: Obtiene calendario de partidos desde NBA API
2. **Players**: Actualiza referencia de jugadores
3. **Injuries**: Scraping de informes de lesiones (PDFs)
4. **Betting**: Líneas de apuestas (ESPN API + Covers.com)
5. **PBP**: Play-by-play de cada partido
6. **GameStates**: Parsea PBP en estados discretos
7. **Boxscores**: Estadísticas tradicionales (PlayerBox, TeamBox)
8. **Features**: Genera features desde estados previos
9. **Predictions**: Ejecuta modelos predictivos

---

## Modelos de Predicción

| Modelo | Algoritmo | MAE | Notes |
|--------|-----------|-----|-------|
| Baseline | PPG formula | - | Simple baseline |
| Linear | Ridge Regression | 11.2 | 34 rolling avg features |
| Tree | XGBoost | 10.1 | Default, mejor rendimiento |
| MLP | PyTorch | 11.1 | Requiere PyTorch |
| Ensemble | Weighted avg | - | 30% Linear + 40% Tree + 30% MLP |

---

## Comandos Útiles

```bash
# Instalación
python setup.py

# Ejecutar web app
python start_app.py

# Usar predictor específico
python start_app.py --predictor=Tree

# Modo debug
python start_app.py --debug --log_level=DEBUG

# Testing
pytest tests/ -v
pytest --cov=src tests/
```

---

## Notas para Desarrollo

- NBA API tiene rate limits - manejar con backoff
- Los modelos se almacenan en `models/` (descargados de Releases)
- Base de datos SQLite en `data/`
- Season actual por defecto: 2025-2026
- GenAI/Deep Learning está en desarrollo (no usa OpenAI actualmente)
