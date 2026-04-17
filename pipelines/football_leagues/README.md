# Football Leagues ML Pipeline

Pipeline reutilizable para entrenar y reentrenar modelos XGBoost de predicción de resultados de futbol.

Soporta múltiples ligas:

- **Bundesliga** (Alemania - D1)
- **Premier League** (Inglaterra - E0)
- **La Liga** (España - SP1)
- **Serie A** (Italia - I1)

## Estructura

```
pipelines/football_leagues/
  __init__.py              # Exports principales
  config.py                # Configuracion de ligas
  data.py                  # DataLoader - carga desde football-data.co.uk
  features.py              # FeatureBuilder - construccion de features
  model.py                 # Entrenador, Reentrenador - lógica ML
  scripts/
    train.py               # Entrenar desde cero
    retrain.py             # Reentrenar con datos frescos
    status.py              # Ver status de modelos
  README.md                # Este archivo
```

## Setup Local

### 1. Verificar entorno

```bash
cd /Users/josecontreras/Desktop/areas/personal/code/machine_learning_base
source .venv/bin/activate
python --version  # Python 3.8+
```

### 2. Instalar dependencias (si no están)

Las dependencias están en `requirements.txt`:

```bash
pip install -r requirements.txt
```

Principales:

- `pandas`, `numpy` — datos y vectorizacion
- `xgboost` — modelo
- `scikit-learn` — calibracion, metrics
- `joblib` — serializacion

## Uso

### Entrenar desde cero

```bash
cd pipelines/football_leagues

# Entrenar una liga
python scripts/train.py --liga german

# Entrenar multiples ligas individualmente
python scripts/train.py --liga german --liga premier --liga spanish

# Entrenar todas las ligas
python scripts/train.py --todas

# Ver ligas disponibles
python scripts/train.py --lista
```

**Tiempo:** ~15 min por liga (carga datos + construye features + entrena)

**Salida:**

- `../../real_models/{liga}/modelo_{liga}.pkl` — modelo principal
- `../../real_models/{liga}/versiones/modelo_{YYYY-MM-DD}.pkl` — snapshot con fecha

### Reentrenar con datos frescos

Ejecuta cada ~3 jornadas (~27 partidos nuevos en Bundesliga/Premier):

```bash
# Reentrenar una liga
python scripts/retrain.py --liga german

# Reentrenar todas
python scripts/retrain.py --todas

# Threshold personalizado (default 2%)
python scripts/retrain.py --liga german --threshold 0.01

# Ver ligas disponibles
python scripts/retrain.py --lista
```

**Lógica de decisión:**

- Carga accuracy del modelo anterior
- Entrena con datos frescos
- Compara: `accuracy_nuevo >= accuracy_anterior - umbral`
- Solo reemplaza si pasa el criterio
- Si falla: mantiene modelo anterior e informa

### Ver status

```bash
# Status de todos los modelos
python scripts/status.py

# Status de una liga
python scripts/status.py --liga german

# Con detalles (log loss, features, temporadas)
python scripts/status.py --detalle
```

## Ejemplo Flujo Completo

### Primer uso (setup inicial)

```bash
cd machine_learning_base
source .venv/bin/activate

cd pipelines/football_leagues

# 1. Entrenar Bundesliga
python scripts/train.py --liga german
# ✓ Genera: ../../real_models/german_league/modelo_german.pkl

# 2. Ver status
python scripts/status.py --liga german
# Muestra: accuracy 53.8%, fecha entrenamiento, etc.
```

### Mantenimiento (cada 3 jornadas)

```bash
# 1. Reentrenar
python scripts/retrain.py --liga german

# 2. Ver status actualizado
python scripts/status.py --liga german --detalle
```

### Si quieres entrenar todas las ligas

```bash
# Primera vez — entrenar todas
python scripts/train.py --todas

# O especificando cada una
python scripts/train.py --liga german --liga premier --liga spanish --liga italian

# Reentrenamientos posteriores
python scripts/retrain.py --todas --threshold 0.02
```

## Integracion con CI/CD (GitHub Actions)

Ejemplo `cron_retrain.yml`:

```yaml
name: Reentrenar Modelos

on:
  schedule:
    # Cada domingo a las 2 AM
    - cron: '0 2 * * 0'
  workflow_dispatch:  # Permitir ejecucion manual

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install -r requirements.txt
      
      - name: Retrain all models
        run: |
          cd pipelines/football_leagues
          python scripts/retrain.py --todas --threshold 0.02
      
      - name: Commit updates
        if: success()
        run: |
          git config user.name "ML Bot"
          git config user.email "ml@example.com"
          git add real_models/*/versiones/
          git commit -m "chore: modelos reentrenenados $(date +'%Y-%m-%d')" || echo "No changes"
          git push
```

## Integracion en Notebooks

Los notebooks pueden importar el pipeline:

```python
from pipelines.football_leagues.config import get_config, listar_ligas
from pipelines.football_leagues.model import Entrenador, Reentrenador

# Entrenar German League
config = get_config('german')
entrenador = Entrenador(config)
metricas = entrenador.entrenar()
print(f"Accuracy: {metricas['accuracy']:.1%}")

# Reentrenar
reentrenador = Reentrenador(config)
actualizado = reentrenador.retrain(umbral_degradacion=0.02)
print("Actualizado" if actualizado else "Mantiene anterior")
```

## Estructura Datos

### Entrada (football-data.co.uk)

```
HomeTeam, AwayTeam, Date, FTHG, FTAG, FTR, HS, AS, HST, AST, HC, AC, ...
Bayern, Augsburg, 2026-04-18, 3, 1, H, 18, 7, 8, 2, 12, 4, ...
```

### Salida Modelo (pickle)

```python
payload = {
    'model_v3': CalibratedClassifierCV,  # XGBoost calibrado
    'feature_cols_v3': [52 features],     # Lista nombres features
    'le': LabelEncoder,                   # Encoder H/D/A
    'df': DataFrame,                      # Datos historicos
    'accuracy': 0.538,                    # En test set
    'accuracy_alta_confianza': 0.644,     # Confianza > 55%
    'logloss': 1.0168,
    'n_alta_confianza': 59,
    'fecha_entrenamiento': '2026-04-08',
    'temporadas': ['2021', '2122', '2223', '2324', '2425', '2526'],
    'temporada_test': '2526',
    'partidos_train': 1485,
    'partidos_test': 249,
}
```

## Features Generadas (52 total)

### Team Stats (por equipo)

- `h_recent_gf_pg` — goles por partido (ultimos 5)
- `h_recent_gc_pg` — goles en contra por partido
- `h_recent_win_rate` — tasa de victoria
- `h_season_*` — idem pero temporada completa
- `h_tabla_posicion` — posicion en tabla
- `h_tabla_pts_pg` — puntos por partido
- `h_tabla_dif_goles_szn` — diferencia goles temporada

### Head-to-Head

- `h2h_home_wr` — tasa victoria local en h2h
- `h2h_draw_rate` — tasa empate
- `h2h_away_wr` — tasa victoria visitante

### Diferenciales

- `dif_recent_wr` — diferencia tasa victoria reciente
- `dif_season_wr` — diferencia tasa victoria temporada
- `dif_recent_gf` — diferencia goles por partido
- `home_advantage` — ventaja jugar en casa

## Criterios Decision Reentrenamiento

Compara: `accuracy_nuevo >= accuracy_anterior - umbral_degradacion`

**Default:** `umbral_degradacion = 0.02` (2%)

**Ejemplos:**

- Anterior: 53.8% → Nuevo: 53.5% → ✓ Actualiza (degradacion 0.3%)
- Anterior: 53.8% → Nuevo: 51.6% → ✗ Mantiene (degradacion 2.2%)

## Troubleshooting

### Error: "No hay modelo anterior"

```bash
python scripts/train.py --liga german
# Entrena desde cero, genera modelo inicial
```

### Error: "Insuficientes partidos nuevos"

```bash
python scripts/retrain.py --liga german
# Mensaje: "Menos de 20 partidos nuevos — esperar hasta tener 27+"
# Esperación: Ejecuta solo despues de ~3 jornadas
```

### Accuracy bajo/alta

Revisa:

1. **Datos frescos:** `python scripts/status.py --detalle`
2. **Temporadas:** ¿Incluye datos desde 2020?
3. **Features:** ¿Se construyeron correctamente?

Ejecuta:

```bash
python scripts/retrain.py --liga german --threshold -0.1
# Force=true overrides degradation check
```

## Roadmap

- [ ] Export a ONNX (para usar sin joblib)
- [ ] Predicciones por jornada (batch)
- [ ] Dashboard de metricas
- [ ] Alerts si accuracy cae abruptamente
- [ ] A/B testing (modelo viejo vs nuevo)

## Autor

jose contreras (@jovicon)
