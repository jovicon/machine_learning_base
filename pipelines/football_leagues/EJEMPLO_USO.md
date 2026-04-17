# Ejemplos de Uso del Pipeline

## En Terminal (recomendado para pipelines)

### Entrenar una liga desde cero

```bash
cd pipelines/football_leagues
python scripts/train.py --liga german
```

### Entrenar todas las ligas

```bash
cd pipelines/football_leagues
python scripts/train.py --todas
```

Output esperado:

```
============================================================
ENTRENAMIENTO: Bundesliga
============================================================

Paso 1: Cargando datos...
✓ Temporada 2021: 306 partidos
✓ Temporada 2122: 300 partidos
...
Total partidos cargados: 1782

Paso 2: Construyendo features (~5 min)...
  Procesado 200/1734 partidos...
  Procesado 400/1734 partidos...
...
Dataset construido:
  Partidos: 1734
  Features: 52

Paso 3: Dividiendo train/test...
  Train: 1485 partidos
  Test:  249 partidos
  Features: 52

Paso 4: Entrenando XGBoost...

Paso 5: Calibrando probabilidades...

Paso 6: Evaluando modelo...
  Accuracy general:        53.8%
  Accuracy alta conf (>55%): 64.4% (59 partidos)
  Log Loss:                1.0168
  Baseline (siempre H):    44.2%
  Mejora vs baseline:      +9.6%

Paso 7: Guardando modelo...
  Principal:  ../../real_models/german_league/modelo_german.pkl
  Versión:    ../../real_models/german_league/versiones/modelo_2026-04-17.pkl
  Fecha:      2026-04-17
```

### Reentrenar después de 3 jornadas

```bash
python scripts/retrain.py --liga german
```

Output esperado:

```
============================================================
REENTRENAMIENTO: Bundesliga
============================================================

Modelo anterior (2026-04-08):
  Accuracy:          53.8%
  Accuracy alta conf: 64.4%
  Partidos train:    1485

Cargando datos frescos...
✓ Temporada 2021: 306 partidos
...

Partidos nuevos disponibles: 45

Construyendo features (~5 min)...
...

Entrenando nuevo modelo...

Evaluando nuevo modelo...

Comparacion:
  Metrica                  Anterior      Nuevo        Cambio
  ────────────────────────────────────────────────────────
  Accuracy                 53.8%         54.2%        +0.4%
  Accuracy alta conf       64.4%         65.1%        +0.7%
  Partidos train           1485          1530

✓ MODELO ACTUALIZADO
  Principal:  ../../real_models/german_league/modelo_german.pkl
  Versión:    ../../real_models/german_league/versiones/modelo_2026-04-17.pkl
```

### Ver status de modelos

```bash
python scripts/status.py
```

Output:

```
============================================================
STATUS MODELOS ENTRENADOS
============================================================

Liga: Bundesliga (german)
  Fecha:         2026-04-08
  Accuracy:      53.8%
  Alta conf (>55%): 64.4% (59 partidos)
  Train/Test:    1485 / 249
  Desde entreno: 9 dias

Liga: Premier League (premier)
  ✗ No existe modelo
```

## En Jupyter Notebook

### Caso 1: Entrenar desde cero

```python
import sys
from pathlib import Path

# Agregar modulo al path
sys.path.insert(0, str(Path.cwd().parent / 'pipelines'))

from football_leagues.config import get_config
from football_leagues.model import Entrenador

# Configurar German League
config = get_config('german')

# Entrenar
entrenador = Entrenador(config)
metricas = entrenador.entrenar()

# Acceder metricas
print(f"Accuracy: {metricas['accuracy']:.1%}")
print(f"Accuracy alta conf: {metricas['accuracy_alta_confianza']:.1%}")
print(f"Mejora vs baseline: {metricas['mejora_baseline']:+.1%}")
```

### Caso 2: Reentrenar

```python
from football_leagues.config import get_config
from football_leagues.model import Reentrenador

config = get_config('german')
reentrenador = Reentrenador(config)

# Reentrenar
actualizado = reentrenador.retrain(umbral_degradacion=0.02)

if actualizado:
    print("✓ Modelo actualizado correctamente")
else:
    print("✗ Modelo anterior mantiene su desempeño")
```

### Caso 3: Usar modelo para predicciones

```python
import joblib
from pathlib import Path
from football_leagues.config import get_config

config = get_config('german')
ruta_modelo = Path(config['ruta_modelo']) / f"modelo_{config['nombre']}.pkl"

# Cargar modelo
datos = joblib.load(ruta_modelo)
model = datos['model_v3']
feature_cols = datos['feature_cols_v3']
le = datos['le']
df = datos['df']

# Ahora puedes usar para predicciones
# (Ver german_league_app.ipynb para ejemplo completo)
```

### Caso 4: Comparar multiples ligas

```python
from football_leagues.config import listar_ligas, get_config
import joblib
from pathlib import Path
import pandas as pd

resultados = []

for liga in listar_ligas():
    try:
        config = get_config(liga)
        ruta = Path(config['ruta_modelo']) / f"modelo_{config['nombre']}.pkl"
        
        if not ruta.exists():
            resultados.append({
                'liga': config['nombre_completo'],
                'status': 'No existe',
                'accuracy': None,
                'fecha': None
            })
            continue
        
        datos = joblib.load(ruta)
        resultados.append({
            'liga': config['nombre_completo'],
            'status': 'OK',
            'accuracy': datos['accuracy'],
            'accuracy_alta': datos['accuracy_alta_confianza'],
            'fecha': datos['fecha_entrenamiento']
        })
    except Exception as e:
        resultados.append({
            'liga': config['nombre_completo'],
            'status': f'Error: {e}',
            'accuracy': None,
            'fecha': None
        })

df_status = pd.DataFrame(resultados)
print(df_status.to_string())
```

## Para Cron/GitHub Actions

### Script wrapper para produccion

```bash
#!/bin/bash
# retrain_all.sh

set -e  # Exit on error

cd /path/to/machine_learning_base/pipelines/football_leagues

echo "[$(date)] Iniciando reentrenamiento..."

python scripts/retrain.py --todas --threshold 0.02

echo "[$(date)] Reentrenamiento completado"
```

### En crontab

```cron
# Reentrenar modelos cada domingo a las 2 AM
0 2 * * 0 cd /home/user/machine_learning_base && ./retrain_all.sh >> logs/retrain.log 2>&1
```

## Tips y Trucos

### Debug: Ver features generadas

```python
from football_leagues.data import DataLoader
from football_leagues.features import FeatureBuilder
from football_leagues.config import get_config

config = get_config('german')
loader = DataLoader(config)
df = loader.cargar_datos_crudos()

builder = FeatureBuilder(df)
df_features = builder.construir_dataset()

# Ver nombres de features
print(df_features.columns.tolist())

# Ver una fila
print(df_features.iloc[0])
```

### Personalizar umbral de reentrenamiento

```bash
# Strict: solo si no degrada nada (0%)
python scripts/retrain.py --liga german --threshold 0.00

# Tolerante: permite hasta 5% de degradacion
python scripts/retrain.py --liga german --threshold 0.05
```

### Entrenar todas excepto una

```bash
python scripts/train.py --liga german --liga premier --liga spanish
# Omite italian_league
```

### Logging a archivo

```bash
python scripts/retrain.py --todas 2>&1 | tee reentrenamiento_$(date +%Y%m%d_%H%M%S).log
```
