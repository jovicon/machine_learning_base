# Predicción de Jornadas Completas

Script para predecir resultados de partidos de una jornada completa usando modelos XGBoost entrenados.

## Uso Rápido

### Todas las ligas desde archivo multi-liga

```bash
python predict.py --partidos ejemplo_partidos.json
```

Predice **automáticamente** todas las ligas que estén en el JSON (premier, italian, german, spanish).

## Uso CLI Completo

### Opción 1: Archivo multi-liga (recomendado)

```bash
python predict.py --partidos ejemplo_partidos.json
```

Procesa todas las ligas incluidas en el archivo.

### Opción 2: Archivo multi-liga, liga específica

```bash
python predict.py --liga premier --partidos ejemplo_partidos.json
```

Procesa solo la liga especificada.

### Opción 3: Liga individual con lista simple

```bash
python predict.py \
  --liga premier \
  --jornada 33 \
  --fecha 2026-04-18 \
  --partidos '[["Brentford", "Fulham"], ["Chelsea", "Man United"]]'
```

## Formato de Partidos

### Multi-liga (archivo JSON)

```json
{
  "premier": {
    "fecha": "2026-04-18",
    "jornada": 33,
    "partidos": [
      ["Brentford", "Fulham"],
      ["Chelsea", "Man United"]
    ]
  },
  "german": {
    "fecha": "2026-04-17",
    "jornada": 31,
    "partidos": [
      ["Bayern Munich", "Stuttgart"],
      ["Dortmund", "Cologne"]
    ]
  }
}
```

### Lista simple (requiere --liga, --jornada, --fecha)

```json
[
  ["Brentford", "Fulham"],
  ["Chelsea", "Man United"],
  ["Man City", "Arsenal"]
]
```

## Script Python

```python
from predict_matchday import predecir_jornada, guardar_predicciones

partidos = [
    ('Brentford', 'Fulham'),
    ('Chelsea', 'Man United'),
]

predicciones, resumen = predecir_jornada(
    liga_codigo='premier',
    partidos=partidos,
    fecha_jornada='2026-04-18',
    jornada=33
)

guardar_predicciones('premier', predicciones, resumen)
```

## Output

### Console Output

```
📊 Predicción jornada 33 - PREMIER
Fecha: 2026-04-18
Partidos: 10

Predicciones:
--------------------------------------------------------------------------------
  Brentford            vs Fulham               → H  (59.3%) ✓ USAR
  Chelsea              vs Man United          → H  (62.1%) ✓ USAR
  Man City             vs Arsenal             → H  (54.8%) ↙ bajo
  ...

Resumen: 10 partidos
  Confianza promedio: 58.2%
  Alta confianza (>55%): 7
✓ Predicciones guardadas en ./real_models/premier_league/predicciones.json
```

### JSON Output (`predicciones.json`)

```json
[
  {
    "jornada": 33,
    "fecha": "2026-04-18",
    "home": "Brentford",
    "away": "Fulham",
    "prediccion": "H",
    "prob_H": 0.593,
    "prob_D": 0.251,
    "prob_A": 0.156,
    "confianza": 0.593,
    "resultado": null,
    "correcto": null,
    "estado": "pendiente"
  }
]
```

## Opciones de Línea de Comando

```
--liga        Liga específica (german, premier, spanish, italian)
              (opcional: si no se especifica, procesa todas las del archivo)
--jornada     Número de jornada (requerido si --partidos es lista simple)
--fecha       Fecha en formato YYYY-MM-DD (requerido si --partidos es lista simple)
--partidos    JSON con partidos o ruta a archivo [REQUIRED]
--append      Append a archivo existente (default: sobrescribe)
```

## Ligas Disponibles

- `german` - Bundesliga
- `premier` - Premier League
- `spanish` - La Liga
- `italian` - Serie A

## Filtro de Confianza

El script marca predicciones con `confianza > 55%` como `✓ USAR`, siguiendo el criterio de la CLAUDE.md.

Predicciones con menor confianza se incluyen en el JSON pero deberían ignorarse en decisiones.

## Errores Comunes

### `FileNotFoundError: Modelo no encontrado`

El modelo entrenado no existe. Ejecuta primero:

```bash
python scripts/train.py --liga <liga>
```

### `ValueError: Liga 'xyz' no encontrada`

La liga no existe. Usa: `german`, `premier`, `spanish`, `italian`

### `JSONDecodeError: Expecting value`

Verifica que el JSON sea válido. Para archivos, usa:

```bash
cat ejemplo_partidos.json  # Verifica que sea JSON válido
```

## Integración con CLAUDE.md

Este script complementa el workflow definido en CLAUDE.md:

1. **Train** (scripts/train.py) → genera modelo
2. **Predict Matchday** (este script) → genera predicciones para jornada
3. **Verify** (cuando los resultados están disponibles) → actualiza predicciones.json con resultado

## Próximas Jornadas

Para predicciones automáticas de próximas jornadas, se puede usar con un scheduler:

```bash
# Cron job para Premier League cada fin de semana a las 10am
0 10 * * 6 cd /path/to/repo && python pipelines/football_leagues/scripts/predict.py \
  --partidos partidos_week.json
```
