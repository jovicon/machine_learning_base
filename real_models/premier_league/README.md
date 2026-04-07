# Premier League — Modelo de Predicción de Resultados

Modelo XGBoost para predecir probabilidades de resultado (H/D/A) en partidos de la Premier League.

---

## Archivos del proyecto

```
premier_league/
├── premier_league_train.ipynb   # Entrena y guarda el modelo
├── premier_league_app.ipynb     # Predice partidos y rastrea resultados
├── modelo_premier.pkl           # Modelo entrenado (generado por train)
├── predicciones.json            # Historial de predicciones y resultados
└── README.md                    # Este archivo
```

---

## Requisitos

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

---

## Arquitectura del modelo

### Datos

- Fuente: [football-data.co.uk](https://www.football-data.co.uk/data.php)
- Liga: Premier League (código E0)
- Temporadas: 2020/21 → 2025/26 (6 temporadas, ~2200 partidos)
- Formato: CSV por temporada, descarga directa sin API key

### Features (52 total por partido)

Cada partido tiene features de ambos equipos calculados antes del partido:

| Grupo | Features | Descripción |
|---|---|---|
| Forma reciente | `recent_gf_pg`, `recent_gc_pg`, `recent_dif_goles`, `recent_sot_f_pg`, `recent_sot_c_pg`, `recent_win_rate`, `recent_draw_rate`, `recent_home_wr` | Stats de los últimos 5 partidos |
| Temporada | `season_gf_pg`, `season_gc_pg`, `season_dif_goles`, `season_sot_f_pg`, `season_sot_c_pg`, `season_win_rate`, `season_draw_rate`, `season_home_wr` | Stats de la temporada actual |
| H2H | `h2h_home_wr`, `h2h_draw_rate`, `h2h_away_wr` | Historial directo entre los dos equipos (últimos 10) |
| Tabla | `posicion`, `pts_pg`, `dif_goles_szn`, `pct_posicion` | Posición en tabla al momento del partido |
| Diferenciales | `dif_recent_wr`, `dif_season_wr`, `dif_recent_gf`, `dif_recent_gc`, `dif_recent_dif`, `dif_season_dif`, `home_advantage`, `dif_pts_pg`, `dif_posicion` | Diferencia entre local y visitante |

Todos los features usan el prefijo `h_` para el equipo local y `a_` para el visitante.

### Algoritmo

- **XGBoost** con calibración de probabilidades (CalibratedClassifierCV, método isotonic)
- Split temporal: train = temporadas anteriores, test = temporada actual
- Sin feature scaling (XGBoost no lo requiere)

### Métricas del modelo actual

| Métrica | Valor |
|---|---|
| Accuracy general | 47.4% |
| Accuracy con confianza >55% | 59.3% |
| Baseline (siempre predecir local) | 41.2% |
| Partidos con confianza >55% | ~30% del total |

### Umbral de uso

**Solo usar predicciones con confianza > 55%.** El modelo descarta el 70% de los partidos donde no tiene suficiente certeza.

```
Confianza 40-45%  →  32% accuracy  ← peor que el azar, ignorar
Confianza 45-50%  →  47% accuracy  ← marginal
Confianza 50-55%  →  43% accuracy  ← ignorar
Confianza  >55%   →  59% accuracy  ← usar
```

---

## Uso

### Primera vez — entrenar el modelo

```
1. Abrir premier_league_train.ipynb
2. Correr todas las celdas (1 → 8)
3. Esperar ~5 minutos (celda 4 es la más lenta)
4. Se genera modelo_premier.pkl
```

### Cada jornada — predecir y registrar

**Antes de los partidos:**

```
1. Abrir premier_league_app.ipynb
2. Correr celdas 1, 2, 3 (setup)
3. Editar celda 4: lista de partidos y fecha de la jornada
4. Correr celda 4 (predicciones)
5. Editar celda 5: solo los partidos marcados ← USAR
6. Correr celda 5 (registra y guarda en predicciones.json)
```

**Después de cada partido:**

```
1. Abrir premier_league_app.ipynb
2. Correr celdas 1, 2, 3 (setup)
3. Editar celda 6: cambiar 'H'/'D'/'A' por resultado real
4. Correr celda 6
5. Correr celda 7 para ver resumen actualizado
```

### Cada 3 jornadas — reentrenar

```
1. Abrir premier_league_train.ipynb
2. Correr solo celda 9 (reentrenamiento automático)
3. El notebook detecta si hay suficientes partidos nuevos
4. Compara el nuevo modelo contra el anterior
5. Solo reemplaza modelo_premier.pkl si el accuracy no degrada
```

---

## Reentrenamiento — guía de decisión

### Cuándo reentrenar

| Situación | Acción |
|---|---|
| Cada 3 jornadas (~30 partidos nuevos) | Correr celda 9 del train notebook |
| Inicio de temporada nueva (agosto) | Correr todas las celdas (1→8) |
| Accuracy real cae >5% durante 4 semanas | Investigar y reentrenar |
| Nuevo equipo ascendido sin historia | Reentrenar con datos frescos |

### Por qué no reentrenar cada semana

Una jornada agrega solo 10 partidos. Con 10 filas nuevas las métricas no se mueven significativamente y el costo computacional (~5 minutos) no justifica el resultado. El umbral mínimo recomendado es 20-30 partidos nuevos (2-3 jornadas).

### Cómo funciona la comparación de modelos (celda 9)

El modelo anterior guarda su accuracy en el pkl al momento de entrenarse:

```python
# Al guardar (celda 8):
joblib.dump({
    'model_v3':                model_v3,
    'accuracy':                0.593,   # guardado aquí
    'accuracy_alta_confianza': 0.593,
    ...
}, 'modelo_premier.pkl')
```

Al reentrenar (celda 9), se carga ese valor y se compara:

```python
modelo_anterior   = joblib.load('modelo_premier.pkl')
accuracy_anterior = modelo_anterior['accuracy_alta_confianza']  # 0.593

# Entrenar nuevo modelo...
nuevo_accuracy = evaluar()  # 0.601

# Umbral de tolerancia: -2%
# Solo rechazar si la degradación supera el 2%
if nuevo_accuracy >= accuracy_anterior - 0.02:
    # 0.601 >= 0.593 - 0.02 → True → reemplazar pkl ✓
    joblib.dump(nuevo_modelo, 'modelo_premier.pkl')
```

**Por qué -0.02 (2%) y no 0:**
La diferencia entre 0.589 y 0.593 es ruido estadístico (0.4%), no una degradación real. El umbral de 2% filtra el ruido y solo rechaza cuando hay una caída significativa. Ajústalo en la variable `UMBRAL_DEGRADACION` de la celda 9.

### Qué usar como métrica de comparación

La celda 9 usa `accuracy_alta_confianza` (accuracy en partidos con confianza >55%) en lugar del accuracy general. Esto es porque:

- El accuracy general incluye todos los partidos, muchos de los cuales el modelo rechaza con `← IGNORAR`
- Lo que importa para el uso real es qué tan preciso es el modelo cuando decide recomendar
- Un modelo que pasa de 59% a 57% en alta confianza es más relevante que uno que pasa de 47% a 45% en general

---

## Mejoras futuras

### Mejoras de datos

| Mejora | Impacto estimado | Complejidad |
|---|---|---|
| Cuotas de apertura (B365H, B365D, B365A) | +5-6% accuracy | Baja — ya están en el CSV |
| Más temporadas históricas (hasta 1993) | +2-3% accuracy | Baja — mismo código |
| Otras ligas (La Liga, Bundesliga, etc.) | +2-3% accuracy | Baja — mismo código |
| xG (expected goals) via understat.com | +3-4% accuracy | Media |
| Lesiones y suspensiones | +4-5% accuracy | Alta |

### Mejoras de modelo

| Mejora | Descripción |
|---|---|
| GridSearchCV de hiperparámetros | Optimizar n_estimators, max_depth, learning_rate |
| Ensemble (XGBoost + LR + RF) | Combinar modelos para predicciones más estables |
| Features de racha | Victorias/derrotas consecutivas al momento del partido |
| Días de descanso | Diferencia de días desde el último partido de cada equipo |

### Integración con FastAPI (próximo paso)

```python
# main.py
import joblib
from fastapi import FastAPI

app  = FastAPI()
data = joblib.load('modelo_premier.pkl')  # carga idéntica al app notebook

@app.post('/predecir')
def predecir(home: str, away: str, fecha: str):
    # misma lógica que predecir_partido_v3()
    return {'prob_H': 0.648, 'prob_D': 0.204, 'prob_A': 0.148}
```

---

## Limitaciones del modelo

1. **Empates difíciles de predecir** — recall de empates es muy bajo (~2%). El fútbol tiene alta varianza en empates.

2. **Equipos recién ascendidos** — los primeros 5-8 partidos de un equipo nuevo en la liga son menos precisos por falta de historia.

3. **Inicio de temporada** — hasta la jornada 8, las stats de temporada son muy pequeñas y el modelo se apoya más en la historia anterior.

4. **Techo real** — con datos públicos, el techo realista es ~55-57% de accuracy en alta confianza. Las casas de apuesta con datos privados (lesiones, estado físico) llegan a 58-60%.

5. **Una sola liga** — el modelo solo conoce la Premier League. No generaliza a otras ligas sin reentrenamiento.

---

## Fuentes de datos

- **football-data.co.uk** — datos históricos de resultados, stats y cuotas. Gratis, sin registro.
  - URL Premier League: `https://www.football-data.co.uk/mmz4281/{temporada}/E0.csv`
  - Temporadas disponibles: 1993/94 hasta la actual
  - Actualización: dos veces por semana (domingos y miércoles)

---

## Comparación con el mercado

| Fuente | Probabilidad Arsenal gana vs Bournemouth |
|---|---|
| Este modelo | 64.8% |
| Polymarket | 69% |
| Betsson | 69.9% (cuota 1.43) |

El modelo está alineado con el mercado. La diferencia del ~5% se explica por factores que el modelo no captura (lesiones, rotaciones, contexto europeo).
