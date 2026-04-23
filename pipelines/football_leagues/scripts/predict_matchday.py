"""
Script para predecir resultados de una jornada completa.
Genera JSON compatible con el formato de predicciones.json
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Importar config y features
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from features import FeatureBuilder


def load_model(liga_codigo: str) -> Dict[str, Any]:
    """
    Carga el modelo entrenado de una liga.

    Args:
        liga_codigo: 'german', 'premier', 'spanish', 'italian'

    Returns:
        dict con modelo, features, label encoder, etc.
    """
    config = get_config(liga_codigo)

    # Construir ruta absoluta desde el directorio del script
    script_dir = Path(__file__).parent
    pipelines_dir = script_dir.parent.parent
    base_path = pipelines_dir / "football_leagues" / "models" / config['nombre']

    model_file = f"modelo_{config['nombre']}.pkl"
    model_path = base_path / model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    return joblib.load(model_path)


def crear_features_partido(
    home: str,
    away: str,
    fecha: str,
    df_historico: pd.DataFrame,
    builder: FeatureBuilder
) -> Dict[str, float]:
    """
    Crea features para un partido usando FeatureBuilder.

    Args:
        home: nombre equipo local
        away: nombre equipo visitante
        fecha: fecha del partido (YYYY-MM-DD)
        df_historico: DataFrame con histórico de la liga
        builder: FeatureBuilder instanciado

    Returns:
        dict con features necesarias para predicción, o None si no hay datos
    """
    # Convertir fecha a timestamp
    fecha_ts = pd.Timestamp(fecha)

    # Obtener estadísticas de cada equipo hasta esta fecha
    home_stats = builder.get_team_stats_v2(home, fecha_ts)
    away_stats = builder.get_team_stats_v2(away, fecha_ts)

    if home_stats is None or away_stats is None:
        return None

    # Combinar estadísticas
    features = {}

    # Agregar features del local
    for key, value in home_stats.items():
        features[f'h_{key}'] = value

    # Agregar features del visitante
    for key, value in away_stats.items():
        features[f'a_{key}'] = value

    # Calcular features derivados (diferencias)
    features['dif_recent_wr'] = home_stats.get('recent_win_rate', 0) - away_stats.get('recent_win_rate', 0)
    features['dif_season_wr'] = home_stats.get('season_win_rate', 0) - away_stats.get('season_win_rate', 0)
    features['dif_recent_gf'] = home_stats.get('recent_gf_pg', 0) - away_stats.get('recent_gf_pg', 0)
    features['dif_recent_gc'] = home_stats.get('recent_gc_pg', 0) - away_stats.get('recent_gc_pg', 0)
    features['dif_season_dif'] = home_stats.get('season_dif_goles', 0) - away_stats.get('season_dif_goles', 0)
    features['dif_recent_dif'] = home_stats.get('recent_dif_goles', 0) - away_stats.get('recent_dif_goles', 0)
    features['home_advantage'] = 1  # Ventaja de localía
    features['dif_pts_pg'] = home_stats.get('season_pts_pg', 0) - away_stats.get('season_pts_pg', 0)
    features['dif_posicion'] = away_stats.get('tabla_posicion', 20) - home_stats.get('tabla_posicion', 20)

    return features


def predecir_partido(
    home: str,
    away: str,
    fecha: str,
    jornada: int,
    model_data: Dict[str, Any],
    df_historico: pd.DataFrame,
    builder: FeatureBuilder
) -> Dict[str, Any]:
    """
    Realiza predicción para un partido individual.

    Args:
        home: nombre equipo local
        away: nombre equipo visitante
        fecha: fecha en formato 'YYYY-MM-DD'
        jornada: número de jornada
        model_data: dict cargado del modelo
        df_historico: DataFrame con histórico
        builder: FeatureBuilder instanciado

    Returns:
        dict con predicción y probabilidades
    """
    modelo = model_data.get('model_v3')
    feature_cols = model_data.get('feature_cols_v3')
    le = model_data.get('le')

    if not all([modelo, feature_cols, le]):
        raise ValueError("Modelo incompleto: faltan componentes clave")

    # Crear features para el partido
    features_dict = crear_features_partido(home, away, fecha, df_historico, builder)

    if features_dict is None:
        return {
            'jornada': jornada,
            'fecha': fecha,
            'home': home,
            'away': away,
            'error': f"No hay histórico para {home} o {away}"
        }

    # Crear DataFrame con features
    X = pd.DataFrame([features_dict])

    # Asegurar que tiene todas las features necesarias
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]

    # Predicción
    pred_proba = modelo.predict_proba(X)[0]
    pred_class = modelo.predict(X)[0]
    pred_label = le.inverse_transform([pred_class])[0]

    # Confianza = máxima probabilidad
    confianza = pred_proba.max()

    # Mapear índices a labels (asumiendo orden: away=0, draw=1, home=2)
    prob_mapping = {le.classes_[i]: pred_proba[i] for i in range(len(le.classes_))}

    return {
        'jornada': jornada,
        'fecha': fecha,
        'home': home,
        'away': away,
        'prediccion': pred_label,
        'prob_H': float(prob_mapping.get('H', 0)),
        'prob_D': float(prob_mapping.get('D', 0)),
        'prob_A': float(prob_mapping.get('A', 0)),
        'confianza': float(confianza),
        'resultado': None,
        'correcto': None,
        'estado': 'pendiente'
    }


def predecir_jornada(
    liga_codigo: str,
    partidos: List[Tuple[str, str]],
    fecha_jornada: str,
    jornada: int,
    df_historico: pd.DataFrame = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Predice todos los partidos de una jornada.

    Args:
        liga_codigo: 'german', 'premier', 'spanish', 'italian'
        partidos: lista de tuplas (home, away)
        fecha_jornada: fecha en formato 'YYYY-MM-DD'
        jornada: número de jornada
        df_historico: DataFrame con histórico (se carga del modelo si no se proporciona)

    Returns:
        (predicciones, resumen) - lista de predicciones y dict resumen
    """
    # Cargar modelo
    model_data = load_model(liga_codigo)

    # Usar histórico del modelo si no se proporciona
    if df_historico is None:
        df_historico = model_data.get('df')
        if df_historico is None:
            raise ValueError("No hay histórico en el modelo")

    # Instanciar FeatureBuilder
    builder = FeatureBuilder(df_historico)

    predicciones = []
    for home, away in partidos:
        pred = predecir_partido(
            home=home,
            away=away,
            fecha=fecha_jornada,
            jornada=jornada,
            model_data=model_data,
            df_historico=df_historico,
            builder=builder
        )
        predicciones.append(pred)

    # Resumen
    predicciones_validas = [p for p in predicciones if 'error' not in p]
    resumen = {
        'partidos_predichos': len(predicciones_validas),
        'partidos_con_error': len(predicciones) - len(predicciones_validas),
        'confianza_promedio': np.mean([p['confianza'] for p in predicciones_validas]) if predicciones_validas else 0,
        'predicciones_alta_confianza': len([p for p in predicciones_validas if p['confianza'] > 0.55])
    }

    return predicciones, resumen


def guardar_predicciones(
    liga_codigo: str,
    predicciones: List[Dict[str, Any]],
    resumen: Dict[str, Any] = None,
    append: bool = False
) -> Path:
    """
    Guarda predicciones en JSON con formato compatible predicciones.json

    Args:
        liga_codigo: 'german', 'premier', 'spanish', 'italian'
        predicciones: lista de predicciones
        resumen: dict resumen (opcional)
        append: si True, append a archivo existente; si False, sobrescribe

    Returns:
        Path del archivo guardado
    """
    config = get_config(liga_codigo)
    output_dir = Path(f"./real_models/{config['nombre']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'predicciones.json'

    # Formato compatible con predicciones.json existente
    if append and output_file.exists():
        with open(output_file, 'r') as f:
            existing = json.load(f)
        existing.extend(predicciones)
        data = existing
    else:
        data = predicciones

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Predicciones guardadas en {output_file}")
    if resumen:
        print(f"  Partidos: {resumen['partidos_predichos']}")
        print(f"  Confianza promedio: {resumen['confianza_promedio']:.2%}")
        print(f"  Alta confianza (>55%): {resumen['predicciones_alta_confianza']}")

    return output_file


def main():
    """Ejemplo de uso del script."""
    print("Ejemplo: predecir_jornada() con múltiples ligas\n")
    print("Para uso CLI, ejecuta: python predict.py --help")


if __name__ == '__main__':
    main()
