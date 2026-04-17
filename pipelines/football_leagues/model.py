"""
Modulo de entrenamiento y reentrenamiento de modelos
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .data import DataLoader
from .features import FeatureBuilder


class Entrenador:
    """Entrena modelo XGBoost para una liga desde cero."""

    def __init__(self, config: Dict):
        """
        Args:
            config: diccionario con configuracion de liga
        """
        self.config = config
        self.liga = config['nombre']
        self.temporada_test = config['temporada_test']

        # Crear carpeta modelo si no existe
        self.ruta_modelo = Path(config['ruta_modelo'])
        self.ruta_modelo.mkdir(parents=True, exist_ok=True)
        self.ruta_versiones = self.ruta_modelo / 'versiones'
        self.ruta_versiones.mkdir(parents=True, exist_ok=True)

    def entrenar(self) -> Dict:
        """
        Flujo completo de entrenamiento:
        1. Cargar datos
        2. Construir features
        3. Dividir train/test
        4. Entrenar modelo
        5. Calibrar
        6. Evaluar
        7. Guardar

        Returns:
            dict con metricas de entrenamiento
        """
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO: {self.config['nombre_completo']}")
        print(f"{'='*60}\n")

        # Paso 1: Cargar datos
        print("Paso 1: Cargando datos...")
        loader = DataLoader(self.config)
        df = loader.cargar_datos_crudos()

        # Paso 2: Construir features
        print("\nPaso 2: Construyendo features (~5 min)...")
        builder = FeatureBuilder(df)
        df_features = builder.construir_dataset()

        # Paso 3: Dividir train/test
        print("\nPaso 3: Dividiendo train/test...")
        train_df = df_features[df_features['season'] != self.temporada_test].copy()
        test_df = df_features[df_features['season'] == self.temporada_test].copy()

        feature_cols = [c for c in df_features.columns
                       if c not in ['date', 'season', 'home_team', 'away_team', 'resultado']]

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        le = LabelEncoder()
        y_train = le.fit_transform(train_df['resultado'])
        y_test = le.transform(test_df['resultado'])

        print(f"  Train: {len(X_train)} partidos")
        print(f"  Test:  {len(X_test)} partidos")
        print(f"  Features: {len(feature_cols)}")

        # Paso 4: Entrenar
        print("\nPaso 4: Entrenando XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        xgb.fit(X_train, y_train)

        # Paso 5: Calibrar
        print("Paso 5: Calibrando probabilidades...")
        model = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
        model.fit(X_train, y_train)

        # Paso 6: Evaluar
        print("\nPaso 6: Evaluando modelo...")
        metricas = self._evaluar(model, X_test, y_test, test_df, le)

        # Paso 7: Guardar
        print("\nPaso 7: Guardando modelo...")
        self._guardar_modelo(model, feature_cols, le, df, metricas)

        return metricas

    def _evaluar(self, model, X_test, y_test, test_df, le) -> Dict:
        """Evalua modelo en test set."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)

        # Confianza alta (>55%)
        proba_df = pd.DataFrame(y_pred_proba, columns=['prob_A', 'prob_D', 'prob_H'])
        proba_df['confianza'] = proba_df[['prob_A', 'prob_D', 'prob_H']].max(axis=1)
        proba_df['pred'] = le.inverse_transform(y_pred)
        proba_df['real'] = le.inverse_transform(y_test)
        proba_df['correcto'] = proba_df['pred'] == proba_df['real']

        mask_alta = proba_df['confianza'] > 0.55
        accuracy_alta = proba_df[mask_alta]['correcto'].mean() if mask_alta.sum() > 0 else 0
        n_alta = mask_alta.sum()

        baseline = (test_df['resultado'] == 'H').mean()

        print(f"  Accuracy general:        {accuracy:.1%}")
        print(f"  Accuracy alta conf (>55%): {accuracy_alta:.1%} ({n_alta} partidos)")
        print(f"  Log Loss:                {logloss:.4f}")
        print(f"  Baseline (siempre H):    {baseline:.1%}")
        print(f"  Mejora vs baseline:      {(accuracy - baseline):+.1%}")

        return {
            'accuracy': round(accuracy, 4),
            'accuracy_alta_confianza': round(accuracy_alta, 4),
            'logloss': round(logloss, 4),
            'n_alta_confianza': int(n_alta),
            'baseline': baseline,
            'mejora_baseline': round(accuracy - baseline, 4),
        }

    def _guardar_modelo(self, model, feature_cols, le, df, metricas):
        """Guarda modelo con versionado por fecha."""
        fecha = datetime.now().strftime('%Y-%m-%d')

        payload = {
            'model_v3': model,
            'feature_cols_v3': feature_cols,
            'le': le,
            'df': df,
            'accuracy': metricas['accuracy'],
            'accuracy_alta_confianza': metricas['accuracy_alta_confianza'],
            'logloss': metricas['logloss'],
            'n_alta_confianza': metricas['n_alta_confianza'],
            'fecha_entrenamiento': fecha,
            'temporadas': self.config['temporadas'],
            'temporada_test': self.temporada_test,
        }

        # Guardar principal
        path_actual = self.ruta_modelo / f"modelo_{self.liga}.pkl"
        joblib.dump(payload, path_actual)

        # Guardar versionado
        path_version = self.ruta_versiones / f"modelo_{fecha}.pkl"
        joblib.dump(payload, path_version)

        print(f"  Principal:  {path_actual}")
        print(f"  Versión:    {path_version}")
        print(f"  Fecha:      {fecha}")


class Reentrenador:
    """Reentrena modelo con datos frescos y lo actualiza si cumple criterios."""

    def __init__(self, config: Dict):
        """
        Args:
            config: diccionario con configuracion de liga
        """
        self.config = config
        self.liga = config['nombre']
        self.temporada_test = config['temporada_test']
        self.ruta_modelo = Path(config['ruta_modelo'])

    def retrain(self, umbral_degradacion: float = 0.02) -> bool:
        """
        Reentrena modelo con datos frescos.

        Solo reemplaza el pkl si:
            accuracy_nuevo >= accuracy_anterior - umbral_degradacion

        Args:
            umbral_degradacion: tolerancia de degradacion (default 2%)

        Returns:
            True si modelo fue actualizado, False si se mantuvo el anterior
        """
        print(f"\n{'='*60}")
        print(f"REENTRENAMIENTO: {self.config['nombre_completo']}")
        print(f"{'='*60}\n")

        # Paso 1: Cargar metricas modelo anterior
        path_modelo = self.ruta_modelo / f"modelo_{self.liga}.pkl"
        try:
            ant = joblib.load(path_modelo)
            acc_ant = ant['accuracy']
            accf_ant = ant['accuracy_alta_confianza']
            n_ant = ant.get('partidos_train', 0) or len(ant.get('df', []))
            fecha_ant = ant['fecha_entrenamiento']
            print(f"Modelo anterior ({fecha_ant}):")
            print(f"  Accuracy:          {acc_ant:.1%}")
            print(f"  Accuracy alta conf: {accf_ant:.1%}")
            print(f"  Partidos train:    {n_ant}")
        except FileNotFoundError:
            print("✗ No hay modelo anterior")
            return False

        # Paso 2: Cargar datos frescos
        print(f"\nCargando datos frescos...")
        loader = DataLoader(self.config)
        df_new = loader.cargar_datos_crudos()

        # Paso 3: Detectar partidos nuevos
        train_df = df_new[df_new['season'] != self.temporada_test]
        n_train_nuevo = len(train_df)
        n_nuevos = n_train_nuevo - n_ant

        print(f"\nPartidos nuevos disponibles: {n_nuevos}")
        if n_nuevos < 20 and n_ant > 0:
            print(f"→ Insuficientes (<20). Espera ~3 jornadas más.")
            return False

        # Paso 4: Construir features
        print(f"\nConstruyendo features (~5 min)...")
        builder = FeatureBuilder(df_new)
        df_features_new = builder.construir_dataset()

        # Paso 5: Entrenar nuevo modelo
        print(f"\nEntrenando nuevo modelo...")
        train_nuevo = df_features_new[df_features_new['season'] != self.temporada_test].copy()
        test_nuevo = df_features_new[df_features_new['season'] == self.temporada_test].copy()

        feature_cols_new = [c for c in df_features_new.columns
                           if c not in ['date', 'season', 'home_team', 'away_team', 'resultado']]

        X_train_new = train_nuevo[feature_cols_new]
        X_test_new = test_nuevo[feature_cols_new]

        le_new = LabelEncoder()
        y_train_new = le_new.fit_transform(train_nuevo['resultado'])
        y_test_new = le_new.transform(test_nuevo['resultado'])

        xgb_new = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42, eval_metric='mlogloss', verbosity=0
        )
        xgb_new.fit(X_train_new, y_train_new)

        model_new = CalibratedClassifierCV(xgb_new, cv=5, method='isotonic')
        model_new.fit(X_train_new, y_train_new)

        # Paso 6: Evaluar
        print(f"\nEvaluando nuevo modelo...")
        y_pred_new = model_new.predict(X_test_new)
        y_pred_proba_new = model_new.predict_proba(X_test_new)

        acc_nuevo = accuracy_score(y_test_new, y_pred_new)
        logloss_nuevo = log_loss(y_test_new, y_pred_proba_new)

        proba_df_new = pd.DataFrame(y_pred_proba_new, columns=['p_A', 'p_D', 'p_H'])
        proba_df_new['conf'] = proba_df_new[['p_A', 'p_D', 'p_H']].max(axis=1)
        proba_df_new['pred'] = le_new.inverse_transform(y_pred_new)
        proba_df_new['real'] = le_new.inverse_transform(y_test_new)
        proba_df_new['ok'] = proba_df_new['pred'] == proba_df_new['real']
        accf_nuevo = proba_df_new[proba_df_new['conf'] > 0.55]['ok'].mean() \
                     if (proba_df_new['conf'] > 0.55).sum() > 0 else 0

        # Paso 7: Comparar y decidir
        print(f"\nComparacion:")
        print(f"  {'Metrica':<25} {'Anterior':<12} {'Nuevo':<12} {'Cambio':<12}")
        print(f"  {'-'*60}")
        print(f"  {'Accuracy':<25} {acc_ant:.1%}{'':<7} {acc_nuevo:.1%}{'':<7} {(acc_nuevo-acc_ant):+.1%}")
        print(f"  {'Accuracy alta conf':<25} {accf_ant:.1%}{'':<7} {accf_nuevo:.1%}{'':<7} {(accf_nuevo-accf_ant):+.1%}")
        print(f"  {'Partidos train':<25} {n_ant}{'':<13} {len(X_train_new)}")

        # Decision
        if acc_nuevo >= acc_ant - umbral_degradacion:
            fecha_hoy = datetime.now().strftime('%Y-%m-%d')

            payload_nuevo = {
                'model_v3': model_new,
                'feature_cols_v3': feature_cols_new,
                'le': le_new,
                'df': df_new,
                'accuracy': round(acc_nuevo, 4),
                'accuracy_alta_confianza': round(accf_nuevo, 4),
                'logloss': round(logloss_nuevo, 4),
                'n_alta_confianza': int((proba_df_new['conf'] > 0.55).sum()),
                'fecha_entrenamiento': fecha_hoy,
                'temporadas': self.config['temporadas'],
                'temporada_test': self.temporada_test,
                'partidos_train': len(X_train_new),
                'partidos_test': len(X_test_new),
            }

            path_actual = self.ruta_modelo / f"modelo_{self.liga}.pkl"
            path_version = self.ruta_modelo / 'versiones' / f"modelo_{fecha_hoy}.pkl"

            joblib.dump(payload_nuevo, path_actual)
            joblib.dump(payload_nuevo, path_version)

            print(f"\n{'✓ MODELO ACTUALIZADO'}")
            print(f"  Principal:  {path_actual}")
            print(f"  Versión:    {path_version}")
            return True
        else:
            print(f"\n{'✗ MODELO NO ACTUALIZADO'}")
            print(f"  Accuracy bajo {umbral_degradacion:.0%} del anterior")
            print(f"  Requiere degradacion máx {umbral_degradacion:.0%}, obtuvo {(acc_ant - acc_nuevo):+.1%}")
            return False
