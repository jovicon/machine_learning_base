#!/usr/bin/env python3
"""
Script para ver status de todos los modelos

Usa:
    python scripts/status.py
    python scripts/status.py --liga german
    python scripts/status.py --detalle              # Mostrar info completa
"""

import sys
import argparse
import joblib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from football_leagues.config import get_config, listar_ligas


def mostrar_status(config, detalle=False):
    """Muestra status de un modelo."""
    ruta_modelo = Path(config['ruta_modelo']) / f"modelo_{config['nombre']}.pkl"

    if not ruta_modelo.exists():
        print(f"  ✗ No existe modelo")
        return False

    try:
        modelo = joblib.load(ruta_modelo)
        fecha = modelo.get('fecha_entrenamiento', '?')
        acc = modelo.get('accuracy', 0)
        acc_alta = modelo.get('accuracy_alta_confianza', 0)
        n_alta = modelo.get('n_alta_confianza', 0)
        train = modelo.get('partidos_train', '?')
        test = modelo.get('partidos_test', '?')

        print(f"  Fecha:         {fecha}")
        print(f"  Accuracy:      {acc:.1%}")
        print(f"  Alta conf (>55%): {acc_alta:.1%} ({n_alta} partidos)")
        print(f"  Train/Test:    {train} / {test}")

        if detalle:
            print(f"  Log Loss:      {modelo.get('logloss', '?'):.4f}")
            print(f"  Features:      {len(modelo.get('feature_cols_v3', []))}")
            print(f"  Temporadas:    {', '.join(modelo.get('temporadas', []))}")

        # Dias desde entrenamiento
        try:
            fecha_obj = datetime.strptime(fecha, '%Y-%m-%d')
            dias = (datetime.now() - fecha_obj).days
            print(f"  Desde entreno: {dias} dias")
        except:
            pass

        return True
    except Exception as e:
        print(f"  ✗ Error al cargar: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Ver status de modelos entrenados'
    )
    parser.add_argument(
        '--liga',
        help='Liga especifica'
    )
    parser.add_argument(
        '--detalle',
        action='store_true',
        help='Mostrar informacion detallada'
    )

    args = parser.parse_args()

    ligas = [args.liga] if args.liga else listar_ligas()

    print(f"\n{'='*60}")
    print("STATUS MODELOS ENTRENADOS")
    print(f"{'='*60}\n")

    for liga in ligas:
        try:
            config = get_config(liga)
            print(f"Liga: {config['nombre_completo']} ({liga})")
            mostrar_status(config, args.detalle)
            print()
        except ValueError as e:
            print(f"✗ {e}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
