#!/usr/bin/env python3
"""
Script para reentrenar modelos con datos frescos

Reentrena periodicamente (cada ~3 jornadas = ~27 partidos nuevos).
Solo actualiza si accuracy no degrada mas del umbral (default 2%).

Uso:
    python scripts/retrain.py --liga german
    python scripts/retrain.py --liga german --threshold 0.01  # Umbral estricto
    python scripts/retrain.py --todas --threshold 0.02        # Reentrenar todas
    python scripts/retrain.py --lista                        # Ver ligas
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from football_leagues.config import get_config, listar_ligas
from football_leagues.model import Reentrenador


def main():
    parser = argparse.ArgumentParser(
        description='Reentrena modelos XGBoost con datos frescos'
    )
    parser.add_argument(
        '--liga',
        action='append',
        help='Liga a reentrenar. Puede usarse multiples veces.'
    )
    parser.add_argument(
        '--todas',
        action='store_true',
        help='Reentrenar todas las ligas'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.02,
        help='Umbral de degradacion permitida (default 0.02 = 2%)'
    )
    parser.add_argument(
        '--lista',
        action='store_true',
        help='Mostrar ligas disponibles'
    )

    args = parser.parse_args()

    if args.lista:
        print("Ligas disponibles:")
        for liga in listar_ligas():
            print(f"  - {liga}")
        return 0

    if not args.liga and not args.todas:
        print("Error: Especifica --liga, --todas o --lista")
        parser.print_help()
        return 1

    ligas_a_reentrenar = listar_ligas() if args.todas else list(set(args.liga))

    resultados = []

    for liga in ligas_a_reentrenar:
        try:
            config = get_config(liga)
            reentrenador = Reentrenador(config)
            actualizado = reentrenador.retrain(umbral_degradacion=args.threshold)
            resultados.append((liga, 'actualizado' if actualizado else 'mantenido', True))
        except FileNotFoundError:
            print(f"\n✗ {liga}: Modelo no existe. Ejecuta primero 'python scripts/train.py --liga {liga}'")
            resultados.append((liga, 'error', False))
        except ValueError as e:
            print(f"\n✗ Error: {e}")
            resultados.append((liga, 'error', False))
        except Exception as e:
            print(f"\n✗ Error inesperado en {liga}: {e}")
            resultados.append((liga, 'error', False))

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN REENTRENAMIENTO")
    print(f"{'='*60}")
    for liga, estado, exito in resultados:
        icon = '✓' if exito else '✗'
        print(f"{icon} {liga:<20} {estado}")

    return 0 if all(r[2] for r in resultados) else 1


if __name__ == '__main__':
    sys.exit(main())
