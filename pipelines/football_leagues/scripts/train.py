#!/usr/bin/env python3
"""
Script para entrenar modelo desde cero para una liga

Uso:
    python scripts/train.py --liga german
    python scripts/train.py --liga premier --liga spanish
    python scripts/train.py --lista              # Mostrar ligas disponibles
"""

import sys
import argparse
from pathlib import Path

# Agregar parent dir al path para importar modulos
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from football_leagues.config import get_config, listar_ligas
from football_leagues.model import Entrenador


def main():
    parser = argparse.ArgumentParser(
        description='Entrena modelos XGBoost para ligas de futbol'
    )
    parser.add_argument(
        '--liga',
        action='append',
        help='Liga a entrenar (german, premier, spanish, italian). Puede usarse multiples veces.'
    )
    parser.add_argument(
        '--todas',
        action='store_true',
        help='Entrenar todas las ligas por separado'
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

    if args.todas:
        ligas_a_entrenar = listar_ligas()
    elif args.liga:
        ligas_a_entrenar = list(set(args.liga))  # Remover duplicados
    else:
        print("Error: Especifica --liga, --todas o --lista")
        parser.print_help()
        return 1
    exitos = []

    for liga in ligas_a_entrenar:
        try:
            config = get_config(liga)
            entrenador = Entrenador(config)
            metricas = entrenador.entrenar()
            exitos.append((liga, True, metricas))
        except ValueError as e:
            print(f"\n✗ Error: {e}")
            exitos.append((liga, False, None))
        except Exception as e:
            print(f"\n✗ Error inesperado en {liga}: {e}")
            exitos.append((liga, False, None))

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    for liga, exito, metricas in exitos:
        if exito:
            print(f"✓ {liga:<20} acc={metricas['accuracy']:.1%} (alta: {metricas['accuracy_alta_confianza']:.1%})")
        else:
            print(f"✗ {liga:<20} FALLO")

    return 0 if all(e[1] for e in exitos) else 1


if __name__ == '__main__':
    sys.exit(main())
