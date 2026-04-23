#!/usr/bin/env python3
"""
CLI para predecir jornadas completas.
Uso: python predict.py --liga premier --jornada 33 --fecha 2026-04-18 [--partidos JSON]
"""

import argparse
import json
from pathlib import Path
from predict_matchday import predecir_jornada, guardar_predicciones


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predice resultados de jornadas completas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Todas las ligas desde archivo multi-liga
  python predict.py --partidos ejemplo_partidos.json

  # Liga específica
  python predict.py --liga premier --jornada 33 --fecha 2026-04-18 --partidos partidos.json

  # Desde argumentos (lista simple)
  python predict.py --liga german --jornada 30 --fecha 2026-04-18 \\
    --partidos '[["Leipzig", "Gladbach"], ["Stuttgart", "Hamburg"]]'

  # Ligas disponibles: german, premier, spanish, italian
        """
    )

    parser.add_argument('--liga', choices=['german', 'premier', 'spanish', 'italian'],
                        help='Liga específica (si no se especifica, usa todas las del archivo)')
    parser.add_argument('--jornada', type=int, help='Número de jornada (requerido si --partidos es lista simple)')
    parser.add_argument('--fecha', help='Fecha YYYY-MM-DD (requerido si --partidos es lista simple)')
    parser.add_argument('--partidos', required=True, help='JSON con partidos: archivo multi-liga o lista simple')
    parser.add_argument('--append', action='store_true', help='Append a archivo existente')

    return parser.parse_args()


def load_partidos(partidos_arg: str):
    """
    Carga partidos desde JSON string o archivo.
    Retorna dict (multi-liga) o list (simple).
    """
    # Intentar como archivo
    try:
        path = Path(partidos_arg)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    # Intentar como JSON string
    try:
        return json.loads(partidos_arg)
    except json.JSONDecodeError:
        raise ValueError(f"No se pudo parsear partidos: {partidos_arg}")


def procesar_liga(liga_codigo, config, append):
    """Procesa predicciones para una liga."""
    partidos = config['partidos']
    fecha = config['fecha']
    jornada = config['jornada']

    # Convertir a tuplas si es necesario
    if partidos and isinstance(partidos[0], list):
        partidos = [tuple(p) for p in partidos]

    print(f"\n📊 Predicción jornada {jornada} - {liga_codigo.upper()}")
    print(f"Fecha: {fecha}")
    print(f"Partidos: {len(partidos)}\n")

    predicciones, resumen = predecir_jornada(
        liga_codigo=liga_codigo,
        partidos=partidos,
        fecha_jornada=fecha,
        jornada=jornada
    )

    # Mostrar resultados
    print("Predicciones:")
    print("-" * 80)
    for pred in predicciones:
        if 'error' in pred:
            print(f"  {pred['home']:20} vs {pred['away']:20} ⚠️  {pred['error']}")
        else:
            marker = "✓ USAR" if pred['confianza'] > 0.55 else "↙ bajo"
            print(f"  {pred['home']:20} vs {pred['away']:20} → {pred['prediccion']:<2} ({pred['confianza']:.1%}) {marker}")

    print("\n" + "-" * 80)
    print(f"Resumen: {resumen['partidos_predichos']} partidos")
    print(f"  Confianza promedio: {resumen['confianza_promedio']:.1%}")
    print(f"  Alta confianza (>55%): {resumen['predicciones_alta_confianza']}")

    guardar_predicciones(liga_codigo, predicciones, resumen, append=append)
    return resumen['partidos_predichos']


def parse_data(data, args):
    """Detecta formato y retorna dict de ligas a procesar."""
    if isinstance(data, dict):
        # Formato multi-liga
        ligas = {k: v for k, v in data.items() if args.liga is None or k == args.liga}
        if not ligas:
            raise ValueError(f"Liga '{args.liga}' no encontrada en archivo")
        return ligas
    elif isinstance(data, list):
        # Formato simple: lista de tuplas/listas
        if not args.liga or not args.jornada or not args.fecha:
            raise ValueError("Con formato de lista simple se requiere --liga, --jornada y --fecha")
        return {
            args.liga: {
                'fecha': args.fecha,
                'jornada': args.jornada,
                'partidos': data
            }
        }
    else:
        raise ValueError("Formato de partidos inválido (debe ser dict o list)")


def main():
    args = parse_args()

    try:
        data = load_partidos(args.partidos)
        ligas_a_procesar = parse_data(data, args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    total_predicciones = 0
    for liga_codigo, config in ligas_a_procesar.items():
        try:
            count = procesar_liga(liga_codigo, config, args.append)
            total_predicciones += count
        except Exception as e:
            print(f"Error durante predicción de {liga_codigo}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n✅ Total predicciones guardadas: {total_predicciones}")
    return 0


if __name__ == '__main__':
    exit(main())
