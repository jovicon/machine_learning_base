#!/usr/bin/env python3
"""
Script para verificar que el pipeline se importa correctamente.
Ejecuta antes de correr scripts reales.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    print("Verificando imports...")

    from football_leagues.config import get_config, listar_ligas, LIGAS_CONFIG
    print("✓ config.py")

    from football_leagues.data import DataLoader
    print("✓ data.py")

    from football_leagues.features import FeatureBuilder
    print("✓ features.py")

    from football_leagues.model import Entrenador, Reentrenador
    print("✓ model.py")

    print("\nVerificando configuracion...")
    print(f"Ligas disponibles: {', '.join(listar_ligas())}")

    for liga in listar_ligas():
        config = get_config(liga)
        print(f"  ✓ {liga}: {config['nombre_completo']}")

    print("\n✅ Todos los imports OK - Pipeline listo para usar")
    sys.exit(0)

except ImportError as e:
    print(f"\n❌ Error de import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
