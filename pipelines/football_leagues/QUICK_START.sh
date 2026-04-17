#!/bin/bash
# QUICK START - Football Leagues ML Pipeline
# Ejecuta este script para setup inicial

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════"
echo "  FOOTBALL LEAGUES ML PIPELINE - QUICK START"
echo "════════════════════════════════════════════════════════════"
echo ""

# Paso 1: Verificar Python y venv
echo "1️⃣  Verificando entorno..."
if [ ! -d ".venv" ]; then
    echo "❌ .venv no encontrado"
    echo "   Ejecuta 'make setup' en la raiz del proyecto"
    exit 1
fi

source ../.venv/bin/activate
echo "✓ Virtual env activado"

# Paso 2: Verificar imports
echo ""
echo "2️⃣  Verificando imports..."
python test_import.py
if [ $? -ne 0 ]; then
    echo "❌ Error en imports"
    exit 1
fi

# Paso 3: Ver ligas disponibles
echo ""
echo "3️⃣  Ligas disponibles:"
python scripts/train.py --lista

# Paso 4: Ofrecer opciones
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PROXIMOS PASOS"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Entrenar Bundesliga:"
echo "  python scripts/train.py --liga german"
echo ""
echo "Ver status:"
echo "  python scripts/status.py --detalle"
echo ""
echo "Leer documentacion:"
echo "  cat README.md"
echo "  cat EJEMPLO_USO.md"
echo ""
