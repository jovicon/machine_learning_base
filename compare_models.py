import joblib
import os
from pathlib import Path

# Rutas de los modelos
real_models_path = Path("/Users/josecontreras/Desktop/areas/personal/code/machine_learning_base/real_models")
pipeline_models_path = Path("/Users/josecontreras/Desktop/areas/personal/code/machine_learning_base/pipelines/football_leagues/models")

leagues = ["premier_league", "spanish_league", "italian_league", "german_league"]

print("=" * 100)
print("COMPARACIÓN DE MODELOS: real_models/ vs pipelines/football_leagues/")
print("=" * 100)

comparison_results = []

for league in leagues:
    print(f"\n{'='*100}")
    print(f"Liga: {league.upper().replace('_', ' ')}")
    print(f"{'='*100}")

    # Rutas de los modelos actuales (no versiones)
    real_model_file = real_models_path / league / f"modelo_{league.replace('_league', '')}.pkl"
    pipeline_model_file = pipeline_models_path / league / f"modelo_{league}.pkl"

    # Alternativamente, buscar el archivo correcto
    if not real_model_file.exists():
        real_files = list((real_models_path / league).glob("modelo_*.pkl"))
        real_model_file = [f for f in real_files if "versiones" not in str(f)]
        real_model_file = real_model_file[0] if real_model_file else None

    if not pipeline_model_file.exists():
        pipeline_files = list((pipeline_models_path / league).glob("modelo_*.pkl"))
        pipeline_model_file = [f for f in pipeline_files if "versiones" not in str(f)]
        pipeline_model_file = pipeline_model_file[0] if pipeline_model_file else None

    real_exists = real_model_file and real_model_file.exists() if isinstance(real_model_file, Path) else False
    pipeline_exists = pipeline_model_file and pipeline_model_file.exists() if isinstance(pipeline_model_file, Path) else False

    print(f"real_models: {real_model_file if real_exists else 'NO ENCONTRADO'}")
    print(f"pipelines:   {pipeline_model_file if pipeline_exists else 'NO ENCONTRADO'}")

    if real_exists and pipeline_exists:
        try:
            real_model = joblib.load(real_model_file)
            pipeline_model = joblib.load(pipeline_model_file)

            # Extraer métricas
            real_accuracy = real_model.get("accuracy", "N/A")
            real_accuracy_alta = real_model.get("accuracy_alta_confianza", "N/A")
            real_date = real_model.get("fecha_entrenamiento", "N/A")

            pipeline_accuracy = pipeline_model.get("accuracy", "N/A")
            pipeline_accuracy_alta = pipeline_model.get("accuracy_alta_confianza", "N/A")
            pipeline_date = pipeline_model.get("fecha_entrenamiento", "N/A")

            print(f"\nreal_models/:")
            print(f"  Accuracy: {real_accuracy}")
            print(f"  Accuracy (alta confianza): {real_accuracy_alta}")
            print(f"  Fecha entrenamiento: {real_date}")

            print(f"\npipelines/:")
            print(f"  Accuracy: {pipeline_accuracy}")
            print(f"  Accuracy (alta confianza): {pipeline_accuracy_alta}")
            print(f"  Fecha entrenamiento: {pipeline_date}")

            # Comparación
            if isinstance(real_accuracy, (int, float)) and isinstance(pipeline_accuracy, (int, float)):
                diff = pipeline_accuracy - real_accuracy
                status = "✓ CUMPLE" if abs(diff) <= 0.02 else ("⚠ MEJOR" if diff > 0 else "✗ PEOR")
                print(f"\n{status} - Diferencia: {diff:+.4f} ({diff*100:+.2f}%)")
                comparison_results.append({
                    "league": league,
                    "real_accuracy": real_accuracy,
                    "pipeline_accuracy": pipeline_accuracy,
                    "difference": diff,
                    "status": status
                })

        except Exception as e:
            print(f"ERROR al cargar modelos: {e}")
    else:
        print("⚠ No se pueden comparar (uno o ambos modelos no existen)")

print(f"\n{'='*100}")
print("RESUMEN DE COMPARACIÓN")
print(f"{'='*100}\n")
print(f"{'Liga':<20} {'real_models':<15} {'pipelines':<15} {'Diferencia':<15} {'Estado':<15}")
print("-" * 80)
for result in comparison_results:
    print(f"{result['league']:<20} {result['real_accuracy']:<15.4f} {result['pipeline_accuracy']:<15.4f} {result['difference']:+<15.4f} {result['status']:<15}")
