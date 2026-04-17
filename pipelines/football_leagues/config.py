"""
Configuracion de ligas de futbol
Mapeo codigo liga -> parametros de entrenamiento
"""

LIGAS_CONFIG = {
    'german': {
        'codigo': 'D1',
        'nombre': 'german_league',
        'nombre_completo': 'Bundesliga',
        'url_base': 'https://www.football-data.co.uk/mmz4281/{}/D1.csv',
        'temporadas': ['2021', '2122', '2223', '2324', '2425', '2526'],
        'temporada_test': '2526',
        'baseline_local': 0.44,
        'ruta_modelo': './models/german_league',
    },
    'premier': {
        'codigo': 'E0',
        'nombre': 'premier_league',
        'nombre_completo': 'Premier League',
        'url_base': 'https://www.football-data.co.uk/mmz4281/{}/E0.csv',
        'temporadas': ['2021', '2122', '2223', '2324', '2425', '2526'],
        'temporada_test': '2526',
        'baseline_local': 0.46,
        'ruta_modelo': './models/premier_league',
    },
    'spanish': {
        'codigo': 'SP1',
        'nombre': 'spanish_league',
        'nombre_completo': 'La Liga',
        'url_base': 'https://www.football-data.co.uk/mmz4281/{}/SP1.csv',
        'temporadas': ['2021', '2122', '2223', '2324', '2425', '2526'],
        'temporada_test': '2526',
        'baseline_local': 0.45,
        'ruta_modelo': './models/spanish_league',
    },
    'italian': {
        'codigo': 'I1',
        'nombre': 'italian_league',
        'nombre_completo': 'Serie A',
        'url_base': 'https://www.football-data.co.uk/mmz4281/{}/I1.csv',
        'temporadas': ['2021', '2122', '2223', '2324', '2425', '2526'],
        'temporada_test': '2526',
        'baseline_local': 0.44,
        'ruta_modelo': './models/italian_league',
    },
}


def get_config(liga_codigo):
    """
    Obtiene configuracion de una liga.

    Args:
        liga_codigo: 'german', 'premier', 'spanish', 'italian'

    Returns:
        dict con configuracion

    Raises:
        ValueError si la liga no existe
    """
    if liga_codigo not in LIGAS_CONFIG:
        raise ValueError(
            f"Liga '{liga_codigo}' no encontrada. "
            f"Opciones: {', '.join(LIGAS_CONFIG.keys())}"
        )
    return LIGAS_CONFIG[liga_codigo]


def listar_ligas():
    """Retorna lista de ligas disponibles."""
    return list(LIGAS_CONFIG.keys())
