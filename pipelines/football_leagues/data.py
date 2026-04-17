"""
Modulo de carga y preparacion de datos
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class DataLoader:
    """Carga datos de football-data.co.uk para una liga."""

    def __init__(self, config: Dict):
        """
        Args:
            config: diccionario con 'url_base', 'temporadas', etc.
        """
        self.config = config
        self.url_base = config['url_base']
        self.temporadas = config['temporadas']

    def cargar_datos_crudos(self) -> pd.DataFrame:
        """
        Carga datos de todas las temporadas desde football-data.co.uk

        Returns:
            DataFrame con columnas: season, Date, HomeTeam, AwayTeam,
                                    FTHG, FTAG, FTR, HS, AS, HST, AST, HC, AC
        """
        dfs = []
        for t in self.temporadas:
            try:
                df_temp = pd.read_csv(self.url_base.format(t))
                df_temp['season'] = t
                dfs.append(df_temp)
                print(f"✓ Temporada {t}: {len(df_temp)} partidos")
            except Exception as e:
                print(f"✗ Error cargando temporada {t}: {e}")
                continue

        if not dfs:
            raise RuntimeError("No se pudieron cargar datos de ninguna temporada")

        df_raw = pd.concat(dfs, ignore_index=True)

        # Seleccionar y limpiar columnas
        cols = ['season', 'Date', 'HomeTeam', 'AwayTeam',
                'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']
        df = df_raw[cols].dropna(subset=['FTR'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)

        print(f"\nTotal partidos cargados: {len(df)}")
        print(f"Rango de fechas: {df['Date'].min().date()} -> {df['Date'].max().date()}")
        print(f"\nDistribucion resultados:")
        print(df['FTR'].value_counts())

        return df

    def obtener_equipos(self, df: pd.DataFrame) -> List[str]:
        """Retorna lista de equipos en el dataset."""
        return sorted(df['HomeTeam'].unique())

    def dividir_train_test(self, df: pd.DataFrame, temporada_test: str):
        """
        Divide datos en train y test por temporada.

        Args:
            df: DataFrame con datos
            temporada_test: codigo temporada para test (ej: '2526')

        Returns:
            (train_df, test_df)
        """
        train = df[df['season'] != temporada_test].copy()
        test = df[df['season'] == temporada_test].copy()

        print(f"\nTrain: {len(train)} partidos")
        print(f"Test:  {len(test)} partidos (temporada {temporada_test})")

        return train, test
