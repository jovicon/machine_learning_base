"""
Construccion de features para predicciones de futbol
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class FeatureBuilder:
    """Construye features a partir de datos de partidos historicos."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame con todos los datos historicos
        """
        self.df = df

    def get_team_stats_v2(self, team: str, date: pd.Timestamp, n_recent: int = 5) -> Optional[Dict]:
        """
        Obtiene estadisticas de un equipo hasta una fecha dada.

        Calcula metricas sobre ultimos n partidos (recent) y temporada actual (season).

        Args:
            team: nombre del equipo
            date: fecha limite (anterior a esta fecha)
            n_recent: cantidad de partidos recientes a considerar

        Returns:
            Dict con features del equipo o None si historia insuficiente
        """
        home_all = self.df[(self.df['HomeTeam'] == team) & (self.df['Date'] < date)]
        away_all = self.df[(self.df['AwayTeam'] == team) & (self.df['Date'] < date)]
        all_games = pd.concat([home_all, away_all]).sort_values('Date')

        if len(all_games) < 3:
            return None

        current_season = self.df[self.df['Date'] < date]['season'].iloc[-1]
        season_games = pd.concat([
            home_all[home_all['season'] == current_season],
            away_all[away_all['season'] == current_season]
        ]).sort_values('Date')

        recent = all_games.tail(n_recent)

        def calc_stats(games, team):
            if len(games) == 0:
                return None
            gf = gc = sot_f = sot_c = wins = draws = home_wins = home_games = 0

            for _, r in games.iterrows():
                ih = r['HomeTeam'] == team
                gf += r['FTHG'] if ih else r['FTAG']
                gc += r['FTAG'] if ih else r['FTHG']
                sot_f += r['HST'] if ih else r['AST']
                sot_c += r['AST'] if ih else r['HST']

                won = (ih and r['FTR'] == 'H') or (not ih and r['FTR'] == 'A')
                drew = r['FTR'] == 'D'
                if won:
                    wins += 1
                if drew:
                    draws += 1

                if ih:
                    home_games += 1
                    if r['FTR'] == 'H':
                        home_wins += 1

            m = len(games)
            return {
                'gf_pg': gf / m,
                'gc_pg': gc / m,
                'dif_goles': (gf - gc) / m,
                'sot_f_pg': sot_f / m,
                'sot_c_pg': sot_c / m,
                'win_rate': wins / m,
                'draw_rate': draws / m,
                'home_wr': home_wins / home_games if home_games > 0 else wins / m
            }

        rs = calc_stats(recent, team)
        ss = calc_stats(season_games, team) if len(season_games) >= 3 else rs

        if rs is None:
            return None

        return {f'recent_{k}': rs[k] for k in rs} | \
               {f'season_{k}': ss[k] if ss else rs[k] for k in rs}

    def get_h2h_stats(self, home_team: str, away_team: str, date: pd.Timestamp, n: int = 10) -> Dict:
        """
        Obtiene estadisticas de enfrentamientos directos (head-to-head).

        Args:
            home_team: equipo local
            away_team: equipo visitante
            date: fecha limite
            n: cantidad de enfrentamientos previos a considerar

        Returns:
            Dict con tasas de victoria/empate por perspectiva
        """
        h2h = self.df[
            ((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) |
            ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))
        ][lambda x: x['Date'] < date].tail(n)

        if len(h2h) < 3:
            return {'h2h_home_wr': 0.33, 'h2h_draw_rate': 0.33, 'h2h_away_wr': 0.33}

        hw = ((h2h['HomeTeam'] == home_team) & (h2h['FTR'] == 'H')).sum() + \
             ((h2h['AwayTeam'] == home_team) & (h2h['FTR'] == 'A')).sum()
        dr = (h2h['FTR'] == 'D').sum()
        m = len(h2h)

        return {
            'h2h_home_wr': hw / m,
            'h2h_draw_rate': dr / m,
            'h2h_away_wr': (m - hw - dr) / m
        }

    def get_tabla_posicion(self, team: str, date: pd.Timestamp) -> Dict:
        """
        Obtiene posicion en tabla y estadisticas de equipo en temporada actual.

        Args:
            team: nombre del equipo
            date: fecha limite

        Returns:
            Dict con posicion, puntos por partido, diferencia de goles, percentil
        """
        season = self.df[self.df['Date'] < date]['season'].iloc[-1]
        ps = self.df[(self.df['season'] == season) & (self.df['Date'] < date)]

        tabla = []
        for eq in pd.concat([ps['HomeTeam'], ps['AwayTeam']]).unique():
            hp = ps[ps['HomeTeam'] == eq]
            ap = ps[ps['AwayTeam'] == eq]
            pts = (hp['FTR'] == 'H').sum() * 3 + (hp['FTR'] == 'D').sum() + \
                  (ap['FTR'] == 'A').sum() * 3 + (ap['FTR'] == 'D').sum()
            tabla.append({
                'equipo': eq,
                'pts': pts,
                'gf': hp['FTHG'].sum() + ap['FTAG'].sum(),
                'gc': hp['FTAG'].sum() + ap['FTHG'].sum(),
                'pj': len(hp) + len(ap)
            })

        tdf = pd.DataFrame(tabla).sort_values('pts', ascending=False).reset_index(drop=True)
        tdf['pos'] = tdf.index + 1

        f = tdf[tdf['equipo'] == team]
        if len(f) == 0:
            return {
                'posicion': 10,
                'pts_pg': 1.0,
                'dif_goles_szn': 0,
                'pct_posicion': 0.5
            }

        f = f.iloc[0]
        pj = max(f['pj'], 1)
        return {
            'posicion': f['pos'],
            'pts_pg': f['pts'] / pj,
            'dif_goles_szn': (f['gf'] - f['gc']) / pj,
            'pct_posicion': 1 - (f['pos'] / len(tdf))
        }

    def construir_dataset(self) -> pd.DataFrame:
        """
        Construye dataset completo con features para cada partido.

        Tarda ~3-5 minutos en ejecutarse.

        Returns:
            DataFrame con features para cada partido
        """
        rows = []
        total = len(self.df)

        for idx, (_, p) in enumerate(self.df.iterrows()):
            if (idx + 1) % 200 == 0:
                print(f"  Procesado {idx + 1}/{total} partidos...")

            ht, at, dt, sn = p['HomeTeam'], p['AwayTeam'], p['Date'], p['season']

            hs = self.get_team_stats_v2(ht, dt)
            as_ = self.get_team_stats_v2(at, dt)

            if hs is None or as_ is None:
                continue

            h2h = self.get_h2h_stats(ht, at, dt)
            ht2 = self.get_tabla_posicion(ht, dt)
            at2 = self.get_tabla_posicion(at, dt)

            row = {
                'date': dt,
                'season': sn,
                'home_team': ht,
                'away_team': at,
                'resultado': p['FTR']
            }

            for k, v in hs.items():
                row[f'h_{k}'] = v
            for k, v in as_.items():
                row[f'a_{k}'] = v
            for k, v in h2h.items():
                row[k] = v
            for k, v in ht2.items():
                row[f'h_tabla_{k}'] = v
            for k, v in at2.items():
                row[f'a_tabla_{k}'] = v

            row['dif_recent_wr'] = hs['recent_win_rate'] - as_['recent_win_rate']
            row['dif_season_wr'] = hs['season_win_rate'] - as_['season_win_rate']
            row['dif_recent_gf'] = hs['recent_gf_pg'] - as_['recent_gf_pg']
            row['dif_recent_gc'] = hs['recent_gc_pg'] - as_['recent_gc_pg']
            row['dif_recent_dif'] = hs['recent_dif_goles'] - as_['recent_dif_goles']
            row['dif_season_dif'] = hs['season_dif_goles'] - as_['season_dif_goles']
            row['home_advantage'] = hs['recent_home_wr'] - as_['recent_win_rate']
            row['dif_pts_pg'] = ht2['pts_pg'] - at2['pts_pg']
            row['dif_posicion'] = at2['posicion'] - ht2['posicion']

            rows.append(row)

        df_features = pd.DataFrame(rows)
        print(f"\nDataset construido:")
        print(f"  Partidos: {len(df_features)}")
        print(f"  Features: {len(df_features.columns) - 5}")

        return df_features
