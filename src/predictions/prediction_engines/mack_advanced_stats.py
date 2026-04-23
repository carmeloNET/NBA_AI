"""
Mack-based predictor using statistical formulas.
Based on Andrew Mack's Bayesian Sports Models approach.
"""

import sqlite3
from typing import Dict, List, Optional

import numpy as np

from src.config import config

DB_PATH = config["database"]["path"]

NBA_MINUTES_PER_GAME = 48
DEFAULT_LEAGUE_AVG_PACE = 100.0
DEFAULT_LEAGUE_AVG_ORtg = 112.0


def calculate_possessions(fga: int, oreb: int, tov: int, fta: int) -> float:
    """
    Calculate possessions using Dean Oliver's formula.
    Possessions = FGA - ORB + TOV + 0.44 * FTA
    """
    return fga - oreb + tov + (0.44 * fta)


def calculate_pace(possessions: float, minutes: int = NBA_MINUTES_PER_GAME) -> float:
    """
    Calculate Pace (possessions per 48 minutes).
    """
    if minutes == 0:
        return 0
    return (possessions / minutes) * NBA_MINUTES_PER_GAME


def calculate_ortg(points: int, possessions: float) -> float:
    """
    Calculate Offensive Rating (points per 100 possessions).
    """
    if possessions == 0:
        return 0
    return (points / possessions) * 100


def calculate_drtg(points_allowed: int, possessions: float) -> float:
    """
    Calculate Defensive Rating (points allowed per 100 possessions).
    """
    if possessions == 0:
        return 0
    return (points_allowed / possessions) * 100


def calculate_efg_pct(fgm: int, fg3m: int, fga: int) -> float:
    """
    Calculate Effective Field Goal Percentage.
    eFG% = (FGM + 0.5 * FG3M) / FGA
    """
    if fga == 0:
        return 0
    return (fgm + 0.5 * fg3m) / fga


def calculate_tov_pct(tov: int, fga: int, fta: int) -> float:
    """
    Calculate Turnover Percentage.
    TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    """
    denominator = fga + (0.44 * fta) + tov
    if denominator == 0:
        return 0
    return tov / denominator


def calculate_orb_pct(orb: int, drb_opp: int) -> float:
    """
    Calculate Offensive Rebound Percentage.
    ORB% = ORB / (ORB + DRB_opp)
    """
    total_reb = orb + drb_opp
    if total_reb == 0:
        return 0
    return orb / total_reb


def calculate_ft_rate(fta: int, fga: int) -> float:
    """
    Calculate Free Throw Rate.
    FT Rate = FTA / FGA
    """
    if fga == 0:
        return 0
    return fta / fga


def get_team_game_stats(team_id: str, game_id: str, conn: sqlite3.Connection) -> Optional[Dict]:
    """Get all stats for a team in a specific game."""
    cursor = conn.cursor()
    cursor.execute(
        """SELECT team_id, game_id, pts, pts_allowed, reb, ast, stl, blk, tov, pf,
                  fga, fgm, fg_pct, fg3a, fg3m, fg3_pct, fta, ftm, ft_pct, plus_minus
           FROM TeamBox WHERE team_id = ? AND game_id = ?""",
        (team_id, game_id),
    )
    row = cursor.fetchone()
    if not row:
        return None
    
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def get_team_stats_from_games(
    team_id: str, 
    game_ids: List[str], 
    conn: sqlite3.Connection
) -> List[Dict]:
    """Get stats for a team across multiple games."""
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(game_ids))
    cursor.execute(
        f"""SELECT team_id, game_id, pts, pts_allowed, reb, ast, stl, blk, tov, pf,
                   fga, fgm, fg_pct, fg3a, fg3m, fg3_pct, fta, ftm, ft_pct, plus_minus
            FROM TeamBox 
            WHERE team_id = ? AND game_id IN ({placeholders})
            ORDER BY game_id DESC""",
        [team_id] + game_ids
    )
    
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def calculate_team_averages(stats_list: List[Dict]) -> Dict:
    """Calculate average advanced stats from a list of game stats."""
    if not stats_list:
        return {}
    
    n = len(stats_list)
    
    total_pts = sum(s['pts'] for s in stats_list)
    total_pts_allowed = sum(s['pts_allowed'] for s in stats_list)
    total_tov = sum(s['tov'] for s in stats_list)
    total_fga = sum(s['fga'] for s in stats_list)
    total_fta = sum(s['fta'] for s in stats_list)
    total_fgm = sum(s['fgm'] for s in stats_list)
    total_fg3m = sum(s['fg3m'] for s in stats_list)
    
    total_orb = 0
    total_drb_opp = 0
    for s in stats_list:
        total_orb += s.get('reb', 0) // 2
    
    total_poss = 0
    for s in stats_list:
        orb = s.get('reb', 0) // 2
        poss = calculate_possessions(s['fga'], orb, s['tov'], s['fta'])
        total_poss += poss
    
    avg_pts = total_pts / n
    avg_pts_allowed = total_pts_allowed / n
    avg_poss = total_poss / n
    avg_pace = calculate_pace(avg_poss)
    avg_ortg = calculate_ortg(avg_pts, avg_poss)
    avg_drtg = calculate_drtg(avg_pts_allowed, avg_poss)
    
    avg_efg = calculate_efg_pct(total_fgm, total_fg3m, total_fga)
    avg_tov_pct = calculate_tov_pct(total_tov, total_fga, total_fta)
    avg_orb_pct = calculate_orb_pct(total_orb, total_drb_opp)
    avg_ft_rate = calculate_ft_rate(total_fta, total_fga)
    
    return {
        'games': n,
        'pts': avg_pts,
        'pts_allowed': avg_pts_allowed,
        'possessions': avg_poss,
        'pace': avg_pace,
        'ortg': avg_ortg,
        'drtg': avg_drtg,
        'efg_pct': avg_efg,
        'tov_pct': avg_tov_pct,
        'orb_pct': avg_orb_pct,
        'ft_rate': avg_ft_rate,
    }


def get_team_recent_stats(
    team_id: str,
    game_ids: List[str],
    window: int = 10,
    conn: sqlite3.Connection = None
) -> Dict:
    """Get rolling average stats for a team over recent games."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    
    recent_game_ids = game_ids[:window]
    stats_list = get_team_stats_from_games(team_id, recent_game_ids, conn)
    
    return calculate_team_averages(stats_list)


def get_season_averages(
    team_id: str,
    season: str,
    conn: sqlite3.Connection = None
) -> Dict:
    """Get full season averages for a team."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    
    cursor = conn.cursor()
    cursor.execute(
        """SELECT game_id FROM Games 
           WHERE (home_team = ? OR away_team = ?) 
           AND season = ? AND status = 3
           ORDER BY date_time_utc DESC""",
        (team_id, team_id, season)
    )
    game_ids = [row[0] for row in cursor.fetchall()]
    
    if not game_ids:
        return {}
    
    return get_team_recent_stats(team_id, game_ids, window=len(game_ids), conn=conn)
