"""
Live Predictor - Tiempo real durante el partido.
Usa Teorema de Bayes para actualizar predicciones.
"""

import sqlite3
from typing import Dict, Optional, Tuple

from scipy import stats

from src.config import config
from src.predictions.prediction_engines.mack_advanced_stats import (
    get_team_recent_stats,
)
from src.predictions.prediction_engines.mack_predictor import (
    HOME_COURT_ADVANTAGE,
    DEFAULT_LEAGUE_AVG_ORtg,
    calculate_expected_score,
    remove_vig,
    add_vig_to_prob,
    to_american_odds,
)

DB_PATH = config["database"]["path"]


def get_live_game_data(game_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Obtiene datos en vivo del partido desde la base de datos.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT home_team, away_team, home_score, away_score, 
               quarter, game_time_remaining, status
        FROM Games 
        WHERE game_id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    
    return {
        'game_id': game_id,
        'home_team': row[0],
        'away_team': row[1],
        'home_score': row[2] or 0,
        'away_score': row[3] or 0,
        'quarter': row[4] or 0,
        'time_remaining': row[5] or '12:00',
        'status': row[6],
    }


def estimate_live_possessions(
    home_fga: int,
    home_orb: int,
    home_tov: int,
    home_fta: int,
    away_fga: int,
    away_orb: int,
    away_tov: int,
    away_fta: int,
) -> Tuple[float, float]:
    """
    Estima posesiones en vivo usando fórmula de Dean Oliver.
    Poss = FGA - ORB + TOV + 0.44 * FTA
    """
    home_poss = home_fga - home_orb + home_tov + (0.44 * home_fta)
    away_poss = away_fga - away_orb + away_tov + (0.44 * away_fta)
    return home_poss, away_poss


def calculate_live_pace(
    home_poss: float,
    away_poss: float,
    minutes_elapsed: float,
    total_minutes: float = 48.0
) -> float:
    """
    Calcula pace proyectado a 48 minutos.
    """
    if minutes_elapsed <= 0:
        return 100.0
    
    avg_poss_per_min = (home_poss + away_poss) / 2
    pace = (avg_poss_per_min / minutes_elapsed) * total_minutes
    return round(pace, 1)


def calculate_live_ortg(points: int, possessions: float) -> float:
    """
    Calcula ORtg en vivo.
    """
    if possessions <= 0:
        return 100.0
    return round((points / possessions) * 100, 1)


def live_prediction_bayes(
    pre_game_home: float,
    pre_game_away: float,
    current_home: float,
    current_away: float,
    minutes_elapsed: float,
    total_minutes: float = 48.0
) -> Tuple[float, float]:
    """
    Actualiza predicción usando Teorema de Bayes.
    
    A medida que avanza el partido, el score actual pesa más.
    
    blend_factor = (time_remaining / total_time)²
    """
    if minutes_elapsed <= 0:
        return pre_game_home, pre_game_away
    
    time_remaining = total_minutes - minutes_elapsed
    
    blend_factor = (time_remaining / total_minutes) ** 2
    
    blended_home = (
        pre_game_home * blend_factor + 
        current_home * (1 - blend_factor)
    )
    blended_away = (
        pre_game_away * blend_factor + 
        current_away * (1 - blend_factor)
    )
    
    return round(blended_home, 1), round(blended_away, 1)


def estimate_final_score_monte_carlo(
    home_score: float,
    away_score: float,
    minutes_remaining: float,
    pace: float,
    simulations: int = 10000
) -> Dict:
    """
    Estima score final usando simulación.
    Asume distribución normal para los puntos restantes.
    """
    if minutes_remaining <= 0:
        return {
            'estimated_home_final': home_score,
            'estimated_away_final': away_score,
            'home_confidence_low': home_score,
            'home_confidence_high': home_score,
            'away_confidence_low': away_score,
            'away_confidence_high': away_score,
        }
    
    points_per_minute_home = home_score / (48 - minutes_remaining)
    points_per_minute_away = away_score / (48 - minutes_remaining)
    
    home_remaining = points_per_minute_home * minutes_remaining
    away_remaining = points_per_minute_away * minutes_remaining
    
    std_dev_per_min = 0.8
    
    home_final = home_score + home_remaining
    away_final = away_score + away_remaining
    
    home_std = std_dev_per_min * minutes_remaining
    away_std = std_dev_per_min * minutes_remaining
    
    return {
        'estimated_home_final': round(home_final, 1),
        'estimated_away_final': round(away_final, 1),
        'home_confidence_low': round(home_final - 1.96 * home_std, 1),
        'home_confidence_high': round(home_final + 1.96 * home_std, 1),
        'away_confidence_low': round(away_final - 1.96 * away_std, 1),
        'away_confidence_high': round(away_final + 1.96 * away_std, 1),
    }


def generate_live_odds(
    home_team_id: str,
    away_team_id: str,
    game_id: str,
    current_home_score: int,
    current_away_score: int,
    minutes_elapsed: float = 0.0,
    home_fga: int = 0,
    home_orb: int = 0,
    home_tov: int = 0,
    home_fta: int = 0,
    away_fga: int = 0,
    away_orb: int = 0,
    away_tov: int = 0,
    away_fta: int = 0,
    pre_game_home_expected: float = None,
    pre_game_away_expected: float = None,
    vig: float = 0.05,
    db_path: str = DB_PATH
) -> Dict:
    """
    Genera cuotas en vivo actualizadas.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        game_id: ID del partido
        current_home_score: Score actual del home
        current_away_score: Score actual del away
        minutes_elapsed: Minutos jugados (ej: 24 = fin del primer tiempo)
        home_fga, home_orb, home_tov, home_fta: Stats actuales del home
        away_fga, away_orb, away_tov, away_fta: Stats actuales del away
        pre_game_home_expected: Predicción pre-partido (opcional)
        pre_game_away_expected: Predicción pre-partido (opcional)
        vig: Margen de la casa
    
    Returns:
        Dict con cuotas en vivo
    """
    conn = sqlite3.connect(db_path)
    
    from src.predictions.prediction_engines.mack_predictor import get_team_id_from_abbreviation
    
    home_numeric = home_team_id
    away_numeric = away_team_id
    
    if not home_team_id.isdigit():
        home_numeric = get_team_id_from_abbreviation(home_team_id, conn)
    if not away_team_id.isdigit():
        away_numeric = get_team_id_from_abbreviation(away_team_id, conn)
    
    if not home_numeric or not away_numeric:
        conn.close()
        return {'error': 'Team not found'}
    
    cursor = conn.cursor()
    cursor.execute(
        """SELECT game_id FROM Games 
           WHERE (home_team = ? OR away_team = ?) 
           AND season = '2025-2026' AND status = 3
           AND game_id != ?
           ORDER BY date_time_utc DESC""",
        (home_team_id, home_team_id, game_id)
    )
    home_game_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.execute(
        """SELECT game_id FROM Games 
           WHERE (home_team = ? OR away_team = ?) 
           AND season = '2025-2026' AND status = 3
           AND game_id != ?
           ORDER BY date_time_utc DESC""",
        (away_team_id, away_team_id, game_id)
    )
    away_game_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    home_stats = get_team_recent_stats(home_numeric, home_game_ids, 10)
    away_stats = get_team_recent_stats(away_numeric, away_game_ids, 10)
    
    if not home_stats or not away_stats:
        return {'error': 'Insufficient data'}
    
    pre_home_expected, pre_away_expected = calculate_expected_score(
        home_stats, away_stats, DEFAULT_LEAGUE_AVG_ORtg, HOME_COURT_ADVANTAGE
    )
    
    if minutes_elapsed > 0:
        home_poss, away_poss = estimate_live_possessions(
            home_fga, home_orb, home_tov, home_fta,
            away_fga, away_orb, away_tov, away_fta
        )
        
        live_pace = calculate_live_pace(home_poss, away_poss, minutes_elapsed)
        
        live_ortg_home = calculate_live_ortg(current_home_score, home_poss) if home_poss > 0 else 100
        live_ortg_away = calculate_live_ortg(current_away_score, away_poss) if away_poss > 0 else 100
        
        home_expected, away_expected = live_prediction_bayes(
            pre_home_expected,
            pre_away_expected,
            current_home_score,
            current_away_score,
            minutes_elapsed
        )
        
        method = "bayes_blend"
    else:
        home_expected = pre_home_expected
        away_expected = pre_away_expected
        live_pace = home_stats.get('pace', 100)
        live_ortg_home = home_stats.get('ortg', 110)
        live_ortg_away = away_stats.get('ortg', 110)
        method = "pre_game"
    
    minutes_remaining = 48 - minutes_elapsed
    
    if minutes_remaining > 0:
        final_estimation = estimate_final_score_monte_carlo(
            current_home_score,
            current_away_score,
            minutes_remaining,
            live_pace
        )
    else:
        final_estimation = {
            'estimated_home_final': current_home_score,
            'estimated_away_final': current_away_score,
        }
    
    score_diff = home_expected - away_expected
    spread_line = round(score_diff, 1)
    
    total_expected = home_expected + away_expected
    total_line = round(total_expected)
    
    prob_home_win = stats.norm.cdf(score_diff / 10)
    prob_away_win = 1 - prob_home_win
    
    fair_home, fair_away = remove_vig(prob_home_win, prob_away_win)
    
    home_ml_vig = add_vig_to_prob(fair_home, vig)
    away_ml_vig = add_vig_to_prob(fair_away, vig)
    
    home_ml_odds = round(1 / home_ml_vig, 2)
    away_ml_odds = round(1 / away_ml_vig, 2)
    
    return {
        'game_id': game_id,
        'home_team': home_team_id,
        'away_team': away_team_id,
        'method': method,
        
        'current_score': {
            'home': current_home_score,
            'away': current_away_score,
        },
        
        'pre_game': {
            'home_expected': pre_home_expected,
            'away_expected': pre_away_expected,
        },
        
        'live': {
            'minutes_elapsed': minutes_elapsed,
            'minutes_remaining': round(minutes_remaining, 1),
            'home_expected': home_expected,
            'away_expected': away_expected,
            'total_expected': round(home_expected + away_expected, 1),
            'pace': live_pace,
            'live_ortg_home': live_ortg_home,
            'live_ortg_away': live_ortg_away,
        },
        
        'final_projection': final_estimation,
        
        'moneyline': {
            'home_decimal': home_ml_odds,
            'away_decimal': away_ml_odds,
            'home_american': to_american_odds(home_ml_odds),
            'away_american': to_american_odds(away_ml_odds),
            'home_prob': round(prob_home_win, 3),
            'away_prob': round(prob_away_win, 3),
            'fair_home_prob': round(fair_home, 3),
            'fair_away_prob': round(fair_away, 3),
        },
        
        'spread': {
            'line': spread_line,
            'home_odds': round(1 / (prob_home_win * (1 + vig)), 2),
            'away_odds': round(1 / (prob_away_win * (1 + vig)), 2),
        },
        
        'over_under': {
            'line': total_line,
            'over_odds': round(1 / (0.5 * (1 + vig)), 2),
            'under_odds': round(1 / (0.5 * (1 + vig)), 2),
        },
        
        'parameters': {
            'vig': vig,
        }
    }


def test_live_predictor():
    """
    Ejemplo: Q2, 6 minutos jugados
    Lakers vs Celtics, score 55-52
    """
    result = generate_live_odds(
        home_team_id='LAL',
        away_team_id='BOS',
        game_id='0022500123',
        current_home_score=55,
        current_away_score=52,
        minutes_elapsed=18.0,
        home_fga=45,
        home_orb=5,
        home_tov=6,
        home_fta=12,
        away_fga=42,
        away_orb=4,
        away_tov=8,
        away_fta=10,
        vig=0.05
    )
    
    print("=" * 60)
    print("LIVE PREDICTOR - EJEMPLO Q2, 6:00")
    print("=" * 60)
    print(f"Score: {result['current_score']['home']} - {result['current_score']['away']}")
    print(f"Minutos jugados: {result['live']['minutes_elapsed']}")
    print(f"Minutos restantes: {result['live']['minutes_remaining']}")
    print("")
    print(f"Pre-game expected: {result['pre_game']['home_expected']} - {result['pre_game']['away_expected']}")
    print(f"Live expected: {result['live']['home_expected']} - {result['live']['away_expected']}")
    print(f"Total expected: {result['live']['total_expected']}")
    print("")
    print(f"Final projection: {result['final_projection']['estimated_home_final']} - {result['final_projection']['estimated_away_final']}")
    print("")
    print(f"Live Pace: {result['live']['pace']}")
    print(f"Live ORtg Home: {result['live']['live_ortg_home']}")
    print(f"Live ORtg Away: {result['live']['live_ortg_away']}")
    print("")
    print(f"ML: {result['moneyline']['home_american']} / {result['moneyline']['away_american']}")
    print(f"Spread: {result['spread']['line']}")
    print(f"O/U: {result['over_under']['line']}")
    
    return result


if __name__ == "__main__":
    test_live_predictor()
