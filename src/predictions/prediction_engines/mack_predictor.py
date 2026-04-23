"""
Mack Predictor - Statistical approach using Andrew Mack's formulas.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple

from scipy import stats

from src.config import config
from src.predictions.prediction_engines.mack_advanced_stats import (
    DEFAULT_LEAGUE_AVG_ORtg,
    get_team_recent_stats,
)

DB_PATH = config["database"]["path"]

HOME_COURT_ADVANTAGE = 3.0
BACK_TO_BACK_PENALTY = 2.0


def get_team_id_from_abbreviation(abbrev: str, conn: sqlite3.Connection) -> Optional[str]:
    """Convert team abbreviation to numeric team_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT team_id FROM Teams WHERE abbreviation = ?", (abbrev,))
    result = cursor.fetchone()
    return result[0] if result else None


def calculate_expected_score(
    home_stats: Dict,
    away_stats: Dict,
    league_avg_ortg: float = DEFAULT_LEAGUE_AVG_ORtg,
    home_advantage: float = HOME_COURT_ADVANTAGE,
) -> Tuple[float, float]:
    """
    Calculate expected scores using Mack's formula.
    
    Expected_Score = Pace × (ORtg × DRtg_opponent) / 100
    
    Then apply home court advantage adjustment.
    """
    home_pace = home_stats.get('pace', 100)
    away_pace = away_stats.get('pace', 100)
    home_ortg = home_stats.get('ortg', 110)
    away_ortg = away_stats.get('ortg', 110)
    home_drtg = home_stats.get('drtg', 110)
    away_drtg = away_stats.get('drtg', 110)
    
    expected_pace = (home_pace + away_pace) / 2
    
    home_expected = expected_pace * (home_ortg * away_drtg) / league_avg_ortg / 100
    away_expected = expected_pace * (away_ortg * home_drtg) / league_avg_ortg / 100
    
    home_expected += home_advantage
    
    return round(home_expected, 1), round(away_expected, 1)


def calculate_win_probability_mack(
    home_score: float,
    away_score: float,
    home_advantage: float = HOME_COURT_ADVANTAGE
) -> float:
    """
    Calculate win probability using normal distribution approximation.
    Based on Feustel's approach.
    
    Assumes standard deviation of ~10 points for NBA games.
    """
    score_diff = home_score - away_score - home_advantage
    
    std_dev = 10.0
    
    prob_home_win = stats.norm.cdf(score_diff / std_dev)
    
    return round(prob_home_win, 3)


def poisson_over_under(
    home_lambda: float,
    away_lambda: float,
    total_line: float,
    vig: float = 0.05
) -> Dict:
    """
    Calculate Over/Under probabilities using Poisson distribution.
    """
    total_lambda = home_lambda + away_lambda
    
    prob_over = 1 - stats.poisson.cdf(total_line - 1, total_lambda)
    prob_under = stats.poisson.cdf(total_line - 1, total_lambda)
    
    prob_over_adj = prob_over * (1 + vig)
    prob_under_adj = prob_under * (1 + vig)
    
    over_odds = 1 / prob_over_adj
    under_odds = 1 / prob_under_adj
    
    return {
        'line': total_line,
        'prob_over': round(prob_over, 3),
        'prob_under': round(prob_under, 3),
        'over_odds': round(over_odds, 2),
        'under_odds': round(under_odds, 2),
    }


def skellam_spread(
    home_lambda: float,
    away_lambda: float,
    spread_line: float = 0,
    vig: float = 0.05
) -> Dict:
    """
    Calculate spread probabilities using Skellam distribution.
    Skellam is the difference between two independent Poisson random variables.
    """
    prob_home_cover = 1 - stats.skellam.cdf(-spread_line, home_lambda, away_lambda)
    prob_away_cover = stats.skellam.cdf(-spread_line, home_lambda, away_lambda)
    
    prob_home_cover_adj = prob_home_cover * (1 + vig)
    prob_away_cover_adj = prob_away_cover * (1 + vig)
    
    home_spread_odds = 1 / prob_home_cover_adj
    away_spread_odds = 1 / prob_away_cover_adj
    
    home_ml_prob = 1 - stats.skellam.cdf(0, home_lambda, away_lambda)
    away_ml_prob = stats.skellam.cdf(0, home_lambda, away_lambda)
    
    total_prob = home_ml_prob + away_ml_prob
    fair_home_prob = home_ml_prob / total_prob
    fair_away_prob = away_ml_prob / total_prob
    
    home_ml_with_vig = fair_home_prob * (1 + vig)
    away_ml_with_vig = fair_away_prob * (1 + vig)
    
    home_ml_odds = 1 / home_ml_with_vig
    away_ml_odds = 1 / away_ml_with_vig
    
    home_ml_american = to_american_odds(home_ml_odds)
    away_ml_american = to_american_odds(away_ml_odds)
    
    return {
        'spread_line': spread_line,
        'prob_home_cover': round(prob_home_cover, 3),
        'prob_away_cover': round(prob_away_cover, 3),
        'home_spread_odds': round(home_spread_odds, 2),
        'away_spread_odds': round(away_spread_odds, 2),
        'home_ml_odds': round(home_ml_odds, 2),
        'away_ml_odds': round(away_ml_odds, 2),
        'home_ml_american': home_ml_american,
        'away_ml_american': away_ml_american,
    }


def to_american_odds(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))


def remove_vig(prob_home: float, prob_away: float) -> Tuple[float, float]:
    """
    Remove vig (overround) from probabilities.
    """
    total = prob_home + prob_away
    fair_home = prob_home / total
    fair_away = prob_away / total
    return fair_home, fair_away


def add_vig_to_prob(prob: float, vig: float = 0.05) -> float:
    """Add vig to a probability."""
    return prob * (1 + vig)


def generate_mack_odds(
    home_team_id: str,
    away_team_id: str,
    game_id: str,
    season: str = "2025-2026",
    rolling_window: int = 10,
    vig: float = 0.05,
    db_path: str = DB_PATH
) -> Dict:
    """
    Generate complete odds using Mack's statistical approach.
    
    Returns:
        Dict with expected scores, probabilities, and odds for all markets.
    """
    conn = sqlite3.connect(db_path)
    
    # Convert abbreviations to numeric IDs if needed
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
           AND season = ? AND status = 3
           AND game_id != ?
           ORDER BY date_time_utc DESC""",
        (home_team_id, home_team_id, season, game_id)
    )
    home_game_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.execute(
        """SELECT game_id FROM Games 
           WHERE (home_team = ? OR away_team = ?) 
           AND season = ? AND status = 3
           AND game_id != ?
           ORDER BY date_time_utc DESC""",
        (away_team_id, away_team_id, season, game_id)
    )
    away_game_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    home_stats = get_team_recent_stats(home_numeric, home_game_ids, rolling_window)
    away_stats = get_team_recent_stats(away_numeric, away_game_ids, rolling_window)
    
    if not home_stats or not away_stats:
        return {
            'error': 'Insufficient data for prediction',
            'home_stats': home_stats,
            'away_stats': away_stats,
        }
    
    home_expected, away_expected = calculate_expected_score(home_stats, away_stats)
    
    win_prob = calculate_win_probability_mack(home_expected, away_expected)
    
    total_expected = home_expected + away_expected
    total_line = round(total_expected)
    
    spread_line = round(home_expected - away_expected, 1)
    
    spread_result = skellam_spread(home_expected, away_expected, spread_line, vig)
    
    ou_result = poisson_over_under(home_expected, away_expected, total_line, vig)
    
    moneyline_home_prob = 1 - stats.skellam.cdf(0, home_expected, away_expected)
    moneyline_away_prob = stats.skellam.cdf(0, home_expected, away_expected)
    
    fair_home, fair_away = remove_vig(moneyline_home_prob, moneyline_away_prob)
    
    home_ml_vig = add_vig_to_prob(fair_home, vig)
    away_ml_vig = add_vig_to_prob(fair_away, vig)
    
    home_ml_odds = round(1 / home_ml_vig, 2)
    away_ml_odds = round(1 / away_ml_vig, 2)
    
    return {
        'game_id': game_id,
        'home_team': home_team_id,
        'away_team': away_team_id,
        'method': 'Mack',
        
        'expected_scores': {
            'home': home_expected,
            'away': away_expected,
            'total': round(home_expected + away_expected, 1),
        },
        
        'probabilities': {
            'home_win': win_prob,
            'away_win': round(1 - win_prob, 3),
        },
        
        'moneyline': {
            'home_decimal': home_ml_odds,
            'away_decimal': away_ml_odds,
            'home_american': to_american_odds(home_ml_odds),
            'away_american': to_american_odds(away_ml_odds),
            'fair_home_prob': round(fair_home, 3),
            'fair_away_prob': round(fair_away, 3),
        },
        
        'spread': {
            'line': spread_line,
            'home_odds': spread_result['home_spread_odds'],
            'away_odds': spread_result['away_spread_odds'],
            'home_prob_cover': spread_result['prob_home_cover'],
            'away_prob_cover': spread_result['prob_away_cover'],
        },
        
        'over_under': {
            'line': total_line,
            'over_odds': ou_result['over_odds'],
            'under_odds': ou_result['under_odds'],
            'prob_over': ou_result['prob_over'],
            'prob_under': ou_result['prob_under'],
        },
        
        'advanced_stats': {
            'home': {
                'pace': round(home_stats.get('pace', 0), 1),
                'ortg': round(home_stats.get('ortg', 0), 1),
                'drtg': round(home_stats.get('drtg', 0), 1),
                'efg_pct': round(home_stats.get('efg_pct', 0), 3),
                'games': home_stats.get('games', 0),
            },
            'away': {
                'pace': round(away_stats.get('pace', 0), 1),
                'ortg': round(away_stats.get('ortg', 0), 1),
                'drtg': round(away_stats.get('drtg', 0), 1),
                'efg_pct': round(away_stats.get('efg_pct', 0), 3),
                'games': away_stats.get('games', 0),
            },
        },
        
        'parameters': {
            'vig': vig,
            'rolling_window': rolling_window,
            'home_advantage': HOME_COURT_ADVANTAGE,
        }
    }


def test_mack_predictor():
    """Test the Mack predictor with sample data."""
    result = generate_mack_odds(
        home_team_id='LAL',
        away_team_id='BOS',
        game_id='TEST001',
        season='2025-2026',
        rolling_window=10,
        vig=0.05
    )
    print("Mack Predictor Test:")
    print(f"  Expected Scores: {result.get('expected_scores', {})}")
    print(f"  Moneyline: {result.get('moneyline', {})}")
    print(f"  Spread: {result.get('spread', {})}")
    print(f"  O/U: {result.get('over_under', {})}")
    return result


if __name__ == "__main__":
    test_mack_predictor()
