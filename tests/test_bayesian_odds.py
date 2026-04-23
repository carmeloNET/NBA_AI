"""
test_bayesian_odds.py

Tests for Bayesian Odds Generation (Mack Methodology).
Tests Prior + Likelihood → Posterior → Odds pipeline.
"""

import pytest
import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scipy import stats


@dataclass
class BayesianConfig:
    """Configuration for Bayesian model."""
    prior_weight: float = 0.3
    likelihood_weight: float = 0.7
    rolling_window_team: int = 15
    rolling_window_player: int = 15
    pythag_exponent: float = 14.0
    overround: float = 0.025
    league_avg_ortg: float = 112.0
    league_avg_pace: float = 100.0


class BayesianAggregator:
    """Computes Bayesian aggregates."""

    def __init__(self, config: BayesianConfig):
        self.config = config

    def compute_posterior(
        self,
        prior: Dict[str, float],
        likelihood: Dict[str, float]
    ) -> Dict[str, float]:
        """posterior = α * prior + β * likelihood"""
        alpha = self.config.prior_weight
        beta = self.config.likelihood_weight

        posterior = {}
        all_keys = set(prior.keys()) | set(likelihood.keys())

        for key in all_keys:
            p = prior.get(key, 0) or 0
            r = likelihood.get(key, 0) or 0

            if r == 0:
                posterior[key] = p
            else:
                posterior[key] = (alpha * p) + (beta * r)

        return posterior


class OddsGenerator:
    """Generates betting odds using statistical distributions."""

    def __init__(self, config: BayesianConfig):
        self.config = config

    def moneyline_prob(self, lambda_home: float, lambda_away: float) -> Tuple[float, float]:
        """Pythagorean win probability."""
        exp = self.config.pythag_exponent
        prob_home = (lambda_home ** exp) / (lambda_home ** exp + lambda_away ** exp)
        return prob_home, 1 - prob_home

    def moneyline_odds(self, prob: float) -> Tuple[float, float]:
        """Probability to decimal odds."""
        fair = 1 / prob
        offered = 1 / (prob + self.config.overround)
        return round(fair, 2), round(offered, 2)

    def total_prob(self, lambda_home: float, lambda_away: float, line: float) -> Tuple[float, float]:
        """P(Over) and P(Under) using Poisson."""
        lambda_total = lambda_home + lambda_away
        prob_over = 1 - stats.poisson.cdf(line, lambda_total)
        prob_under = stats.poisson.cdf(line, lambda_total)
        return prob_over, prob_under

    def spread_prob(self, lambda_home: float, lambda_away: float, line: float = 0.0) -> Tuple[float, float]:
        """P(Home covers) using Normal approximation to Skellam."""
        mean_diff = lambda_home - lambda_away
        std = np.sqrt(lambda_home + lambda_away) if lambda_home + lambda_away > 0 else 1
        prob_home_cover = 1 - stats.norm.cdf(line, loc=mean_diff, scale=std)
        return prob_home_cover, 1 - prob_home_cover

    def player_prop_prob(self, expected: float, line: float) -> Tuple[float, float]:
        """Probability for player props using Poisson."""
        if expected <= 0:
            return 0.0, 1.0
        prob_over = 1 - stats.poisson.cdf(line, expected)
        prob_under = stats.poisson.cdf(line, expected)
        return prob_over, prob_under

    def double_double_prob(self, pts: float, reb: float, ast: float) -> float:
        """P(Double-Double) - simplified."""
        if pts <= 0 or reb <= 0 or ast <= 0:
            return 0.0
        prob_pts = 1 - stats.poisson.cdf(10, pts)
        prob_rb = 1 - stats.poisson.cdf(10, reb)
        prob_ast = 1 - stats.poisson.cdf(10, ast)
        prob_dd = min(prob_pts + prob_rb + prob_ast, 1.0)
        return prob_dd

    def triple_double_prob(self, pts: float, reb: float, ast: float) -> float:
        """P(Triple-Double) - very rare."""
        if pts <= 0 or reb <= 0 or ast <= 0:
            return 0.0
        prob_pts = 1 - stats.poisson.cdf(10, pts)
        prob_rb = 1 - stats.poisson.cdf(10, reb)
        prob_ast = 1 - stats.poisson.cdf(10, ast)
        return min(prob_pts * prob_rb * prob_ast * 3, 0.1)


class TeamOddsCalculator:
    """Calculate team odds from Bayesian aggregates."""

    def __init__(self, config: BayesianConfig):
        self.config = config
        self.aggregator = BayesianAggregator(config)
        self.odds = OddsGenerator(config)

    def calculate_team_odds(
        self,
        home_prior: Dict[str, float],
        away_prior: Dict[str, float],
        home_rolling: Dict[str, float],
        away_rolling: Dict[str, float]
    ) -> Dict:
        """Calculate all team betting odds."""

        home_post = self.aggregator.compute_posterior(home_prior, home_rolling)
        away_post = self.aggregator.compute_posterior(away_prior, away_rolling)

        league_avg = self.config.league_avg_ortg
        home_ortg = home_post.get('ortg', 110)
        away_ortg = away_post.get('ortg', 110)
        home_drtg = away_post.get('drtg', 110)
        away_drtg = home_post.get('drtg', 110)

        pace = (home_post.get('pace', 100) + away_post.get('pace', 100)) / 2

        exp_home = pace * (home_ortg * away_drtg / league_avg / 100)
        exp_away = pace * (away_ortg * home_drtg / league_avg / 100)
        exp_total = exp_home + exp_away

        prob_home, prob_away = self.odds.moneyline_prob(exp_home, exp_away)
        fair_home, comm_home = self.odds.moneyline_odds(prob_home)
        fair_away, comm_away = self.odds.moneyline_odds(prob_away)

        total_line = round(exp_total, 1)
        prob_over, prob_under = self.odds.total_prob(exp_home, exp_away, total_line)

        exp_margin = exp_home - exp_away
        spread_line = round(exp_margin, 1)
        prob_home_cover, _ = self.odds.spread_prob(exp_home, exp_away, spread_line)

        return {
            'expected_home': round(exp_home, 1),
            'expected_away': round(exp_away, 1),
            'expected_total': round(exp_total, 1),
            'total_line': total_line,
            'prob_home_win': round(prob_home, 4),
            'prob_away_win': round(prob_away, 4),
            'fair_odds_home': fair_home,
            'fair_odds_away': fair_away,
            'offered_odds_home': comm_home,
            'offered_odds_away': comm_away,
            'prob_over': round(prob_over, 4),
            'prob_under': round(prob_under, 4),
            'fair_odds_over': round(1/prob_over, 2),
            'fair_odds_under': round(1/prob_under, 2),
            'expected_margin': round(exp_margin, 1),
            'spread_line': spread_line,
            'prob_home_cover': round(prob_home_cover, 4),
            'prob_away_cover': round(1 - prob_home_cover, 4),
        }


class PlayerPropsCalculator:
    """Calculate player props from Bayesian aggregates."""

    def __init__(self, config: BayesianConfig):
        self.config = config
        self.odds = OddsGenerator(config)

    def calculate_player_props(
        self,
        prior_pts: float,
        prior_reb: float,
        prior_ast: float,
        prior_three: float,
        rolling_pts: float,
        rolling_reb: float,
        rolling_ast: float,
        rolling_three: float,
    ) -> Dict:
        """Calculate player props."""

        alpha = self.config.prior_weight
        beta = self.config.likelihood_weight

        pts = alpha * prior_pts + beta * rolling_pts
        reb = alpha * prior_reb + beta * rolling_reb
        ast = alpha * prior_ast + beta * rolling_ast
        three = alpha * prior_three + beta * rolling_three

        pts_line = round(pts)
        reb_line = round(reb)
        ast_line = round(ast)
        three_line = round(three)

        pts_over, _ = self.odds.player_prop_prob(pts, pts_line)
        reb_over, _ = self.odds.player_prop_prob(reb, reb_line)
        ast_over, _ = self.odds.player_prop_prob(ast, ast_line)
        three_over, _ = self.odds.player_prop_prob(three, three_line)

        par = pts + reb + ast
        par_line = round(par)
        par_over, _ = self.odds.player_prop_prob(par, par_line)

        prob_dd = self.odds.double_double_prob(pts, reb, ast)
        prob_td = self.odds.triple_double_prob(pts, reb, ast)

        return {
            'expected_pts': round(pts, 1),
            'expected_reb': round(reb, 1),
            'expected_ast': round(ast, 1),
            'expected_three': round(three, 1),
            'pts_line': pts_line,
            'pts_prob_over': round(pts_over, 4),
            'reb_line': reb_line,
            'reb_prob_over': round(reb_over, 4),
            'ast_line': ast_line,
            'ast_prob_over': round(ast_over, 4),
            'three_line': three_line,
            'three_prob_over': round(three_over, 4),
            'par_expected': round(par, 1),
            'par_line': par_line,
            'par_prob_over': round(par_over, 4),
            'prob_double_double': round(prob_dd, 4),
            'prob_triple_double': round(prob_td, 4),
        }


# =============================================================================
# Unit Tests - Configuration
# =============================================================================

class TestBayesianConfig:
    """Tests for Bayesian configuration."""

    def test_default_config_values(self):
        """Default config should have correct values."""
        config = BayesianConfig()
        assert config.prior_weight == 0.3
        assert config.likelihood_weight == 0.7
        assert config.rolling_window_team == 15
        assert config.rolling_window_player == 15
        assert config.pythag_exponent == 14.0
        assert config.overround == 0.025
        assert config.league_avg_ortg == 112.0
        assert config.league_avg_pace == 100.0

    def test_custom_config_values(self):
        """Custom config should override defaults."""
        config = BayesianConfig(
            prior_weight=0.5,
            likelihood_weight=0.5,
            rolling_window_team=20
        )
        assert config.prior_weight == 0.5
        assert config.likelihood_weight == 0.5
        assert config.rolling_window_team == 20


# =============================================================================
# Unit Tests - Bayesian Posterior
# =============================================================================

class TestBayesianPosterior:
    """Tests for Bayesian posterior calculation."""

    def test_posterior_combines_prior_and_likelihood(self):
        """Posterior should be weighted combination."""
        config = BayesianConfig()
        agg = BayesianAggregator(config)

        prior = {'ortg': 110.0, 'drtg': 108.0, 'pace': 100.0}
        likelihood = {'ortg': 115.0, 'drtg': 105.0, 'pace': 102.0}

        posterior = agg.compute_posterior(prior, likelihood)

        assert 'ortg' in posterior
        assert 'drtg' in posterior
        assert 'pace' in posterior
        # 0.3 * 110 + 0.7 * 115 = 33 + 80.5 = 113.5
        assert abs(posterior['ortg'] - 113.5) < 0.1
        # 0.3 * 108 + 0.7 * 105 = 32.4 + 73.5 = 105.9
        assert abs(posterior['drtg'] - 105.9) < 0.1

    def test_posterior_with_missing_likelihood_uses_prior(self):
        """Missing likelihood should fall back to prior."""
        config = BayesianConfig()
        agg = BayesianAggregator(config)

        prior = {'ortg': 110.0}
        likelihood = {}

        posterior = agg.compute_posterior(prior, likelihood)
        assert posterior['ortg'] == 110.0

    def test_posterior_weights_sum_to_one(self):
        """α + β should equal 1."""
        config = BayesianConfig()
        total = config.prior_weight + config.likelihood_weight
        assert abs(total - 1.0) < 0.001


# =============================================================================
# Unit Tests - Moneyline Odds
# =============================================================================

class TestMoneylineOdds:
    """Tests for moneyline probability and odds."""

    def test_pythagorean_probability_home_favored(self):
        """When home has higher λ, should have higher win probability."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_home, prob_away = odds.moneyline_prob(115.0, 108.0)

        assert prob_home > prob_away
        assert 0 < prob_home < 1
        assert 0 < prob_away < 1

    def test_pythagorean_probabilities_sum_to_one(self):
        """Home + Away probabilities should sum to 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_home, prob_away = odds.moneyline_prob(115.0, 108.0)

        assert abs(prob_home + prob_away - 1.0) < 0.0001

    def test_fair_odds_greater_than_one(self):
        """Fair odds should always be greater than 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        for prob in [0.3, 0.4, 0.5, 0.6, 0.7]:
            fair, _ = odds.moneyline_odds(prob)
            assert fair > 1.0

    def test_offered_odds_less_than_fair(self):
        """Offered odds (with overround) should be less than fair."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        fair, offered = odds.moneyline_odds(0.5)
        assert offered < fair


# =============================================================================
# Unit Tests - Total Odds (Over/Under)
# =============================================================================

class TestTotalOdds:
    """Tests for total (Over/Under) odds."""

    def test_over_under_sum_to_one(self):
        """Over + Under probabilities should sum to 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_over, prob_under = odds.total_prob(115.0, 108.0, 220.0)

        assert abs(prob_over + prob_under - 1.0) < 0.001

    def test_over_prob_is_valid(self):
        """Over probability should be between 0 and 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_over, _ = odds.total_prob(115.0, 108.0, 220.0)
        assert 0 <= prob_over <= 1

    def test_higher_line_lower_over_prob(self):
        """Higher total line should have lower over probability."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_over_low, _ = odds.total_prob(115.0, 108.0, 220.0)
        prob_over_high, _ = odds.total_prob(115.0, 108.0, 240.0)

        assert prob_over_low > prob_over_high


# =============================================================================
# Unit Tests - Spread Odds
# =============================================================================

class TestSpreadOdds:
    """Tests for spread ( ATS) odds."""

    def test_spread_probabilities_sum_to_one(self):
        """Home + Away cover probabilities should sum to 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_home, prob_away = odds.spread_prob(115.0, 108.0, 4.0)

        assert abs(prob_home + prob_away - 1.0) < 0.0001

    def test_spread_probabilities_are_valid(self):
        """Spread probabilities should be between 0 and 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_home, prob_away = odds.spread_prob(115.0, 108.0, 4.0)

        assert 0 <= prob_home <= 1
        assert 0 <= prob_away <= 1


# =============================================================================
# Unit Tests - Player Props
# =============================================================================

class TestPlayerProps:
    """Tests for player props probability calculations."""

    def test_player_prop_probabilities_sum_to_one(self):
        """Over + Under should sum to 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_over, prob_under = odds.player_prop_prob(25.0, 24.5)

        assert abs(prob_over + prob_under - 1.0) < 0.001

    def test_zero_expected_returns_zero_over(self):
        """Zero expected value should return 0 for over."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_over, prob_under = odds.player_prop_prob(0, 10)

        assert prob_over == 0
        assert prob_under == 1

    def test_double_double_is_rare(self):
        """Double double probability should be between 0 and 1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_dd = odds.double_double_prob(25.0, 8.0, 7.0)

        assert 0 <= prob_dd <= 1

    def test_triple_double_is_very_rare(self):
        """Triple double probability should be capped at 0.1."""
        config = BayesianConfig()
        odds = OddsGenerator(config)

        prob_td = odds.triple_double_prob(25.0, 10.0, 12.0)

        assert 0 <= prob_td <= 0.1


# =============================================================================
# Unit Tests - Full Pipeline
# =============================================================================

class TestFullOddsPipeline:
    """Tests for complete odds calculation pipeline."""

    def test_team_odds_output_structure(self):
        """Team odds should have all expected fields."""
        config = BayesianConfig()
        calc = TeamOddsCalculator(config)

        home_prior = {'ortg': 110.0, 'drtg': 108.0, 'pace': 100.0}
        away_prior = {'ortg': 112.0, 'drtg': 110.0, 'pace': 98.0}
        home_rolling = {'ortg': 113.0, 'drtg': 106.0, 'pace': 102.0}
        away_rolling = {'ortg': 108.0, 'drtg': 112.0, 'pace': 96.0}

        result = calc.calculate_team_odds(
            home_prior, away_prior, home_rolling, away_rolling
        )

        required_fields = [
            'expected_home', 'expected_away', 'expected_total',
            'total_line', 'prob_home_win', 'prob_away_win',
            'fair_odds_home', 'fair_odds_away',
            'prob_over', 'prob_under', 'spread_line', 'prob_home_cover'
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_team_odds_probabilities_sum_to_one(self):
        """ML probabilities should sum to 1."""
        config = BayesianConfig()
        calc = TeamOddsCalculator(config)

        home_prior = {'ortg': 110.0, 'drtg': 108.0, 'pace': 100.0}
        away_prior = {'ortg': 112.0, 'drtg': 110.0, 'pace': 98.0}
        home_rolling = {'ortg': 113.0, 'drtg': 106.0, 'pace': 102.0}
        away_rolling = {'ortg': 108.0, 'drtg': 112.0, 'pace': 96.0}

        result = calc.calculate_team_odds(
            home_prior, away_prior, home_rolling, away_rolling
        )

        assert abs(result['prob_home_win'] + result['prob_away_win'] - 1.0) < 0.001
        assert abs(result['prob_over'] + result['prob_under'] - 1.0) < 0.001
        assert abs(result['prob_home_cover'] + result['prob_away_cover'] - 1.0) < 0.001

    def test_player_props_output_structure(self):
        """Player props should have all expected fields."""
        config = BayesianConfig()
        calc = PlayerPropsCalculator(config)

        result = calc.calculate_player_props(
            prior_pts=20.0, prior_reb=7.0, prior_ast=6.0, prior_three=2.0,
            rolling_pts=25.0, rolling_reb=8.0, rolling_ast=7.0, rolling_three=3.0
        )

        required_fields = [
            'expected_pts', 'expected_reb', 'expected_ast', 'expected_three',
            'pts_line', 'pts_prob_over', 'reb_line', 'ast_line',
            'prob_double_double', 'prob_triple_double'
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_player_props_expected_close_to_line(self):
        """Expected value should be close to the line (within 1.5)."""
        config = BayesianConfig()
        calc = PlayerPropsCalculator(config)

        result = calc.calculate_player_props(
            prior_pts=20.0, prior_reb=7.0, prior_ast=6.0, prior_three=2.0,
            rolling_pts=25.0, rolling_reb=8.0, rolling_ast=7.0, rolling_three=3.0
        )

        assert abs(result['expected_pts'] - result['pts_line']) <= 1.5
        assert abs(result['expected_reb'] - result['reb_line']) <= 1.5
        assert abs(result['expected_ast'] - result['ast_line']) <= 1.5


# =============================================================================
# Integration Tests - Full Projection for Simulated Games
# =============================================================================

class TestSimulatedProjections:
    """Integration tests with simulated data."""

    def test_simulated_game_lakers_vs_celtics(self):
        """Simulate LAL vs BOS game."""
        config = BayesianConfig()
        calc = TeamOddsCalculator(config)

        # LAL priors (strong offense)
        lal_prior = {'ortg': 115.0, 'drtg': 112.0, 'pace': 104.0}
        lal_rolling = {'ortg': 118.0, 'drtg': 108.0, 'pace': 106.0}

        # BOS priors (strong defense)
        bos_prior = {'ortg': 113.0, 'drtg': 108.0, 'pace': 98.0}
        bos_rolling = {'ortg': 112.0, 'drtg': 106.0, 'pace': 96.0}

        result = calc.calculate_team_odds(
            lal_prior, bos_prior, lal_rolling, bos_rolling
        )

        print(f"\nLAL vs BOS Projection:")
        print(f"  Expected Score: {result['expected_home']:.1f} - {result['expected_away']:.1f}")
        print(f"  Total: {result['expected_total']:.1f} (Line: {result['total_line']})")
        print(f"  ML: Home {result['prob_home_win']:.1%} vs Away {result['prob_away_win']:.1%}")
        print(f"  Spread: {result['spread_line']:+.1f} (Home cover: {result['prob_home_cover']:.1%})")

        # Validations
        assert result['expected_total'] > 200
        assert result['expected_total'] < 250
        assert result['prob_home_win'] > 0
        assert result['prob_home_win'] < 1

    def test_simulated_game_warriors_vs_nuggets(self):
        """Simulate GSW vs DEN game."""
        config = BayesianConfig()
        calc = TeamOddsCalculator(config)

        # GSW (high pace, 3PT heavy)
        gsw_prior = {'ortg': 114.0, 'drtg': 114.0, 'pace': 106.0}
        gsw_rolling = {'ortg': 116.0, 'drtg': 112.0, 'pace': 108.0}

        # DEN (slow, efficient)
        den_prior = {'ortg': 118.0, 'drtg': 110.0, 'pace': 96.0}
        den_rolling = {'ortg': 120.0, 'drtg': 108.0, 'pace': 94.0}

        result = calc.calculate_team_odds(
            gsw_prior, den_prior, gsw_rolling, den_rolling
        )

        print(f"\nGSW vs DEN Projection:")
        print(f"  Expected Score: {result['expected_home']:.1f} - {result['expected_away']:.1f}")
        print(f"  Total: {result['expected_total']:.1f} (Line: {result['total_line']})")
        print(f"  ML: Home {result['prob_home_win']:.1%} vs Away {result['prob_away_win']:.1%}")

        assert result['expected_total'] > 200
        assert result['prob_home_win'] + result['prob_away_win'] == pytest.approx(1.0, rel=0.001)

    def test_simulated_player_lebron(self):
        """Simulate LeBron James props."""
        config = BayesianConfig()
        calc = PlayerPropsCalculator(config)

        # LeBron: high usage, good all-around
        result = calc.calculate_player_props(
            prior_pts=25.0, prior_reb=7.5, prior_ast=8.0, prior_three=2.0,
            rolling_pts=24.0, rolling_reb=7.5, rolling_ast=8.5, rolling_three=1.8
        )

        print(f"\nLeBron James Props Projection:")
        print(f"  PTS: {result['expected_pts']} (Line: {result['pts_line']}, Over: {result['pts_prob_over']:.1%})")
        print(f"  REB: {result['expected_reb']} (Line: {result['reb_line']}, Over: {result['reb_prob_over']:.1%})")
        print(f"  AST: {result['expected_ast']} (Line: {result['ast_line']}, Over: {result['ast_prob_over']:.1%})")
        print(f"  DD: {result['prob_double_double']:.1%}")
        print(f"  TD: {result['prob_triple_double']:.1%}")

        assert result['expected_pts'] > 20
        assert result['expected_ast'] > 5
        assert result['prob_double_double'] > 0


# =============================================================================
# Projections for Today/Tomorrow (Mock Data)
# =============================================================================

class TestTodaysProjections:
    """Projections for games today/tomorrow (using mock data)."""

    def test_todays_games_projection(self):
        """Generate projections for today's games (mock)."""
        config = BayesianConfig()
        team_calc = TeamOddsCalculator(config)
        player_calc = PlayerPropsCalculator(config)

        today = date.today()
        tomorrow = today + timedelta(days=1)

        games = [
            {
                'date': today,
                'game_id': '0022500010',
                'home': 'LAL',
                'away': 'GSW',
                'home_prior': {'ortg': 114.0, 'drtg': 113.0, 'pace': 102.0},
                'home_rolling': {'ortg': 116.0, 'drtg': 110.0, 'pace': 104.0},
                'away_prior': {'ortg': 113.0, 'drtg': 112.0, 'pace': 104.0},
                'away_rolling': {'ortg': 115.0, 'drtg': 108.0, 'pace': 106.0},
                'home_players': [
                    {'name': 'LeBron James', 'pts': 24, 'reb': 7, 'ast': 8, 'three': 1.8},
                    {'name': 'Anthony Davis', 'pts': 26, 'reb': 12, 'ast': 3, 'three': 0.5},
                ],
                'away_players': [
                    {'name': 'Stephen Curry', 'pts': 28, 'reb': 5, 'ast': 6, 'three': 5.0},
                    {'name': 'Draymond Green', 'pts': 12, 'reb': 7, 'ast': 8, 'three': 1.5},
                ]
            },
            {
                'date': today,
                'game_id': '0022500011',
                'home': 'BOS',
                'away': 'MIA',
                'home_prior': {'ortg': 118.0, 'drtg': 110.0, 'pace': 98.0},
                'home_rolling': {'ortg': 120.0, 'drtg': 108.0, 'pace': 96.0},
                'away_prior': {'ortg': 112.0, 'drtg': 112.0, 'pace': 100.0},
                'away_rolling': {'ortg': 110.0, 'drtg': 114.0, 'pace': 98.0},
                'home_players': [
                    {'name': 'Jayson Tatum', 'pts': 27, 'reb': 8, 'ast': 4, 'three': 3.0},
                    {'name': 'Jaylen Brown', 'pts': 24, 'reb': 6, 'ast': 3, 'three': 2.5},
                ],
                'away_players': [
                    {'name': 'Jimmy Butler', 'pts': 22, 'reb': 5, 'ast': 5, 'three': 1.0},
                    {'name': 'Bam Adebayo', 'pts': 18, 'reb': 10, 'ast': 4, 'three': 0.2},
                ]
            }
        ]

        print(f"\n{'='*60}")
        print(f"BAYESIAN ODDS PROJECTIONS - {today.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        all_projections = []

        for game in games:
            team_result = team_calc.calculate_team_odds(
                game['home_prior'], game['away_prior'],
                game['home_rolling'], game['away_rolling']
            )

            player_props = []
            for player in game['home_players'] + game['away_players']:
                props = player_calc.calculate_player_props(
                    prior_pts=player['pts'] * 0.8,
                    prior_reb=player['reb'] * 0.8,
                    prior_ast=player['ast'] * 0.8,
                    prior_three=player['three'] * 0.8,
                    rolling_pts=player['pts'],
                    rolling_reb=player['reb'],
                    rolling_ast=player['ast'],
                    rolling_three=player['three'],
                )
                props['player_name'] = player['name']
                player_props.append(props)

            projection = {
                'game_id': game['game_id'],
                'date': game['date'],
                'home': game['home'],
                'away': game['away'],
                'team_odds': team_result,
                'player_props': player_props,
            }
            all_projections.append(projection)

            print(f"\n{game['home']} vs {game['away']} ({game['date']})")
            print(f"  Score Projection: {team_result['expected_home']:.1f} - {team_result['expected_away']:.1f}")
            print(f"  Total: {team_result['expected_total']:.1f} | Line: {team_result['total_line']}")
            print(f"  Over Prob: {team_result['prob_over']:.1%} | Fair Odds: {team_result['fair_odds_over']}")
            print(f"  ML: Home {team_result['prob_home_win']:.1%} ({team_result['fair_odds_home']})")
            print(f"  Spread: {team_result['spread_line']:+.1f} | Home Cover: {team_result['prob_home_cover']:.1%}")
            print(f"  Player Props:")
            for p in player_props:
                print(f"    {p['player_name']}: PTS {p['expected_pts']} ({p['pts_prob_over']:.0%} O) | "
                      f"REB {p['expected_reb']} ({p['reb_prob_over']:.0%} O) | "
                      f"AST {p['expected_ast']} ({p['ast_prob_over']:.0%} O)")

        return all_projections