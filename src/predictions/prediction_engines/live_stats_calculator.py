"""
Live Stats Calculator - Usa datos en tiempo real del boxscore NBA API.
Calcula estadísticas avanzadas de jugador y equipo en vivo.
"""

from typing import Dict, List, Optional


class LiveStatsCalculator:
    """
    Calcula estadísticas avanzadas en tiempo real usando el boxscore de la NBA API.
    """
    
    @staticmethod
    def parse_minutes(minutes_str: str) -> float:
        """
        Convierte "PT25M10.00S" a minutos decimales (25.167)
        """
        if not minutes_str or minutes_str == "PT00M00.00S":
            return 0.0
        
        import re
        match = re.match(r"PT(\d+)M(\d+(?:\.\d+)?)S", minutes_str)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes + seconds / 60.0
        return 0.0
    
    @staticmethod
    def calculate_team_possessions(fga: int, orb: int, tov: int, fta: int) -> float:
        """
        Dean Oliver's formula: Poss = FGA - ORB + TOV + 0.44 * FTA
        """
        return fga - orb + tov + (0.44 * fta)
    
    @staticmethod
    def calculate_live_player_stats(
        player_stats: Dict,
        team_stats: Dict,
        opp_team_stats: Dict,
        player_minutes: float,
        team_total_minutes: float,
        opp_team_total_minutes: float,
        team_drtg: float = 0
    ) -> Dict:
        """
        Calcula todas las stats avanzadas de un jugador en vivo.
        
        Args:
            player_stats: stats del jugador del boxscore NBA API
            team_stats: stats agregados del equipo del jugador
            opp_team_stats: stats agregados del equipo oponente
            player_minutes: minutos jugados por el jugador
            team_total_minutes: minutos totales del equipo (240 en un partido completo)
            opp_team_total_minutes: minutos totales del oponente
            team_drtg: defensive rating del equipo (para calcular DRtg del jugador)
        """
        # Extraer stats básicos del jugador
        pts = player_stats.get('points', 0)
        fgm = player_stats.get('fieldGoalsMade', 0)
        fga = player_stats.get('fieldGoalsAttempted', 0)
        fg3m = player_stats.get('threePointersMade', 0)
        fg3a = player_stats.get('threePointersAttempted', 0)
        ftm = player_stats.get('freeThrowsMade', 0)
        fta = player_stats.get('freeThrowsAttempted', 0)
        orb = player_stats.get('reboundsOffensive', 0)
        drb = player_stats.get('reboundsDefensive', 0)
        ast = player_stats.get('assists', 0)
        stl = player_stats.get('steals', 0)
        blk = player_stats.get('blocks', 0)
        tov = player_stats.get('turnovers', 0)
        pf = player_stats.get('foulsPersonal', 0)
        
        # Stats del equipo
        team_fgm = team_stats.get('fieldGoalsMade', 0)
        team_fga = team_stats.get('fieldGoalsAttempted', 0)
        team_fta = team_stats.get('freeThrowsAttempted', 0)
        team_tov = team_stats.get('turnovers', 0)
        team_orb = team_stats.get('reboundsOffensive', 0)
        team_drb = team_stats.get('reboundsDefensive', 0)
        team_trb = team_stats.get('reboundsTotal', 0)
        
        # Stats del oponente
        opp_fga = opp_team_stats.get('fieldGoalsAttempted', 0)
        opp_fgm = opp_team_stats.get('fieldGoalsMade', 0)
        opp_tov = opp_team_stats.get('turnovers', 0)
        opp_fta = opp_team_stats.get('freeThrowsAttempted', 0)
        opp_drb = opp_team_stats.get('reboundsDefensive', 0)
        opp_orb = opp_team_stats.get('reboundsOffensive', 0)
        
        # Calcular posesiones
        team_poss = LiveStatsCalculator.calculate_team_possessions(
            team_fga, team_orb, team_tov, team_fta
        )
        opp_poss = LiveStatsCalculator.calculate_team_possessions(
            opp_fga, opp_orb, opp_tov, opp_fta
        )
        
        # ===== ESTADÍSTICAS DE JUGADOR =====
        
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        ts_pct = 0
        if fga > 0 or fta > 0:
            denominator = 2 * (fga + 0.44 * fta)
            ts_pct = round(pts / denominator, 3) if denominator > 0 else 0
        
        # eFG% = (FG + 0.5 * 3PM) / FGA
        efg_pct = 0
        if fga > 0:
            efg_pct = round((fgm + 0.5 * fg3m) / fga, 3)
        
        # 3PAr = 3PA / FGA
        three_par = 0
        if fga > 0:
            three_par = round(fg3a / fga, 3)
        
        # FTr = FTA / FGA
        ftr = 0
        if fga > 0:
            ftr = round(fta / fga, 3)
        
        # ORB% = (ORB * (TeamMin/5)) / (Min * TeamORB)
        orb_pct = 0
        if player_minutes > 0 and team_orb > 0 and team_total_minutes > 0:
            numerator = orb * (team_total_minutes / 5)
            denominator = player_minutes * team_orb
            orb_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # DRB% = (DRB * (TeamMin/5)) / (Min * TeamDRB)
        drb_pct = 0
        if player_minutes > 0 and team_drb > 0 and team_total_minutes > 0:
            numerator = drb * (team_total_minutes / 5)
            denominator = player_minutes * team_drb
            drb_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # TRB% = (ORB+DRB) * (TeamMin/5) / (Min * TeamTRB)
        trb_pct = 0
        if player_minutes > 0 and team_trb > 0 and team_total_minutes > 0:
            total_rb = orb + drb
            numerator = total_rb * (team_total_minutes / 5)
            denominator = player_minutes * team_trb
            trb_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # AST% = (AST * (TeamMin/5)) / (Min * (TeamFG - FG))
        ast_pct = 0
        if player_minutes > 0 and team_total_minutes > 0:
            team_fg_made = team_fgm - fgm
            if team_fg_made > 0:
                numerator = ast * (team_total_minutes / 5)
                denominator = player_minutes * team_fg_made
                ast_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # STL% = (STL * (TeamMin/5)) / (Min * OppPoss)
        stl_pct = 0
        if player_minutes > 0 and team_total_minutes > 0 and opp_poss > 0:
            numerator = stl * (team_total_minutes / 5)
            denominator = player_minutes * opp_poss
            stl_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # BLK% = (BLK * (TeamMin/5)) / (Min * OppFGA)
        blk_pct = 0
        if player_minutes > 0 and team_total_minutes > 0 and opp_fga > 0:
            numerator = blk * (team_total_minutes / 5)
            denominator = player_minutes * opp_fga
            blk_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        tov_pct = 0
        denominator = fga + 0.44 * fta + tov
        if denominator > 0:
            tov_pct = round(tov / denominator, 1)
        
        # USG% = ((FGA + 0.44*FTA + TOV) * (TeamMin/5)) / (Min * TeamPoss)
        usg_pct = 0
        if player_minutes > 0 and team_poss > 0 and team_total_minutes > 0:
            numerator = (fga + 0.44 * fta + tov) * (team_total_minutes / 5)
            denominator = player_minutes * team_poss
            usg_pct = round(numerator / denominator, 1) if denominator > 0 else 0
        
        # ===== ORtg (Offensive Rating for Player) =====
        # Individual Possessions = (FGA - ORB) + TOV + 0.44*FTA
        individual_poss = max((fga - orb) + tov + 0.44 * fta, 1)
        
        team_fg3m = team_stats.get('threePointersMade', 0)
        team_efg = (team_fgm + 0.5 * team_fg3m) / team_fga if team_fga > 0 else 0
        
        points_produced = (
            pts + 
            ftm * 0.5 +
            ast * 0.5 +
            orb * 0.7 -
            (fga - fgm) * team_efg * 0.5 -
            tov * 0.5
        )
        
        player_ortg = round((points_produced / individual_poss) * 100, 1)
        
        # ===== DRtg (Defensive Rating for Player) =====
        # Simplified: estimate based on steals, blocks, defensive rebounds
        opp_fga = opp_team_stats.get('fieldGoalsAttempted', 0)
        opp_fta = opp_team_stats.get('freeThrowsAttempted', 0)
        opp_tov = opp_team_stats.get('turnovers', 0)
        
        opp_poss = opp_fga + 0.44 * opp_fta - opp_orb + opp_tov
        stop_pct = min((stl + blk + drb * 0.5) / max(opp_poss / 5, 1), 0.5)
        
        player_drtg = round(team_drtg - (stop_pct * 20), 1) if team_drtg > 0 else 0
        
        # GmSc = PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA-FT) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
        gmsc = (
            pts 
            + 0.4 * fgm 
            - 0.7 * fga 
            - 0.4 * (fta - ftm) 
            + 0.7 * orb 
            + 0.3 * drb 
            + stl 
            + 0.7 * ast 
            + 0.7 * blk 
            - 0.4 * pf 
            - tov
        )
        
        return {
            'player_id': player_stats.get('personId'),
            'player_name': player_stats.get('name'),
            'minutes': round(player_minutes, 1),
            'points': pts,
            
            # Shooting
            'ts_pct': ts_pct,
            'efg_pct': efg_pct,
            'three_par': three_par,
            'ftr': ftr,
            
            # Rebounds
            'orb_pct': orb_pct,
            'drb_pct': drb_pct,
            'trb_pct': trb_pct,
            
            # Playmaking
            'ast_pct': ast_pct,
            
            # Defense
            'stl_pct': stl_pct,
            'blk_pct': blk_pct,
            
            # Turnovers & Usage
            'tov_pct': tov_pct,
            'usg_pct': usg_pct,
            
            # Ratings
            'ortg': player_ortg,
            'drtg': player_drtg,
            
            # Composite
            'gmsc': round(gmsc, 1),
        }
    
    @staticmethod
    def calculate_team_four_factors(team_stats: Dict, opp_team_stats: Dict, minutes_played: float = 240) -> Dict:
        """
        Calcula Four Factors para un equipo vs su oponente.
        
        Args:
            team_stats: stats del equipo (cuando ataca)
            opp_team_stats: stats del oponente (cuando ataca)
            minutes_played: minutos jugados (default 240 = partido completo)
        
        Basketball-Reference Four Factors:
        - Pace: ritmo del partido (posesiones por 48 min)
        - eFG%: efectividad de tiro
        - TOV%: pérdidas de balón
        - ORB%: rebotes ofensivos
        - FT/FGA: tasa de tiro libre
        - ORtg: rating ofensivo
        """
        
        # ===== TEAM (cuando ataca) =====
        team_fgm = team_stats.get('fieldGoalsMade', 0)
        team_fga = team_stats.get('fieldGoalsAttempted', 0)
        team_fg3m = team_stats.get('threePointersMade', 0)
        team_fta = team_stats.get('freeThrowsAttempted', 0)
        team_ft = team_stats.get('freeThrowsMade', 0)
        team_orb = team_stats.get('reboundsOffensive', 0)
        team_drb = team_stats.get('reboundsDefensive', 0)
        team_trb = team_stats.get('reboundsTotal', 0)
        team_tov = team_stats.get('turnovers', 0)
        team_pts = team_stats.get('points', 0)
        team_ast = team_stats.get('assists', 0)
        
        # ===== OPPONENT (cuando ataca) =====
        opp_fgm = opp_team_stats.get('fieldGoalsMade', 0)
        opp_fga = opp_team_stats.get('fieldGoalsAttempted', 0)
        opp_fg3m = opp_team_stats.get('threePointersMade', 0)
        opp_fta = opp_team_stats.get('freeThrowsAttempted', 0)
        opp_ft = opp_team_stats.get('freeThrowsMade', 0)
        opp_orb = opp_team_stats.get('reboundsOffensive', 0)
        opp_drb = opp_team_stats.get('reboundsDefensive', 0)
        opp_tov = opp_team_stats.get('turnovers', 0)
        opp_pts = opp_team_stats.get('points', 0)
        
        # ===== POSSESSIONS =====
        # Simple formula for ORtg/DRtg (Dean Oliver):
        # Poss = FGA - ORB + TOV + 0.44 * FTA
        team_poss_simple = team_fga - team_orb + team_tov + 0.44 * team_fta
        opp_poss_simple = opp_fga - opp_orb + opp_tov + 0.44 * opp_fta
        
        # ===== COMPLEX FORMULA (BBR-style) =====
        # Basketball-Reference formula: Poss = FGA + 0.4*FTA - 1.07*ORB%*FGmissed + TOV
        # where ORB% = ORB / (ORB + OppDRB)
        # 
        # NOTE: BBR uses slightly different underlying data that results in:
        # - Pacers ORB% = 0.333 (vs 0.295 from NBA API data)
        # - Hawks ORB% = 0.218 (vs 0.279 from NBA API data)
        # This explains the small discrepancies in Pace/ORtg/DRtg
        
        team_orb_ratio = team_orb / (team_orb + opp_drb) if (team_orb + opp_drb) > 0 else 0
        team_fg_missed = team_fga - team_fgm
        team_poss = team_fga + 0.4 * team_fta - 1.07 * team_orb_ratio * team_fg_missed + team_tov
        
        opp_orb_ratio = opp_orb / (opp_orb + team_drb) if (opp_orb + team_drb) > 0 else 0
        opp_fg_missed = opp_fga - opp_fgm
        opp_poss = opp_fga + 0.4 * opp_fta - 1.07 * opp_orb_ratio * opp_fg_missed + opp_tov
        
        # Pace = average of both teams' possessions (using complex formula)
        pace = 0.5 * (team_poss + opp_poss)
        
        # ===== eFG% (del equipo cuando ataca) =====
        team_efg = 0
        if team_fga > 0:
            team_efg = round((team_fgm + 0.5 * team_fg3m) / team_fga, 3)
        
        # ===== TOV% (del equipo cuando ataca) =====
        team_tov_pct = 0
        team_tov_denom = team_fga + 0.44 * team_fta + team_tov
        if team_tov_denom > 0:
            team_tov_pct = round(team_tov / team_tov_denom, 3)
        
        # ===== ORB% (del equipo cuando ataca) =====
        # ORB% = ORB / (ORB + OppDRB)
        # IMPORTANTE: Usa DRB del oponente, no el total
        team_orb_pct = 0
        total_reb_chance = team_orb + opp_drb
        if total_reb_chance > 0:
            team_orb_pct = round(team_orb / total_reb_chance, 3)
        
        # ===== FT/FGA (del equipo cuando ataca) =====
        team_ft_rate = 0
        if team_fga > 0:
            team_ft_rate = round(team_fta / team_fga, 3)
        
        # ===== ORtg (del equipo cuando ataca) =====
        # ORtg = Pts / Poss * 100 (using team's own possessions)
        team_ortg = 0
        if team_poss > 0:
            team_ortg = round((team_pts / team_poss) * 100, 1)
        
        # ===== DRtg (del equipo cuando DEFINE) =====
        # DRtg = Pts que permite / Poss del oponente * 100
        team_drtg = 0
        if opp_poss > 0:
            team_drtg = round((opp_pts / opp_poss) * 100, 1)
        
        return {
            'possessions': round(team_poss, 1),
            'opp_possessions': round(opp_poss, 1),
            'pace': round(pace, 1),
            'efg_pct': team_efg,
            'tov_pct': team_tov_pct,
            'orb_pct': team_orb_pct,
            'ft_rate': team_ft_rate,
            'ortg': team_ortg,
            'drtg': team_drtg,
        }


def parse_nba_api_boxscore(boxscore_json: Dict) -> Dict:
    """
    Parsea el JSON del boxscore de la NBA API y calcula stats avanzadas.
    """
    game = boxscore_json.get('game', {})
    home_team = game.get('homeTeam', {})
    away_team = game.get('awayTeam', {})
    
    home_stats = home_team.get('statistics', {})
    away_stats = away_team.get('statistics', {})
    
    # Calcular Four Factors para ambos
    home_ff = LiveStatsCalculator.calculate_team_four_factors(home_stats, away_stats)
    away_ff = LiveStatsCalculator.calculate_team_four_factors(away_stats, home_stats)
    
    # Calcular stats de jugadores
    calculator = LiveStatsCalculator()
    
    home_players = []
    for player in home_team.get('players', []):
        if player.get('played') != '1':
            continue
        
        player_stats = player.get('statistics', {})
        minutes_str = player_stats.get('minutes', 'PT00M00.00S')
        minutes = calculator.parse_minutes(minutes_str)
        
        # Asumimos 240 minutos totales (5 jugadores × 48 min)
        player_advanced = calculator.calculate_live_player_stats(
            player_stats,
            home_stats,
            away_stats,
            minutes,
            240,
            240,
            home_ff['drtg']
        )
        home_players.append(player_advanced)
    
    away_players = []
    for player in away_team.get('players', []):
        if player.get('played') != '1':
            continue
        
        player_stats = player.get('statistics', {})
        minutes_str = player_stats.get('minutes', 'PT00M00.00S')
        minutes = calculator.parse_minutes(minutes_str)
        
        player_advanced = calculator.calculate_live_player_stats(
            player_stats,
            away_stats,
            home_stats,
            minutes,
            240,
            240,
            away_ff['drtg']
        )
        away_players.append(player_advanced)
    
    return {
        'game_id': game.get('gameId'),
        'game_status': game.get('gameStatus'),
        'period': game.get('period'),
        'game_clock': game.get('gameClock'),
        
        'home_team': {
            'team_id': home_team.get('teamId'),
            'team_name': home_team.get('teamName'),
            'score': home_team.get('score'),
            'four_factors': home_ff,
            'players': home_players,
        },
        
        'away_team': {
            'team_id': away_team.get('teamId'),
            'team_name': away_team.get('teamName'),
            'score': away_team.get('score'),
            'four_factors': away_ff,
            'players': away_players,
        },
    }


def test_with_sample_data():
    """Test con el JSON que proporcionaste."""
    import json
    
    # Tu JSON de ejemplo
    sample_json = {
        "game": {
            "gameId": "0022500020",
            "gameStatus": 3,
            "period": 4,
            "homeTeam": {
                "teamId": 1610612754,
                "teamName": "Pacers",
                "score": 108,
                "statistics": {
                    "fieldGoalsMade": 35,
                    "fieldGoalsAttempted": 99,
                    "threePointersMade": 13,
                    "threePointersAttempted": 47,
                    "freeThrowsMade": 25,
                    "freeThrowsAttempted": 33,
                    "reboundsOffensive": 18,
                    "reboundsDefensive": 31,
                    "reboundsTotal": 49,
                    "assists": 23,
                    "steals": 6,
                    "blocks": 3,
                    "turnovers": 12,
                    "points": 108
                },
                "players": [
                    {"played": "1", "personId": "1628381", "name": "T. Haliburton", "statistics": {
                        "minutes": "PT36M12.00S", "points": 22, "fieldGoalsMade": 8, "fieldGoalsAttempted": 18,
                        "threePointersMade": 4, "threePointersAttempted": 10, "freeThrowsMade": 2, "freeThrowsAttempted": 2,
                        "reboundsOffensive": 1, "reboundsDefensive": 4, "assists": 11, "steals": 2, "blocks": 1, "turnovers": 3, "foulsPersonal": 2
                    }},
                    {"played": "1", "personId": "1630167", "name": "P. Siakam", "statistics": {
                        "minutes": "PT34M05.00S", "points": 18, "fieldGoalsMade": 7, "fieldGoalsAttempted": 15,
                        "threePointersMade": 2, "threePointersAttempted": 5, "freeThrowsMade": 2, "freeThrowsAttempted": 3,
                        "reboundsOffensive": 2, "reboundsDefensive": 6, "assists": 4, "steals": 1, "blocks": 0, "turnovers": 2, "foulsPersonal": 3
                    }},
                    {"played": "1", "personId": "1631102", "name": "M. Turner", "statistics": {
                        "minutes": "PT28M30.00S", "points": 15, "fieldGoalsMade": 6, "fieldGoalsAttempted": 12,
                        "threePointersMade": 3, "threePointersAttempted": 7, "freeThrowsMade": 0, "freeThrowsAttempted": 0,
                        "reboundsOffensive": 3, "reboundsDefensive": 5, "assists": 2, "steals": 0, "blocks": 2, "turnovers": 1, "foulsPersonal": 4
                    }}
                ]
            },
            "awayTeam": {
                "teamId": 1610612737,
                "teamName": "Hawks",
                "score": 128,
                "statistics": {
                    "fieldGoalsMade": 51,
                    "fieldGoalsAttempted": 94,
                    "threePointersMade": 10,
                    "threePointersAttempted": 30,
                    "freeThrowsMade": 16,
                    "freeThrowsAttempted": 20,
                    "reboundsOffensive": 12,
                    "reboundsDefensive": 43,
                    "reboundsTotal": 55,
                    "assists": 30,
                    "steals": 10,
                    "blocks": 5,
                    "turnovers": 10,
                    "points": 128
                },
                "players": [
                    {"played": "1", "personId": "1630029", "name": "T. Young", "statistics": {
                        "minutes": "PT35M45.00S", "points": 32, "fieldGoalsMade": 12, "fieldGoalsAttempted": 22,
                        "threePointersMade": 4, "threePointersAttempted": 8, "freeThrowsMade": 4, "freeThrowsAttempted": 4,
                        "reboundsOffensive": 1, "reboundsDefensive": 2, "assists": 12, "steals": 3, "blocks": 0, "turnovers": 4, "foulsPersonal": 1
                    }},
                    {"played": "1", "personId": "1629648", "name": "D. Murray", "statistics": {
                        "minutes": "PT33M20.00S", "points": 24, "fieldGoalsMade": 10, "fieldGoalsAttempted": 18,
                        "threePointersMade": 2, "threePointersAttempted": 6, "freeThrowsMade": 2, "freeThrowsAttempted": 2,
                        "reboundsOffensive": 2, "reboundsDefensive": 5, "assists": 8, "steals": 2, "blocks": 1, "turnovers": 2, "foulsPersonal": 2
                    }}
                ]
            }
        }
    }
    
    result = parse_nba_api_boxscore(sample_json)
    
    print("=" * 60)
    print("LIVE STATS - FINAL (Partido completo)")
    print("=" * 60)
    
    print(f"\nHome ({result['home_team']['team_name']}): {result['home_team']['score']}")
    print(f"  Four Factors:")
    print(f"    Pace: {result['home_team']['four_factors']['pace']}")
    print(f"    ORtg: {result['home_team']['four_factors']['ortg']}")
    print(f"    DRtg: {result['home_team']['four_factors']['drtg']}")
    print(f"    eFG%: {result['home_team']['four_factors']['efg_pct']}")
    print(f"    TOV%: {result['home_team']['four_factors']['tov_pct']}")
    print(f"    ORB%: {result['home_team']['four_factors']['orb_pct']}")
    
    print(f"\nAway ({result['away_team']['team_name']}): {result['away_team']['score']}")
    print(f"  Four Factors:")
    print(f"    Pace: {result['away_team']['four_factors']['pace']}")
    print(f"    ORtg: {result['away_team']['four_factors']['ortg']}")
    print(f"    DRtg: {result['away_team']['four_factors']['drtg']}")
    print(f"    eFG%: {result['away_team']['four_factors']['efg_pct']}")
    print(f"    TOV%: {result['away_team']['four_factors']['tov_pct']}")
    print(f"    ORB%: {result['away_team']['four_factors']['orb_pct']}")
    
    # Mostrar stats de algunos jugadores
    print("\n" + "=" * 60)
    print("PLAYER STATS (TOP 3)")
    print("=" * 60)
    
    for team_key, team_data in [('home', result['home_team']), ('away', result['away_team'])]:
        print(f"\n{team_data['team_name']}:")
        for p in sorted(team_data['players'], key=lambda x: x.get('points', 0), reverse=True)[:3]:
            print(f"  {p['player_name']}: {p['points']} pts")
            print(f"    TS%: {p['ts_pct']}, eFG%: {p['efg_pct']}, USG%: {p['usg_pct']}")
            print(f"    ORtg: {p['ortg']}, DRtg: {p['drtg']}, GmSc: {p['gmsc']}")
    
    return result


if __name__ == "__main__":
    test_with_sample_data()
