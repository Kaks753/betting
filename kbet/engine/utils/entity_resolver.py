"""
KBet Entity Resolver — The Glue
Maps all team name variations across data sources to a unified internal UUID.

"Man City" = "Manchester City FC" = "Manchester City" = one UUID.
This must be built FIRST before any joins. Silent mismatches corrupt everything.
"""

import os
import sys
import uuid
import json
import difflib
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_PROCESSED_DIR


REGISTRY_PATH = os.path.join(DATA_PROCESSED_DIR, "entity_registry.json")

# ─── Canonical name aliases ────────────────────────────────────────────────────
# Format: "canonical_name": ["alias1", "alias2", ...]
KNOWN_ALIASES = {
    "Manchester City":      ["Man City", "Manchester City FC", "Man. City", "MCFC"],
    "Manchester United":    ["Man United", "Man Utd", "Manchester Utd", "MUFC", "Man. Utd"],
    "Arsenal":              ["Arsenal FC"],
    "Liverpool":            ["Liverpool FC"],
    "Chelsea":              ["Chelsea FC"],
    "Tottenham":            ["Spurs", "Tottenham Hotspur", "Tottenham FC"],
    "Newcastle":            ["Newcastle United", "Newcastle Utd", "NUFC"],
    "Aston Villa":          ["Aston Villa FC"],
    "West Ham":             ["West Ham United", "West Ham Utd"],
    "Everton":              ["Everton FC"],
    "Leicester":            ["Leicester City", "Leicester City FC"],
    "Brighton":             ["Brighton & Hove Albion", "Brighton and Hove Albion"],
    "Brentford":            ["Brentford FC"],
    "Fulham":               ["Fulham FC"],
    "Wolves":               ["Wolverhampton", "Wolverhampton Wanderers", "Wolverhampton W"],
    "Crystal Palace":       ["Crystal Palace FC"],
    "Nottm Forest":         ["Nottingham Forest", "Nott'm Forest"],
    "Burnley":              ["Burnley FC"],
    "Luton":                ["Luton Town", "Luton Town FC"],
    "Sheffield United":     ["Sheffield Utd", "Sheff United"],
    "Leeds":                ["Leeds United", "Leeds United FC", "Leeds Utd"],
    "Barcelona":            ["FC Barcelona", "Barca"],
    "Real Madrid":          ["Real Madrid CF"],
    "Atletico Madrid":      ["Atl. Madrid", "Atletico de Madrid", "Atlético Madrid"],
    "Sevilla":              ["Sevilla FC"],
    "Valencia":             ["Valencia CF"],
    "Villarreal":           ["Villarreal CF"],
    "Real Sociedad":        ["Real Sociedad CF"],
    "Athletic Bilbao":      ["Athletic Club", "Athletic Club Bilbao"],
    "Getafe":               ["Getafe CF"],
    "Bayern Munich":        ["Bayern München", "FC Bayern München", "Bayern Munchen"],
    "Borussia Dortmund":    ["Dortmund", "BVB"],
    "RB Leipzig":           ["Leipzig", "RasenBallsport Leipzig"],
    "Bayer Leverkusen":     ["Leverkusen"],
    "Eintracht Frankfurt":  ["Frankfurt"],
    "Wolfsburg":            ["VfL Wolfsburg"],
    "Freiburg":             ["SC Freiburg"],
    "Juventus":             ["Juve", "Juventus FC"],
    "Inter":                ["Inter Milan", "FC Internazionale", "Internazionale"],
    "AC Milan":             ["Milan"],
    "Napoli":               ["SSC Napoli"],
    "Roma":                 ["AS Roma"],
    "Lazio":                ["SS Lazio"],
    "Atalanta":             ["Atalanta BC"],
    "PSG":                  ["Paris Saint-Germain", "Paris SG", "Paris Saint Germain"],
    "Marseille":            ["Olympique Marseille", "OM"],
    "Lyon":                 ["Olympique Lyonnais"],
    "Monaco":               ["AS Monaco"],
    "Lille":                ["LOSC Lille", "LOSC"],
    "Ajax":                 ["Ajax Amsterdam", "AFC Ajax"],
    "PSV":                  ["PSV Eindhoven"],
    "Feyenoord":            ["Feyenoord Rotterdam"],
    "Benfica":              ["SL Benfica"],
    "Porto":                ["FC Porto"],
    "Sporting":             ["Sporting CP", "Sporting Lisbon"],
}


class EntityRegistry:
    """
    Maintains a bidirectional map:
      name_variant → canonical_uuid
      canonical_uuid → {canonical_name, all_variants, metadata}
    """

    def __init__(self):
        self.registry = {}       # uuid → {canonical, variants, league, ...}
        self.name_to_uuid = {}   # any_name_lower → uuid
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r") as f:
                data = json.load(f)
            self.registry     = data.get("registry", {})
            self.name_to_uuid = data.get("name_to_uuid", {})
        else:
            self._seed_from_known_aliases()

    def _seed_from_known_aliases(self):
        """Pre-populate with known aliases."""
        for canonical, aliases in KNOWN_ALIASES.items():
            team_uuid = str(uuid.uuid4())
            self.registry[team_uuid] = {
                "canonical": canonical,
                "variants":  [canonical] + aliases,
                "league":    None,
                "country":   None,
            }
            for variant in [canonical] + aliases:
                self.name_to_uuid[variant.lower().strip()] = team_uuid
        self.save()

    def save(self):
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(
                {"registry": self.registry, "name_to_uuid": self.name_to_uuid},
                f, indent=2
            )

    def resolve(self, name: str, fuzzy_threshold: float = 0.82) -> str:
        """
        Resolve a team name to its UUID.
        1. Exact match (case-insensitive)
        2. Fuzzy match above threshold
        3. Auto-register if new team
        Returns UUID string.
        """
        key = name.lower().strip()

        # 1. Exact match
        if key in self.name_to_uuid:
            return self.name_to_uuid[key]

        # 2. Fuzzy match
        all_known = list(self.name_to_uuid.keys())
        matches = difflib.get_close_matches(key, all_known, n=1, cutoff=fuzzy_threshold)
        if matches:
            matched_uuid = self.name_to_uuid[matches[0]]
            # Register this variant so we don't fuzzy-match again
            self.name_to_uuid[key] = matched_uuid
            self.registry[matched_uuid]["variants"].append(name)
            self.save()
            return matched_uuid

        # 3. Auto-register as new team
        new_uuid = str(uuid.uuid4())
        self.registry[new_uuid] = {
            "canonical": name,
            "variants":  [name],
            "league":    None,
            "country":   None,
        }
        self.name_to_uuid[key] = new_uuid
        self.save()
        return new_uuid

    def canonical_name(self, team_uuid: str) -> str:
        """Return canonical name for a UUID."""
        return self.registry.get(team_uuid, {}).get("canonical", team_uuid)

    def resolve_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add home_uuid and away_uuid columns to a matches DataFrame.
        Also normalizes team names to canonical form.
        """
        df = df.copy()
        df["home_uuid"] = df["home_team"].apply(self.resolve)
        df["away_uuid"] = df["away_team"].apply(self.resolve)
        df["home_team_canonical"] = df["home_uuid"].apply(self.canonical_name)
        df["away_team_canonical"] = df["away_uuid"].apply(self.canonical_name)
        return df

    def get_summary(self) -> dict:
        return {
            "total_teams": len(self.registry),
            "total_variants": len(self.name_to_uuid),
            "avg_variants_per_team": round(len(self.name_to_uuid) / max(len(self.registry), 1), 1),
        }


# Global singleton
_registry_instance = None

def get_registry() -> EntityRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EntityRegistry()
    return _registry_instance


if __name__ == "__main__":
    reg = get_registry()
    summary = reg.get_summary()
    print(f"\n{'─'*50}")
    print(f"  ENTITY REGISTRY SUMMARY")
    print(f"{'─'*50}")
    print(f"  Total teams:       {summary['total_teams']}")
    print(f"  Total variants:    {summary['total_variants']}")
    print(f"  Avg variants/team: {summary['avg_variants_per_team']}")

    # Test resolutions
    tests = ["Man City", "Manchester City FC", "Barca", "Bayern München", "Nott'm Forest"]
    print(f"\n  RESOLUTION TESTS:")
    for t in tests:
        uid = reg.resolve(t)
        canonical = reg.canonical_name(uid)
        print(f"  '{t}' → '{canonical}' ({uid[:8]}...)")
    print(f"{'─'*50}\n")
