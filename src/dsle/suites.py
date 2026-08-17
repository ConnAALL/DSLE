"""Named benchmark subsets; agents and algorithms do not live in this module."""

DSLE_5 = (
    "asylum_demon",
    "capra_demon",
    "chaos_witch_quelaag",
    "ornstein_and_smough",
    "gwyn_lord_of_cinder",
)

DSLE_MELEE = (
    "asylum_demon",
    "taurus_demon",
    "capra_demon",
    "gaping_dragon",
    "great_grey_wolf_sif",
    "iron_golem",
    "demon_firesage",
    "gwyn_lord_of_cinder",
)

DSLE_RANGED = (
    "moonlight_butterfly",
    "dark_sun_gwyndolin",
    "ceaseless_discharge",
    "seath_the_scaleless",
    "nito",
)

DSLE_MULTI = ("bell_gargoyles", "ornstein_and_smough", "four_kings", "pinwheel")

UNCONFIGURED_DSR_BOSSES: tuple[str, ...] = ()

SUITES = {
    "dsle5": DSLE_5,
    "melee": DSLE_MELEE,
    "ranged": DSLE_RANGED,
    "multi": DSLE_MULTI,
}


def resolve_suite(name: str) -> tuple[str, ...]:
    """Resolve a suite name, with ``full`` loaded from the boss registry."""

    key = str(name).strip().lower()
    if key == "full":
        from dsle.config import list_bosses

        return list_bosses(include_experimental=False)
    try:
        return SUITES[key]
    except KeyError as exc:
        available = ", ".join((*sorted(SUITES), "full"))
        raise KeyError(f"Unknown suite {name!r}. Available: {available}") from exc
