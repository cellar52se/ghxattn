from __future__ import annotations

VF_LOCATIONS = 52
HORIZON_YEARS = 2.0
MCID_SLOPE = -0.5
MCID_MAGNITUDE = 0.5

SITES: tuple[str, ...] = ("UWHVF", "HarvardGDP", "GRAPE")
VENDORS: tuple[str, ...] = ("Cirrus", "Spectralis", "Topcon")
RACES: tuple[str, ...] = ("Asian", "Black", "White")

BUNDLE_NAMES: tuple[str, ...] = ("MAC", "SAR", "IAR", "SNA", "INA", "TEM", "PER")
BUNDLE_SECTORS: dict[str, int] = {
    "MAC": 8,
    "SAR": 12,
    "IAR": 12,
    "SNA": 6,
    "INA": 6,
    "TEM": 4,
    "PER": 4,
}

BUNDLE_ADJACENCY: frozenset[tuple[int, int]] = frozenset(
    {
        (0, 1),
        (0, 6),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (5, 0),
        (5, 6),
    }
)

SITE_TO_VENDORS: dict[str, tuple[str, ...]] = {
    "UWHVF": (),
    "HarvardGDP": ("Cirrus", "Spectralis"),
    "GRAPE": ("Topcon",),
}


def _expand_bundle_ids(counts: tuple[int, ...], total: int) -> list[int]:
    base = sum(counts)
    ids: list[int] = []
    for bundle_index, count in enumerate(counts):
        scaled = max(1, round(count * total / base))
        ids.extend([bundle_index] * scaled)
    if len(ids) < total:
        ids.extend([len(counts) - 1] * (total - len(ids)))
    return ids[:total]


def vf_bundle_ids() -> list[int]:
    counts = tuple(BUNDLE_SECTORS[name] for name in BUNDLE_NAMES)
    ids: list[int] = []
    for bundle_index, count in enumerate(counts):
        ids.extend([bundle_index] * count)
    return ids


def structural_bundle_ids(n_oct: int) -> list[int]:
    counts = tuple(BUNDLE_SECTORS[name] for name in BUNDLE_NAMES)
    return _expand_bundle_ids(counts, n_oct)


def is_adjacent(a: int, b: int) -> bool:
    if a == b:
        return False
    return (a, b) in BUNDLE_ADJACENCY or (b, a) in BUNDLE_ADJACENCY


def site_index(site: str) -> int:
    return SITES.index(site)


def vendor_index(vendor: str | None) -> int:
    if vendor is None:
        return -1
    return VENDORS.index(vendor)


def race_index(race: str) -> int:
    return RACES.index(race)
