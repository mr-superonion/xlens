
def _resolve_flux_min(
    flux_min: float | dict,
    bands: str,
) -> dict[str, float]:
    """Return per-band flux_min as dict{band: value}."""
    if isinstance(flux_min, dict):
        return {b: float(flux_min[b]) for b in bands}
    else:
        f = float(flux_min)
        return {b: f for b in bands}


def _resolve_flux_name(flux_name: str) -> str:
    """Normalize flux_name to column suffix ('' or f'_{name}')."""
    if len(flux_name) > 0:
        if flux_name[0] != "_":
            fn = "_" + flux_name
        else:
            fn = flux_name
    else:
        fn = ""
    return fn
