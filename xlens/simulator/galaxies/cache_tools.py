import functools

from astropy.table import Table


@functools.lru_cache(maxsize=8)
def cached_catalog_read(fname):
    return Table.read(fname).as_array()
