class ExtendedDict(dict):
    """Read-only dict with attribute-style access (e.g. d.key). Immutable after construction."""
    __getattr__ = dict.get