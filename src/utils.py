class ExtendedDict(dict):
    """Read-only dict with attribute-style access (e.g. d.key). Immutable after construction."""

    __getattr__ = dict.get

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_frozen", False)
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if name == "_frozen":
            object.__setattr__(self, name, value)
            return
        if getattr(self, "_frozen", False):
            raise TypeError("ExtendedDict is immutable")
        dict.__setitem__(self, name, value)

    def __setitem__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError("ExtendedDict is immutable")
        dict.__setitem__(self, key, value)