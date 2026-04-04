"""Hierarchical names."""

class Name:
    __slots__ = ('parts',)
    def __init__(self, *parts): self.parts = parts
    @staticmethod
    def from_str(s): return Name(*s.split('.'))
    @staticmethod
    def anon(): return Name()
    def is_anon(self): return len(self.parts) == 0
    def __repr__(self): return '.'.join(self.parts) if self.parts else '[anon]'
    def __eq__(self, o): return isinstance(o, Name) and self.parts == o.parts
    def __hash__(self): return hash(self.parts)
