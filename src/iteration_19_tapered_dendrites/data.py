from dataclasses import dataclass

from brian2 import Quantity, farad, meter, ohm, siemens, second
from brian2.units.allunits import ampere
from dataclasses import replace
import numpy as np

def _to_SI(value, unit):
    if value is None:
        return None
    value_in_unit = value / unit

    if np.isscalar(value_in_unit):
        return float(value_in_unit)

    return np.asarray(value_in_unit, dtype=float)

@dataclass(frozen=True)
class NumericalCableParameters:
    """
    Solver representation.
    All values are floats in SI base units.
    """

    c_m: float        # F/m^2
    Rm: float         # ohm*m^2
    gL: float         # S/m^2
    ra: float         # ohm*m

    N: int
    L: float          # m
    dx: float         # m
    tau: float        # s
    r0: float         # m
    b: float          # m^2/s

    I_e: float        # A
    x: np.ndarray | None = None

    def __str__(self):
        return (
            "NumericalCableParameters (SI)\n"
            f"  c_m = {self.c_m:.4e} F/m²\n"
            f"  Rm  = {self.Rm:.4e} Ω·m²\n"
            f"  ra  = {self.ra:.4e} Ω·m\n"
            f"  N   = {self.N}\n"
            f"  L   = {self.L:.4e} m\n"
            f"  dx  = {self.dx:.4e} m\n"
            f"  tau = {self.tau:.4e} s\n"
            f"  b   = {self.b:.4e} m²/s\n"
            f"  Ie  = {self.I_e:.4e} A"
        )
@dataclass(frozen=True)
class CableParameters:

    N: int = 101
    c_m: Quantity | None = None
    Rm: Quantity | None = None
    gL: Quantity | None = None
    ra: Quantity | None = None

    L: Quantity | None = None
    dx: Quantity | None = None

    tau: Quantity | None = None
    r0: Quantity | None = None
    b: Quantity | None = None

    I_e: Quantity | None = None

    x: np.ndarray | None = None

    def __str__(self):
        s = (
            "CableParameters:\n"
            f"  c_m = {self.c_m}\n"
            f"  Rm  = {self.Rm}\n"
            f"  ra  = {self.ra}\n"
            f"  L   = {self.L}\n"
            f"  N   = {self.N}\n"
            f"  dx  = {self.dx}\n"
            f"  tau = {self.tau}\n"
            f"  r0  = {self.r0}\n"
            f"  b   = {self.b}\n"
            f"  I_e = {self.I_e}\n"
        )

        if self.x is not None:
            s += f"  x   = [{self.x[0]} -> {self.x[-1]}] ({len(self.x)} points)\n"

        return s

    def __post_init__(self):

        if self.Rm is not None and self.gL is None:
            object.__setattr__(
                self,
                "gL",
                1 / self.Rm
            )

        if (
                self.c_m is not None
                and self.gL is not None
                and self.tau is None
        ):
            object.__setattr__(
                self,
                "tau",
                self.c_m / self.gL
            )

        if self.x is None and self.L is not None and self.N is not None:
            object.__setattr__(
                self,
                "x",
                np.linspace(0, float(self.L / meter), self.N) * meter
            )

        if (
                self.L is not None
                and self.N is not None
                and self.dx is None
        ):
            object.__setattr__(
                self,
                "dx",
                self.L / (self.N - 1)
            )

        if (
                self.r0 is not None
                and self.c_m is not None
                and self.ra is not None
                and self.b is None
        ):
            object.__setattr__(
                self,
                "b",
                self.r0 / (2 * self.c_m * self.ra)
            )

    def to_numerical(self):
        return NumericalCableParameters(
            c_m=_to_SI(self.c_m, farad / meter ** 2),
            Rm=_to_SI(self.Rm, ohm * meter ** 2),
            gL=_to_SI(self.gL, siemens / meter ** 2),
            ra=_to_SI(self.ra, ohm * meter),
            N=self.N,
            L=_to_SI(self.L, meter),
            dx=_to_SI(self.dx, meter),
            x=_to_SI(self.x, meter),
            tau=_to_SI(self.tau, second),
            r0=_to_SI(self.r0, meter),

            b=_to_SI(self.b, meter ** 2 / second),

            I_e=_to_SI(self.I_e, ampere)
        )

    @classmethod
    def from_numerical(cls, other: NumericalCableParameters):
        """
        Construct CableParameters from a NumericalCableParameters object.
        All numerical values are assumed to be in SI units.
        """
        return cls.from_SI(
            c_m=other.c_m,
            Rm=other.Rm,
            gL=other.gL,
            ra=other.ra,
            L=other.L,
            N=other.N,
            dx=other.dx,
            x=other.x,
            tau=other.tau,
            r0=other.r0,
            b=other.b,
            I_e=other.I_e,
        )

    @classmethod
    def from_SI(
            cls,
            *,
            c_m=None,  # F/m^2
            Rm=None,  # ohm*m^2
            gL=None,  # S/m^2
            ra=None,  # ohm*m

            L=None,  # m
            N=101,

            x=None,
            dx=None,  # m

            tau=None,  # s
            r0=None,  # m

            b=None,  # m^2/s

            I_e=None  # A
    ):

        # Attach Brian2 units
        c_m = None if c_m is None else c_m * farad / meter ** 2
        Rm = None if Rm is None else Rm * ohm * meter ** 2
        gL = None if gL is None else gL * siemens / meter ** 2
        ra = None if ra is None else ra * ohm * meter

        L = None if L is None else L * meter
        x = None if x is None else np.asarray(x) * meter
        dx = None if dx is None else dx * meter

        tau = None if tau is None else tau * second
        r0 = None if r0 is None else r0 * meter

        b = None if b is None else b * meter ** 2 / second

        I_e = None if I_e is None else I_e * ampere

        # Derived quantities
        if gL is None and Rm is not None:
            gL = 1 / Rm

        if tau is None and c_m is not None and gL is not None:
            tau = c_m / gL

        if dx is None and L is not None and N is not None:
            dx = L / (N - 1)

        if b is None and r0 is not None and c_m is not None and ra is not None:
            b = r0 / (2 * c_m * ra)

        return cls(
            c_m=c_m,
            Rm=Rm,
            gL=gL,
            ra=ra,

            L=L,
            N=N,
            x=x,
            dx=dx,

            tau=tau,
            r0=r0,
            b=b,

            I_e=I_e
        )

    def with_SI_properties(self, **changes):

        converted = {}

        for key, value in changes.items():

            if value is None:
                converted[key] = None

            elif key == "c_m":
                converted[key] = value * farad / meter ** 2

            elif key == "Rm":
                converted[key] = value * ohm * meter ** 2

            elif key == "gL":
                converted[key] = value * siemens / meter ** 2

            elif key == "ra":
                converted[key] = value * ohm * meter

            elif key == "L":
                converted[key] = value * meter

            elif key == "dx":
                converted[key] = value * meter

            elif key == "tau":
                converted[key] = value * second

            elif key == "r0":
                converted[key] = value * meter

            elif key == "b":
                converted[key] = value * meter ** 2 / second

            elif key == "I_e":
                converted[key] = value * ampere

            elif key == "N":
                converted[key] = value

            else:
                raise ValueError(
                    f"Unknown cable parameter '{key}'"
                )

        values = self.__dict__.copy()
        values.update(converted)

        # Recompute derived quantities
        if "Rm" in converted or "gL" in converted:
            if values["Rm"] is not None:
                values["gL"] = 1 / values["Rm"]

        if (
                values["c_m"] is not None
                and values["gL"] is not None
        ):
            values["tau"] = (
                    values["c_m"] /
                    values["gL"]
            )

        if (
                values["L"] is not None
                and values["N"] is not None
        ):
            values["dx"] = (
                    values["L"] /
                    (values["N"] - 1)
            )

        if (
                values["r0"] is not None
                and values["c_m"] is not None
                and values["ra"] is not None
        ):
            values["b"] = (
                    values["r0"]
                    /
                    (2 * values["c_m"] * values["ra"])
            )

        return replace(self, **values)