import math
import numpy as np
import typing
from typing import Literal, Sequence, Optional, Union
import numpy.typing as npt
from decimal import Decimal
from fractions import Fraction


_legalNumType = [Decimal, Fraction, float, int]
LegalNumType = Union[Decimal, Fraction, float, int]

class Multi3D:
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    @staticmethod
    def cross(v1: Sequence, v2: Sequence) -> np.ndarray:
        return np.array([
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ])

    @staticmethod
    def dot(v1: Sequence, v2: Sequence) -> float:
        assert len(v1) == len(v2)
        return sum(_a * _b for _a, _b in zip(v1, v2))

    @staticmethod
    def triple(v1: Sequence, v2: Sequence, v3: Sequence) -> float:
        return dot(v1, cross(v2, v3))

    @staticmethod
    def parallelepiped_volume(v1: Sequence, v2: Sequence, v3: Sequence) -> float:
        return abs(Multi3D.triple(v1, v2, v3))

    @ staticmethod
    def execute():
        print(
            ""
        )

    @staticmethod
    def vector(*args: list, dim=3, arr: Optional[npt.ArrayLike] = None) -> np.ndarray:
        if arr is not None:
            return np.array(arr)
        if len(args) > 0:
            return np.array(args)
        return np.array([0] * dim)

    @staticmethod
    def plane_equation(v1: Sequence, v2: Sequence, v3: Sequence, out=True) -> Optional[str]:
        normal_vector = Multi3D.cross((v2 - v1), (v3 - v2))
        if out:
            print(msg1:=f"{normal_vector = }")
        x, y, z = v2
        a, b, c = normal_vector
        msg2 = f"Equation: {a}(x - {x}) + {b}(y - {y}) + {c}(z - {z})"
        d = a * x + b * y + c * z
        d *= -1
        msg3 = f"Standard Form: {a}x + {b}y + {c}z + {d} = 0"
        if out:
            print(msg2)
            print(msg3)
        else:
            return msg1 + "\n" + msg2 + "\n" + msg3

    @staticmethod
    def magnitude(vec: npt.ArrayLike):
        return np.sqrt(sum(num ** 2 for num in vec))


    

class BaseMathExpression:
    _content: Union[str, LegalNumType]
    _mode: type[LegalNumType]
    _special_constant_mode: bool = False
    isNum: bool
    val: LegalNumType

    def __init__(
        self, 
        content: Union[str, LegalNumType], 
        mode: type[LegalNumType] = float,
        special_constant_mode = False
    ) -> None:
        self._content = content
        self._mode = mode
        self._special_constant_mode = special_constant_mode

    def __float__(self) -> float:
        if self._special_constant_mode:
            if self._content.lower() == "e":
                return math.e
            elif self._content.lower() == "pi":
                return math.pi
            else:
                raise ValueError("Unsupported Special Index")
        try: 
            return float(self._content)
        except ValueError:
            return 0

    def __str__(self) -> str:
        return str(float(self))

    @property
    def val(self) -> LegalNumType:
        if self._special_constant_mode:
            return float(self)
        if self._mode is float:
            return float(self)
        elif self._mode in _legalNumType:
            return self._content
        return 0.

    @property
    def isNum(self) -> bool:
        if self._special_constant_mode:
            return True
        try: 
            float(self._content)
        except ValueError:
            return False
        return True

    def __add__(self, other: "BaseMathExpression"):
        if self.isNum:
            if other.isNum:
                new_content = self.val + other.val
                return BaseMathExpression(new_content, type(new_content), False)
        return NotImplemented

cross = Multi3D.cross
dot = Multi3D.dot
i = Multi3D.i
j = Multi3D.j
k = Multi3D.k
vector = vec = Multi3D.vector
triple = Multi3D.triple
plane_equation = Multi3D.plane_equation

test = True
if __name__ == "__main__" and test:
    a = BaseMathExpression("e", float, True)
    print(a._content)
    print(a.isNum)
    print(a.val)
    print(Multi3D.dot([1,2,3], [4,5,6]))




p = vector(-2, 4, -1)
v = vec(2,-4,1)
q = vec(4, 1,0)

plane_equation(p, v, q)

print(
    cross(vec(1,2,3), vec(1,2,6)) / Multi3D.magnitude(vec(1,2,3))
)