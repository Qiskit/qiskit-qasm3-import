import abc
import typing


class Type(abc.ABC):
    """An internal representation of the OpenQASM 3 type system values, or at least that parts of it
    that we have some sort of support for.  The reference AST does not have a nice unified object to
    use here; there is :class:`~openqasm3.ast.ClassicalType` but no quantum equivalent (since it's
    implicit).

    This class is just a typing base, and should never be instantiated.  Its subclasses will be,
    though."""

    __slots__ = ()

    @abc.abstractmethod
    def pretty(self) -> str:
        pass

    def __eq__(self, other):
        return type(self) is type(other) and all(
            getattr(self, slot) == getattr(other, slot) for slot in self.__slots__
        )

    def __repr__(self):
        return f"<type '{self.pretty()}'>"


@typing.final
class Error(Type):
    __slots__ = ()

    def pretty(self):
        return "<type error>"


@typing.final
class Never(Type):
    __slots__ = ()

    def pretty(self):
        return "!"


@typing.final
class BitArray(Type):
    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = size

    def pretty(self):
        return f"bit[{self.size}]"


@typing.final
class QubitArray(Type):
    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = size

    def pretty(self):
        return f"qubit[{self.size}]"


@typing.final
class Bit(Type):
    __slots__ = ()

    def pretty(self):
        return "bit"


@typing.final
class Bool(Type):
    __slots__ = ("const",)

    def __init__(self, const: bool):
        self.const = const

    def pretty(self):
        return "const bool" if self.const else "bool"


@typing.final
class Qubit(Type):
    __slots__ = ()

    def pretty(self):
        return "qubit"


@typing.final
class Int(Type):
    __slots__ = ("size", "const")

    def __init__(self, const: bool = False, size: typing.Optional[int] = None):
        self.const = const
        self.size = size

    def pretty(self):
        return "".join(
            [
                "const " if self.const else "",
                "int",
                f"[{self.size}]" if self.size is not None else "",
            ]
        )


@typing.final
class Uint(Type):
    __slots__ = ("size", "const")

    def __init__(self, const: bool = False, size: typing.Optional[int] = None):
        self.const = const
        self.size = size

    def pretty(self):
        return "".join(
            [
                "const " if self.const else "",
                "uint",
                f"[{self.size}]" if self.size is not None else "",
            ]
        )


@typing.final
class Float(Type):
    __slots__ = ("size", "const")

    def __init__(self, const: bool = False, size: typing.Optional[int] = None):
        self.const = const
        self.size = size

    def pretty(self):
        return "".join(
            [
                "const " if self.const else "",
                "float",
                f"[{self.size}]" if self.size is not None else "",
            ]
        )


@typing.final
class Angle(Type):
    __slots__ = ("size", "const")

    def __init__(self, const: bool = False, size: typing.Optional[int] = None):
        self.const = const
        self.size = size

    def pretty(self):
        return "".join(
            [
                "const " if self.const else "",
                "angle",
                f"[{self.size}]" if self.size is not None else "",
            ]
        )


@typing.final
class Duration(Type):
    __slots__ = ("const",)

    def __init__(self, const: bool):
        self.const = const

    def pretty(self):
        return "const duration" if self.const else "duration"


@typing.final
class Range(Type):
    __slots__ = ("base",)

    def __init__(self, base: Type):
        self.base = base

    def pretty(self):
        return f"range[{self.base.pretty()}]"


@typing.final
class Sequence(Type):
    __slots__ = ("base",)

    def __init__(self, base: Type):
        self.base = base

    def pretty(self):
        return f"sequence[{self.base.pretty()}]"


@typing.final
class Gate(Type):
    __slots__ = ("n_classical", "n_quantum")

    def __init__(self, n_classical: int, n_quantum: int):
        self.n_classical = n_classical
        self.n_quantum = n_quantum

    def pretty(self):
        return f"gate[{self.n_classical}, {self.n_quantum}]"
