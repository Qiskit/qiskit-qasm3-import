# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may obtain a copy of this license
# in the LICENSE.txt file in the root directory of this source tree or at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this copyright notice, and modified
# files need to carry a notice indicating that they have been altered from the originals.

import abc
import typing


class Type(abc.ABC):
    """An internal representation of the OpenQASM 3 type system values, or at least that parts of it
    that Terra has some sort of support for.  The reference AST does not have a nice unified object
    to use here; there is :class:`~openqasm3.ast.ClassicalType` but no quantum equivalent (since
    it's implicit).

    This class is just a typing base, and should never be instantiated.  Its subclasses will be,
    though."""

    __slots__ = ()

    @abc.abstractmethod
    def pretty(self) -> str:
        """A pretty string representation of the type, useful for debugging."""

    def __eq__(self, other):
        return type(self) is type(other) and all(
            getattr(self, slot) == getattr(other, slot) for slot in self.__slots__
        )

    def __repr__(self):
        return f"<type '{self.pretty()}'>"


@typing.final
class Error(Type):
    """A zero type that represents an error during type checking."""

    __slots__ = ()

    def pretty(self):
        return "<type error>"


@typing.final
class Never(Type):
    """The bottom type.  There are no valid values of this type, as it can never be instantiated.
    This is used during inference in cases where multiple types must combined into their join, but
    some of the elements have missing values, such as a range that has a start but no stop value."""

    __slots__ = ()

    def pretty(self):
        return "!"


@typing.final
class BitArray(Type):
    """An array of bits.  This roughly corresponds to Terra's
    :class:`~qiskit.circuit.ClassicalRegister`."""

    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = size

    def pretty(self):
        return f"bit[{self.size}]"


@typing.final
class QubitArray(Type):
    """An array of qubits.  This roughly corresponds to Terra's
    :class:`~qiskit.circuit.QuantumRegister`."""

    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = size

    def pretty(self):
        return f"qubit[{self.size}]"


@typing.final
class Bit(Type):
    """A single bit.  This corresponds to Terra's :class:`~qiskit.circuit.Clbit`."""

    __slots__ = ()

    def pretty(self):
        return "bit"


@typing.final
class Bool(Type):
    """A Boolean value.  This is only used by Terra in for single-bit conditions."""

    __slots__ = ("const",)

    def __init__(self, const: bool):
        self.const = const

    def pretty(self):
        return "const bool" if self.const else "bool"


@typing.final
class Qubit(Type):
    """A single qubit.  This corresponds to Terra's :class:`~qiskit.circuit.Qubit`."""

    __slots__ = ()

    def pretty(self):
        return "qubit"


@typing.final
class HardwareQubit(Type):
    """A hardware qubit.
    This corresponds a hardware qubits referenced in Terra's :class:`~qiskit.transpiler.TranspilerLayout`."""

    __slots__ = ()

    def pretty(self):
        return "hardware qubit"


@typing.final
class Int(Type):
    """An integer value.  This is generally only encountered as a constant and so is represented by
    a Python integer, but can also be the type of the Qiskit :class:`~qiskit.circuit.Parameter` used
    to represent ``for``-loop variables."""

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
    """An unsigned integer value."""

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
    """A floating-point type.  Terra can use this either in a constant form as a Python ``float``,
    or as a :class:`~qiskit.circuit.Parameter`."""

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
    """An angle type.  OpenQASM 3 makes a large distinction between ``angle`` and ``float`` (the
    OpenQASM angle is integer-like), but Terra currently treats them as interchangeable.  This might
    change in the future."""

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
    """A duration.  Right now, this is only recognised in constant form, which is represented within
    :class:`.ValueResolver` as a 2-tuple of a float and its unit."""

    __slots__ = ("const",)

    def __init__(self, const: bool):
        self.const = const

    def pretty(self):
        return "const duration" if self.const else "duration"


@typing.final
class Range(Type):
    """A range selector.  The inner type is the join of the types of the start and end values."""

    __slots__ = ("base",)

    def __init__(self, base: Type):
        self.base = base

    def pretty(self):
        return f"range[{self.base.pretty()}]"


@typing.final
class Sequence(Type):
    """A general sequence of values.  This is represented internally as a list or tuple of the
    contained type."""

    __slots__ = ("base",)

    def __init__(self, base: Type):
        self.base = base

    def pretty(self):
        return f"sequence[{self.base.pretty()}]"


@typing.final
class Gate(Type):
    """The type of a gate.  Since the classical parameters of gates have a fixed type in OpenQASM 3,
    this just stores the counts of the classical and quantum arguments."""

    __slots__ = ("n_classical", "n_quantum")

    def __init__(self, n_classical: int, n_quantum: int):
        self.n_classical = n_classical
        self.n_quantum = n_quantum

    def pretty(self):
        return f"gate[{self.n_classical}, {self.n_quantum}]"
