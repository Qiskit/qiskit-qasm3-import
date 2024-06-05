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

import re
import enum
from typing import Optional, Union
import itertools
import math

from qiskit.circuit import QuantumCircuit, Parameter

from qiskit.circuit.library import standard_gates as _std

from . import types
from .data import Scope, Symbol
from .exceptions import raise_from_node


_BUILTINS = {
    "U": Symbol(
        "U",
        _std.UGate,
        types.Gate([types.Angle() for _ in [None] * 3], [types.Qubit()]),
        Scope.BUILTIN,
    ),
    "pi": Symbol("pi", math.pi, types.Float(const=True), Scope.BUILTIN),
    "π": Symbol("π", math.pi, types.Float(const=True), Scope.BUILTIN),
    "tau": Symbol("tau", math.tau, types.Float(const=True), Scope.BUILTIN),
    "τ": Symbol("τ", math.tau, types.Float(const=True), Scope.BUILTIN),
    "euler": Symbol("euler", math.e, types.Float(const=True), Scope.BUILTIN),
    "ℇ": Symbol("ℇ", math.e, types.Float(const=True), Scope.BUILTIN),
}


_PHYSICAL_QUBIT_RE = re.compile(r"\$(?P<index>\d+)")


def physical_qubit_index(name: Union[str, Symbol]) -> Optional[int]:
    """If this name is a physical qubit, return its integer index.  If not, return ``None``."""
    if isinstance(name, Symbol):
        name = name.name
    if match := _PHYSICAL_QUBIT_RE.fullmatch(name):
        return int(match["index"])
    return None


def add_dummy_parameter_reference(circuit: QuantumCircuit, parameter: Parameter):
    """Ensure that a circuit contains at least one reference to a given parameter."""
    # TODO: this is a hack because Terra doesn't have any public way to add a parameter with
    # no uses to a circuit, but we need to ensure that things work in later bindings if
    # they're not all there.  This uses the fact that `parameter - parameter` is a
    # `ParameterExpression` representation of zero that still tracks that it was once
    # parametric over `parameter`.
    circuit.global_phase += parameter - parameter


class AddressingMode:
    """Addressing mode for qubits in OpenQASM 3 programs.

    This class is useful as long as we allow only physical or virtual addressing modes, but not
    mixed. If the latter is supported in the future, this class will be modified or removed.
    """

    _Mode = enum.Enum("_Mode", ["UNKNOWN", "PHYSICAL", "VIRTUAL"])
    (_UNKNOWN, _PHYSICAL, _VIRTUAL) = (_Mode.UNKNOWN, _Mode.PHYSICAL, _Mode.VIRTUAL)

    def __init__(self):
        self._state = self._UNKNOWN

    def set_physical_mode(self, node):
        """Set the addressing mode to physical. On success return `True`, otherwise raise an
        exception."""
        if self._state is self._PHYSICAL:  # Fast exit for most common case
            return
        if self._state is self._VIRTUAL:
            raise_from_node(
                node,
                "Physical qubit referenced in virtual addressing mode. Mixing "
                "modes not currently supported.",
            )
        self._state = self._PHYSICAL

    def set_virtual_mode(self, node):
        """Set the addressing mode to virtual.
        On success return `True`, otherwise raise an exception."""
        if self._state is self._VIRTUAL:
            return
        if self._state is self._PHYSICAL:
            raise_from_node(
                node,
                "Virtual qubit declared in physical addressing mode. Mixing modes not currently "
                "supported.",
            )
        self._state = self._VIRTUAL

    def is_physical(self):
        return self._state is self._PHYSICAL

    def __repr__(self):
        return f"AddressingMode({self._state})"


def _check_visible(symbol, context_scope, node):
    if (
        symbol.scope is Scope.GLOBAL
        and context_scope is Scope.GATE
        and not (
            isinstance(symbol.type, types.Gate)
            or (
                isinstance(
                    symbol.type, (types.Int, types.Uint, types.Float, types.Angle, types.Duration)
                )
                and symbol.type.const
            )
        )
    ):
        raise_from_node(node, f"Symbol {symbol.name} is not visible in the scope of a gate")
    return symbol


class SymbolTable:
    __slots__ = ("scope", "symbols")

    def __init__(self, scope, symbols=None):
        self.scope = scope
        self.symbols = symbols if symbols is not None else {}


class SymbolTables:
    __slots__ = ("_stack",)

    def __init__(self):
        self._stack = []
        self.push(SymbolTable(Scope.GLOBAL, _BUILTINS.copy()))

    def __len__(self):
        return len(self._stack)

    def __getitem__(self, n):
        return self._stack[n]

    def __contains__(self, name: str):
        for symbol_table in self._stack:
            if name in symbol_table.symbols:
                return True
        return False

    def get(self, name: str, node=None):
        top_scope = self[len(self) - 1].scope
        if top_scope is Scope.GATE and physical_qubit_index(name) is not None:
            raise_from_node(
                node,
                f"Illegal qubit reference '{name}'. References to hardware "
                "qubits not allowed in gate definitions.",
            )
        for symbol_table in reversed(self._stack):
            if (symbol := symbol_table.symbols.get(name, None)) is not None:
                return _check_visible(symbol, top_scope, node)
        return None

    @property
    def _global_symbol_table(self):
        return self[0]

    @property
    def _top_symbol_table(self):
        return self[len(self) - 1]

    def push(self, symbol_table: SymbolTable):
        if symbol_table.scope is Scope.GLOBAL and len(self) > 1:
            raise RuntimeError("Only one global symbol table may be pushed to the stack.")
        return self._stack.append(symbol_table)

    def pop(self):
        return self._stack.pop()

    def insert(self, symbol):
        target = (
            self._global_symbol_table.symbols
            if symbol.scope is Scope.GLOBAL
            else self._top_symbol_table.symbols
        )
        if (other_symbol := target.get(symbol.name, None)) is not None:
            if other_symbol.name == symbol.name and other_symbol.scope == other_symbol.scope:
                raise_from_node(
                    symbol.definer,
                    f"Symbol '{symbol.name}' already inserted in symbol table in this scope: {symbol.scope}",
                )
        target[symbol.name] = symbol

    def globals(self):
        """Return an iterator over the global symbols."""
        return self._global_symbol_table.symbols.values()

    def _dump(self):
        for n, table in enumerate(self._stack):
            print(f"Table number {n}, {len(table)} syms, scope={table.scope}:")
            print(table.symbols, "\n")


class State:
    """
    Mutable state used during translation of OpenQASM code to a QuantumCircuit.

    Slots:
      scope (Scope): The current lexical scope for the statements being translated.
      _source (str): The entire OpenQASM program being translated. This is not directly
                  translated, but rather an AST derived from the source is the input.
                  Instead, this source is used for diagnostics.
      circuit (qiskit.circuit.QuantumCircuit): The output of the translation.
      symbol_table (SymbolTables): A structure that tracks the symbols (e.g. identifiers) that
                  have been encountered along with some information about them.
      _unique (function) : A function that returns unique symbol names.
      addressing_mode: A structure that tracks the state of the addressing mode; either unknown,
                  virtual, or hardware.
    """

    __slots__ = (
        "scope",
        "_source",
        "circuit",
        "symbol_table",
        "_unique",
        "addressing_mode",
        "all_parameters",
    )

    def __init__(self, source: Optional[str] = None):
        # We use the entire source, because at the moment, that's what all the error messages
        # expect; the nodes have references to the complete source in their spans.
        self._source = source
        self.scope = Scope.GLOBAL
        self.symbol_table = SymbolTables()
        self.addressing_mode = AddressingMode()
        self.all_parameters = set()
        self.circuit = QuantumCircuit()
        self._unique = (f"_{x}" for x in itertools.count())
        self._finish_init()

    def _finish_init(self):
        self.circuit = QuantumCircuit()
        self._unique = (f"_{x}" for x in itertools.count())

    @classmethod
    def _new_scope(cls, scope, context):
        new_context = State.__new__(State)
        new_context.scope = scope
        new_context.symbol_table = context.symbol_table
        new_context.symbol_table.push(SymbolTable(scope))
        new_context.addressing_mode = context.addressing_mode
        new_context.all_parameters = set()
        new_context._source = context._source
        return new_context

    @classmethod
    def new_with_local_scope(cls, context):
        """Return a copy of `context` modified to include a new local scope on the stack."""
        new_context = State._new_scope(Scope.LOCAL, context)
        new_context.circuit = context.circuit
        new_context._unique = context._unique  # pylint: disable=protected-access
        return new_context

    @classmethod
    def new_with_gate_scope(cls, context):
        """Return a copy of `context` modified to include a new gate scope on the stack."""
        new_context = State._new_scope(Scope.GATE, context)
        new_context._finish_init()
        return new_context

    def unique_name(self, prefix=None):
        """Get a name that is not defined in the current scope."""
        while (name := f"{prefix}{next(self._unique)}") in self.symbol_table:
            pass
        return name


class LocalScope:
    def __init__(self, context: State):
        self._local_scope = State.new_with_local_scope(context)

    def __enter__(self) -> State:
        return self._local_scope

    def __exit__(self, _exc_type, _exc_value, _traceback):
        for parameter in self._local_scope.all_parameters - set(
            self._local_scope.circuit.parameters
        ):
            add_dummy_parameter_reference(self._local_scope.circuit, parameter)
        self._local_scope.symbol_table.pop()


class GateScope:
    def __init__(self, context: State):
        self._gate_scope = State.new_with_gate_scope(context)

    def __enter__(self) -> State:
        return self._gate_scope

    def __exit__(self, _exc_type, _exc_value, _traceback):
        for parameter in self._gate_scope.all_parameters - set(self._gate_scope.circuit.parameters):
            add_dummy_parameter_reference(self._gate_scope.circuit, parameter)
        self._gate_scope.symbol_table.pop()
