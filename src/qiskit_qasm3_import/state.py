import enum
from typing import Optional
import itertools
import math

from qiskit.circuit import QuantumCircuit

from qiskit.circuit.library import standard_gates as _std

from . import types
from .data import Scope, Symbol
from .exceptions import raise_from_node, PhysicalQubitInGateError


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
        if self._state == self._VIRTUAL:
            raise_from_node(
                node,
                "Physical qubit referenced in virtual addressing mode. Mixing "
                "modes not currently supported.",
            )
        self._state = self._PHYSICAL

    def set_virtual_mode(self, node):
        """Set the addressing mode to virtual.
        On success return `True`, otherwise raise an exception."""
        if self._state == self._PHYSICAL:
            raise_from_node(
                node,
                "Virtual qubit declared in physical addressing mode. Mixing modes not currently "
                "supported.",
            )
        self._state = self._VIRTUAL


class SymbolTable:
    """
    Slots:
       _symbols (dict): A dict of non-global symbols
       _scope (Scope): The scope associated with this symbol table
       _base (SymbolTable): The parent to this symbol table
       _global_symbols (dict): A dict of global symbols
    """

    __slots__ = ("_symbols", "_scope", "_base", "_global_symbols")

    def __init__(self, scope: Scope, base: Optional["SymbolTable"] = None):
        self._symbols = {}
        self._scope = scope
        self._base = base
        self._global_symbols = base._global_symbols if base is not None else _BUILTINS.copy()

    def __contains__(self, name: str):
        return name in (self._symbols, self._global_symbols) or (
            self._base is not None and name in self._base._symbols
        )

    def get(self, name: str, node=None):
        """Lookup symbol `name`

        Return `None` if no symbol is found. Raise an exception if the symbol is
        inaccessible in the scope of this table.
        """
        if (symbol := self._symbols.get(name, None)) is not None:
            return symbol
        if self._base is not None:
            if (symbol := self._base.get(name, None)) is None:
                return None
            return self._check_visible(symbol, node)
        return self._global_symbols.get(name, None)

    def _check_visible(self, symbol, node):
        if (
            symbol.scope is Scope.BUILTIN
            or self._scope is not Scope.GATE
            or symbol.scope is not Scope.GLOBAL
        ):
            return symbol
        if self._scope is Scope.GATE:
            if isinstance(symbol.type, types.Gate) or (
                isinstance(
                    symbol.type, (types.Int, types.Uint, types.Float, types.Angle, types.Duration)
                )
                and symbol.type.const
            ):
                return symbol
        if isinstance(symbol.type, types.HardwareQubit):
            raise PhysicalQubitInGateError(node.name, node)
        raise_from_node(node, f"Symbol {symbol.name} is not visible in the scope of a gate")

    def insert(self, symbol):
        target = self._global_symbols if symbol.type == types.HardwareQubit() else self._symbols
        if (other_symbol := target.get(symbol.name, None)) is not None:
            if other_symbol.name == symbol.name and other_symbol.scope == other_symbol.scope:
                raise_from_node(
                    symbol.definer,
                    f"Symbol '{symbol.name}' already inserted in symbol table in this scope: {symbol.scope}",
                )
        target[symbol.name] = symbol

    def globals(self):
        """Return an iterator over the global symbols."""
        return self._global_symbols.values()


class State:
    """
    Mutable state used during translation of OpenQASM code to a QuantumCircuit.

    Slots:
      scope (Scope): The current lexical scope for the statements being translated.
      _source (str): The entire OpenQASM program being translated. This is not directly
                  translated, but rather an AST derived from the source is the input.
                  Instead, this source is used for diagnostics.
      circuit (qiskit.circuit.QuantumCircuit): The output of the translation.
      symbol_table (SymbolTable): A structure that tracks the symbols (e.g. identifiers) that
                  have been encountered along with some information about them.
      _unique (function) : A function that returns unique symbol names.
      addressing_mode: A structure that tracks the state of the addressing mode; either unknown,
                  virtual, or hardware.
    """

    __slots__ = ("scope", "_source", "circuit", "symbol_table", "_unique", "addressing_mode")

    def __init__(self, source: Optional[str] = None):
        # We use the entire source, because at the moment, that's what all the error messages
        # expect; the nodes have references to the complete source in their spans.
        self._source = source
        self._init_inner()

    def _init_inner(
        self,
        scope: Scope = None,
        state_in: "State" = None,
        circuit=None,
        unique=None,
    ):
        self.scope = scope if scope is not None else Scope.GLOBAL
        if state_in is None:
            self.symbol_table = SymbolTable(Scope.GLOBAL)
            self.addressing_mode = AddressingMode()
        else:
            self.symbol_table = SymbolTable(scope, state_in.symbol_table)
            self.addressing_mode = state_in.addressing_mode
            self._source = state_in._source
        self.circuit = circuit if circuit is not None else QuantumCircuit()
        self._unique = unique if unique is not None else (f"_{x}" for x in itertools.count())

        return self

    def gate_scope(self):
        """Get a new state for entry to a "gate" scope."""
        # pylint: disable=protected-access
        return State()._init_inner(Scope.GATE, self, None, None)

    def local_scope(self):
        """Get a new state on entry to a local block scope."""
        # pylint: disable=protected-access
        return State()._init_inner(Scope.LOCAL, self, self.circuit, self._unique)

    def unique_name(self, prefix=None):
        """Get a name that is not defined in the current scope."""
        while (name := f"{prefix}{next(self._unique)}") in self.symbol_table:
            pass
        return name
