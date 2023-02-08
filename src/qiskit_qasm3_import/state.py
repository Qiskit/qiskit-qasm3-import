import enum
from typing import Optional, Union, List
import itertools
import math

from qiskit.circuit import QuantumCircuit

from qiskit.circuit.library import standard_gates as _std

from . import types
from .data import Scope, Symbol
from .exceptions import raise_from_node

_STDGATES = {
    "p": (_std.PhaseGate, 1, 1),
    "x": (_std.XGate, 0, 1),
    "y": (_std.YGate, 0, 1),
    "z": (_std.ZGate, 0, 1),
    "h": (_std.HGate, 0, 1),
    "s": (_std.SGate, 0, 1),
    "sdg": (_std.SdgGate, 0, 1),
    "t": (_std.TGate, 0, 1),
    "tdg": (_std.TdgGate, 0, 1),
    "sx": (_std.SXGate, 0, 1),
    "rx": (_std.RXGate, 1, 1),
    "ry": (_std.RYGate, 1, 1),
    "rz": (_std.RZGate, 1, 1),
    "cx": (_std.CXGate, 0, 2),
    "cy": (_std.CYGate, 0, 2),
    "cz": (_std.CZGate, 0, 2),
    "cp": (_std.CPhaseGate, 1, 2),
    "crx": (_std.CRXGate, 1, 2),
    "cry": (_std.CRYGate, 1, 2),
    "crz": (_std.CRZGate, 1, 2),
    "ch": (_std.CHGate, 0, 2),
    "swap": (_std.SwapGate, 0, 2),
    "ccx": (_std.CCXGate, 0, 3),
    "cswap": (_std.CSwapGate, 0, 3),
    "cu": (_std.CUGate, 4, 2),
    "CX": (_std.CXGate, 0, 2),
    "phase": (_std.PhaseGate, 1, 1),
    "cphase": (_std.CPhaseGate, 1, 2),
    "id": (
        lambda: _std.UGate(0, 0, 0),  # Qiskit's "id" gate does not strictly have equal semantics.
        0,
        1,
    ),
    "u1": (_std.U1Gate, 1, 1),
    "u2": (_std.U2Gate, 2, 1),
    "u3": (_std.U3Gate, 3, 1),
}

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

    This class is useful as long as we allow only physical or virtual addressing modes, but
    not mixed. If the latter is supported in the future, this class will be modified or removed.
    """

    _Mode = enum.Enum("_Mode", ["UNKNOWN", "PHYSICAL", "VIRTUAL"])
    (UNKNOWN, PHYSICAL, VIRTUAL) = (_Mode.UNKNOWN, _Mode.PHYSICAL, _Mode.VIRTUAL)

    def __init__(self):
        self._state = self.UNKNOWN

    def set_physical_mode(self, node):
        """Set the addressing mode to physical. On success return `True`, otherwise raise an exception."""
        if self._state == self.VIRTUAL:
            raise_from_node(
                node,
                "Physical qubit referenced in virtual addressing mode. Mixing modes not currently supported.",
            )
        self._state = self.PHYSICAL

    def set_virtual_mode(self, node):
        """Set the addressing mode to virtual. On success return `True`, otherwise raise an exception."""
        if self._state == self.PHYSICAL:
            raise_from_node(
                node,
                "Virtual qubit declared in physical addressing mode. Mixing modes not currently supported.",
            )
        self._state = self.VIRTUAL


class SymbolTable:
    __slots__ = ("_local_table", "_global_table", "_builtin_table")

    def __init__(self):
        self._global_table = {}
        self._builtin_table = _BUILTINS.copy()
        self._local_table = {}  # For everything else

    def gate_scope_copy(self):
        """Return a copy of the symbol table for use in the lexical scope of a gate definition.

        The target (returned) symbol table contains: all builtin symbols, no symbols from a surrounding local scope,
        all symbols from the global scope referring to gates gates, all symbols from the global scope
        that are marked constant and refer to numeric data.
        """
        # pylint: disable=protected-access
        out = SymbolTable.__new__(SymbolTable)
        out._local_table = {}
        out._builtin_table = self._builtin_table
        out._global_table = {}
        for name, item in self._global_table.items():
            if (
                isinstance(item.type, types.Gate)
                or isinstance(
                    item.type, (types.Int, types.Uint, types.Float, types.Angle, types.Duration)
                )
                and item.type.const
            ):
                out._global_table[name] = item  # TODO: use insert
        return out

    def local_scope_copy(self):
        """Return a copy of the symbol table for use with a new local scope.


        Local variables created in the target (returned) symbol table will not
        appear in the source symbol table. Thus, these local variables are
        discarded and not visible upon returning to the surrounding scope. In
        contrast, changes to data referred to by global symbols will be visible
        in the surrounding scope.
        """
        # pylint: disable=protected-access
        out = SymbolTable.__new__(SymbolTable)
        out._local_table = self._local_table.copy()
        out._global_table = self._global_table
        out._builtin_table = self._builtin_table

        return out

    def insert(self, symbol: Union[List, Symbol]):
        """Insert a `Symbol` (or each of a list thereof) into the symbol table."""
        if isinstance(symbol, list):
            for sym in symbol:
                self._insert(sym)
        else:
            self._insert(symbol)

    def _insert(self, symbol: Symbol):  # This does not catch shadowing builtins
        if symbol.scope == Scope.GLOBAL:
            self._global_table[symbol.name] = symbol
        else:
            self._local_table[symbol.name] = symbol

    def __contains__(self, name: str):
        # Return `True` if `name` is in symbol table.
        return (
            name in self._builtin_table or name in self._global_table or name in self._local_table
        )

    def in_local(self, name: str):
        """Return true if ``name`` is in the symbol table and has local scope."""
        return name in self._local_table

    def __getitem__(self, name: str):
        # An opaque interface for looking up `name` in the symbol table.
        for table in (self._local_table, self._global_table, self._builtin_table):
            if symbol := table.get(name):
                return symbol
        raise KeyError(f"Symbol {name} not found.")

    def get(self, name: str):
        """Return `Symbol` corresponding to `name`, or `None` if none exists."""
        # Search order determines which symbols can shadow others.
        for table in (self._local_table, self._global_table, self._builtin_table):
            if symbol := table.get(name):
                return symbol
        return None

    def hardware_qubits(self):
        """Return an iterator over the `Symbol`s referring to hardware qubits."""
        return (
            sym for sym in self._global_table.values() if isinstance(sym.type, types.HardwareQubit)
        )


class State:
    __slots__ = ("scope", "source", "circuit", "symbol_table", "_unique", "addressing_mode")

    def __init__(self, scope: Scope, source: Optional[str] = None):
        self.scope = scope
        self.source = source
        self.circuit = QuantumCircuit()
        self.symbol_table = SymbolTable()
        self._unique = (f"_{x}" for x in itertools.count())
        self.addressing_mode = AddressingMode()

    def gate_scope(self):
        """Get a new state for entry to a "gate" scope."""
        # We use the entire source, because at the moment, that's what all the error messages
        # expect; the nodes have references to the complete source in their spans.
        out = State(Scope.GATE, self.source)
        out.symbol_table = self.symbol_table.gate_scope_copy()  # A bit inefficient

        return out

    def local_scope(self):
        """Get a new state on entry to a local block scope."""
        # pylint: disable=protected-access
        out = State.__new__(State)
        out.scope = Scope.LOCAL
        out.source = self.source
        out.circuit = self.circuit  # No copy; we want to keep modifying this one.
        out.symbol_table = self.symbol_table.local_scope_copy()
        out.addressing_mode = self.addressing_mode
        out._unique = self._unique

        return out

    def unique_name(self, prefix=""):
        """Get a name that is not defined in the current scope."""
        while self.symbol_table.in_local(name := f"{prefix}{next(self._unique)}"):
            pass
        return name
