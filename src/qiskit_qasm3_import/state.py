import enum
from typing import Optional
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
        """Set the addressing mode to physical. On success return `True`, otherwise `False`."""
        if self._state == self.VIRTUAL:
            raise_from_node(
                node,
                "Physical qubit referenced in virtual addressing mode. Mixing modes not currently supported.",
            )
        self._state = self.PHYSICAL

    def set_virtual_mode(self, node):
        """Set the addressing mode to virtual. On success return `True`, otherwise `False`."""
        if self._state == self.PHYSICAL:
            raise_from_node(
                node,
                "Virtual qubit declared in physical addressing mode. Mixing modes not currently supported.",
            )
        self._state = self.VIRTUAL


class SymbolTable:
    __slots__ = ("local_table", "global_table", "builtin_table")

    def __init__(self):
        self.global_table = {}
        self.builtin_table = _BUILTINS.copy()
        self.local_table = {} # For everything else

    def gate_scope_copy(self):
        out = SymbolTable.__new__(SymbolTable)
        out.local_table = {}
        out.builtin_table = self.builtin_table
        out.global_table = {}
        for name, item in self.global_table.items():
            if (isinstance(item.type, types.Gate) or
                isinstance(
                    item.type, (types.Int, types.Uint, types.Float, types.Angle, types.Duration)
                )
                and item.type.const):
                out.global_table[name] = item  # TODO: use insert
        return out

    def local_scope_copy(self):
        out = SymbolTable.__new__(SymbolTable)
        out.local_table = self.local_table.copy()
        out.global_table = self.global_table
        out.builtin_table = self.builtin_table

        return out

    def insert(self, symbol: Symbol): # This does not catch shadowing builtins
        if symbol.scope == Scope.GLOBAL:
            self.global_table[symbol.name] = symbol
        else:
            self.local_table[symbol.name] = symbol

    def exists(self, name: str):
        return name in self.builtin_table or name in self.global_table or name in self.local_table

    def get(self, name: str):
        # Search order determines which symbols can shadow others.
        for table in (self.local_table, self.global_table, self.builtin_table):
            if (symbol := table.get(name)):
                return symbol
        return None
#        raise KeyError(f"Symbol {name} not found.")  # TODO: remove this if we decide against raising

    def global_symbols(self):
        return self.global_table.values()


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
        out.symbol_table = self.symbol_table.gate_scope_copy() # A bit inefficient

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
        while (name := f"{prefix}{next(self._unique)}") in self.symbol_table.local_table: # Not using an API
            pass
        return name
