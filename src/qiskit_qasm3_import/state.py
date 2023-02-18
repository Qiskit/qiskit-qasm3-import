import enum
from typing import Optional
import itertools
import math

from qiskit.circuit import QuantumCircuit

from qiskit.circuit.library import standard_gates as _std

from . import types
from .data import Scope, Symbol
from .exceptions import raise_from_node, PhysicalQubitInGateError

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

    This class is useful as long as we allow only physical or virtual addressing modes, but not
    mixed. If the latter is supported in the future, this class will be modified or removed.

    """

    _Mode = enum.Enum("_Mode", ["UNKNOWN", "PHYSICAL", "VIRTUAL"])
    (UNKNOWN, PHYSICAL, VIRTUAL) = (_Mode.UNKNOWN, _Mode.PHYSICAL, _Mode.VIRTUAL)

    def __init__(self):
        self._state = self.UNKNOWN

    def set_physical_mode(self, node):
        """Set the addressing mode to physical. On success return `True`, otherwise raise an
        exception."""
        if self._state == self.VIRTUAL:
            raise_from_node(
                node,
                "Physical qubit referenced in virtual addressing mode. Mixing "
                "modes not currently supported.",
            )
        self._state = self.PHYSICAL

    def set_virtual_mode(self, node):
        """Set the addressing mode to virtual.
        On success return `True`, otherwise raise an exception."""
        if self._state == self.PHYSICAL:
            raise_from_node(
                node,
                "Virtual qubit declared in physical addressing mode. Mixing modes not currently "
                "supported.",
            )
        self._state = self.VIRTUAL


class SymbolTable:
    __slots__ = ("symbols", "scope", "base", "global_symbols")

    def __init__(self, scope: Scope, base: Optional["SymbolTable"] = None):
        self.symbols = {}
        self.scope = scope
        self.base = base
        self.global_symbols = base.global_symbols if base is not None else _BUILTINS.copy()

    def __contains__(self, name: str):
        return name in (self.symbols, self.global_symbols) or (
            self.base is not None and name in self.base.symbols
        )

    def get(self, name: str, node=None):
        """Lookup symbol `name`

        Return `None` if no symbol is found. Raise an exception if the symbol is
        inaccessible in the scope of this table.
        """
        if (symbol := self.symbols.get(name, None)) is not None:
            return symbol
        if self.base is not None:
            if (symbol := self.base.get(name, None)) is None:
                return None
            return self._check_visible(symbol, node)
        return self.global_symbols.get(name, None)

    def _check_visible(self, symbol, node):
        if (
            symbol.scope is Scope.BUILTIN
            or self.scope is not Scope.GATE
            or symbol.scope is not Scope.GLOBAL
        ):
            return symbol
        if self.scope is Scope.GATE:
            if (
                isinstance(symbol.type, types.Gate)
                or isinstance(
                    symbol.type, (types.Int, types.Uint, types.Float, types.Angle, types.Duration)
                )
                and symbol.type.const
            ):
                return symbol
        if isinstance(symbol.type, types.HardwareQubit):
            raise PhysicalQubitInGateError(node.name, node)
        raise_from_node(node, f"Symbol {symbol.name} is not visible in the scope of a gate")

    def insert(self, symbol):
        target = self.global_symbols if symbol.type == types.HardwareQubit() else self.symbols
        target[symbol.name] = symbol

    def globals(self):
        if self.global_symbols is None:
            return []
        return self.global_symbols.values()


class State:
    __slots__ = ("scope", "source", "circuit", "symbol_table", "_unique", "addressing_mode")

    def __init__(self, source: Optional[str] = None):
        # We use the entire source, because at the moment, that's what all the error messages
        # expect; the nodes have references to the complete source in their spans.
        self.source = source
        self.init_inner()

    def init_inner(
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
            self.source = state_in.source
        self.circuit = circuit if circuit is not None else QuantumCircuit()
        self._unique = unique if unique is not None else (f"_{x}" for x in itertools.count())

        return self

    def gate_scope(self):
        """Get a new state for entry to a "gate" scope."""
        return State().init_inner(Scope.GATE, self, None, None)

    def local_scope(self):
        """Get a new state on entry to a local block scope."""
        return State().init_inner(Scope.LOCAL, self, self.circuit, self._unique)

    def unique_name(self, prefix=None):
        """Get a name that is not defined in the current scope."""
        while (name := f"{prefix}{next(self._unique)}") in self.symbol_table:
            pass
        return name
