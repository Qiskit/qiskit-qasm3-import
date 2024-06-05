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

__all__ = ["ConvertVisitor"]

import re
import string
import sys

if sys.version_info < (3, 9):
    from typing import Iterator, Sequence
else:
    from collections.abc import Iterator, Sequence
from typing import Any, Callable, List, NoReturn, Optional, Tuple, Union

from openqasm3 import ast
from openqasm3.visitor import QASMVisitor
from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    Gate,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.transpiler import Layout
from qiskit.transpiler.layout import TranspileLayout
from qiskit.circuit.library import standard_gates as _std

from . import types
from .data import Scope, Symbol
from .exceptions import ConversionError, raise_from_node
from .expression import ValueResolver, resolve_condition
from .state import State, LocalScope, GateScope, add_dummy_parameter_reference

_QASM2_IDENTIFIER = re.compile(r"[a-z]\w*", flags=re.ASCII)

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


def _escape_qasm2(name: str) -> str:
    """Escape a `name` to produce a valid OpenQASM 2 identifier (ignoring things like reserved
    keywords).  This is necessary for registers as of Terra 0.22 beacuse their names have an
    initialisation check that they match this regex.  It should be able to be removed once that
    restriction is lifted."""
    if _QASM2_IDENTIFIER.fullmatch(name):
        return name
    name = re.sub(r"\W", "_", name)
    if not name or name[0] not in string.ascii_lowercase:
        name = "esc_" + name
    return name


class GateBuilder:
    def __init__(
        self, name: str, definition: QuantumCircuit, order: Optional[Sequence[Parameter]] = None
    ):
        self._name = name
        self._definition = definition
        self._order = tuple(self._definition.parameters) if order is None else tuple(order)

    def __call__(self, *parameters):
        if len(parameters) != len(self._order):
            raise ConversionError(
                "incorrect number of parameters in call. Expecting "
                f" {len(self._order)}, got {len(parameters)}."
            )
        out = Gate(self._name, self._definition.num_qubits, parameters)
        if parameters:
            out._definition = self._definition.assign_parameters(dict(zip(self._order, parameters)))
        else:
            out._definition = self._definition.copy()
        return out


class ConvertVisitor(QASMVisitor[State]):
    """Internal visitor of an OpenQASM 3 AST to convert it to a
    :class:`~qiskit.circuit.QuantumCircuit`.  The complete conversion is done by calling
    :meth:`convert` on a :class:`openqasm3.ast.Program` node.

    The other methods on this class are internal only, and generally not part of the public
    interface."""

    # This class assumes that the given AST was a valid OpenQASM 3 program.  It is not within our
    # scope for this simple package to gracefully handle arbitrary semantically invalid programs.
    # In some places, such as symbol definitions, we do some simple checks to help everyone's
    # sanity, as the reference package doesn't yet do this.

    # pylint: disable=missing-function-docstring,no-self-use,unused-argument

    def convert(self, node: ast.Program, *, source: Optional[str] = None) -> QuantumCircuit:
        """Convert a program node into a :class:`~qiskit.circuit.QuantumCircuit`.

        If given, `source` is a string containing the OpenQASM 3 source code that was parsed into
        `node`.  This is used to generated improved error messages. A :class:`.State` containing
        information about the conversion is returned. The :class:`~qiskit.circuit.QuantumCircuit` is
        stored in property thereof named `circuit`.
        """

        state: State = self.visit(node, State(source))
        if state.addressing_mode.is_physical():
            # pylint: disable=protected-access
            state.circuit._layout = TranspileLayout(
                initial_layout=Layout.from_qubit_list(state.circuit.qubits),
                input_qubit_mapping={bit: i for i, bit in enumerate(state.circuit.qubits)},
                final_layout=None,
            )
        for parameter in state.all_parameters - set(state.circuit.parameters):
            add_dummy_parameter_reference(state.circuit, parameter)
        return state

    def _raise_previously_defined(self, new: Symbol, old: Symbol, node: ast.QASMNode) -> NoReturn:
        message = f"'{new.name}' is already defined."
        if old.definer and (span := old.definer.span) is not None:
            message += f" Previous definition on line {span.start_line}."
        raise_from_node(node, message)

    def _define_gate(
        self,
        name: str,
        definition: Callable,
        n_parameters: int,
        n_qubits: int,
        definer: ast.QASMNode,
        context: State,
    ) -> State:
        if context.scope is not Scope.GLOBAL:
            raise_from_node(definer, "gates can only be declared globally")
        type = types.Gate(n_parameters, n_qubits)
        symbol = Symbol(name, definition, type, Scope.GLOBAL, definer)
        if (previous := context.symbol_table.get(name, definer)) is not None:
            self._raise_previously_defined(symbol, previous, definer)
        context.symbol_table.insert(symbol)
        return context

    def _apply_gate_modifier(
        self, modifier: ast.QuantumGateModifier, gate: Gate, context: State
    ) -> Gate:
        if modifier.modifier is ast.GateModifierName.inv:
            return gate.inverse()
        if modifier.modifier is ast.GateModifierName.pow:
            if modifier.argument is None:
                # Should be handled by AST creation.
                raise_from_node(modifier, "'pow' requires exactly one argument")
            return gate.power(self._resolve_constant_float(modifier.argument, context))
        # ctrl / negctrl
        num_controls = (
            1
            if modifier.argument is None
            else self._resolve_constant_int(modifier.argument, context)
        )
        ctrl_state = (
            (0b1 << num_controls) - 1 if modifier.modifier is ast.GateModifierName.ctrl else 0
        )
        return gate.control(num_controls, ctrl_state=ctrl_state)

    def _broadcast_gate(
        self,
        arguments: Sequence[Union[Qubit, Sequence[Qubit]]],
        node: ast.QASMNode,
    ) -> Iterator[Tuple[Qubit, ...]]:
        max_length = max(1 if isinstance(x, Qubit) else len(x) for x in arguments)

        def args():
            for argument in arguments:
                if isinstance(argument, Qubit):
                    yield (argument,) * max_length
                elif len(argument) != max_length:
                    raise_from_node(node, "mismatched lengths in gate broadcast")
                else:
                    yield tuple(argument)

        return zip(*args())

    def _resolve_generic(
        self, node: ast.Expression, context: State, strict: bool
    ) -> Tuple[Any, types.Type]:
        return ValueResolver(context, strict).resolve(node)

    def _resolve_constant_int(self, node: ast.Expression, context: State) -> int:
        value, type = self._resolve_generic(node, context, strict=True)
        if not isinstance(type, (types.Int, types.Uint)) or not type.const:
            raise_from_node(node, "required a constant integer")
        return value

    def _resolve_constant_float(self, node: ast.Expression, context: State) -> float:
        value, type = self._resolve_generic(node, context, strict=True)
        if not isinstance(type, (types.Int, types.Uint, types.Float)) or not type.const:
            raise_from_node(node, "required a constant floating-point number")
        return value

    def _resolve_constant_duration(self, node: ast.Expression, context: State) -> Tuple[float, str]:
        value, type = self._resolve_generic(node, context, strict=True)
        if not isinstance(type, types.Duration) or not type.const:
            raise_from_node(node, "required a constant duration")
        return value

    def _resolve_angle(
        self, node: ast.Expression, context: State
    ) -> Union[float, ParameterExpression]:
        value, type = self._resolve_generic(node, context, strict=False)
        if not isinstance(type, (types.Int, types.Uint, types.Angle, types.Float)):
            raise_from_node(node, "required an angle-like value")
        return value

    def _resolve_carg(
        self, node: ast.Expression, context: State
    ) -> Union[Clbit, ClassicalRegister, List[Clbit]]:
        value, type = self._resolve_generic(node, context, strict=True)
        if not isinstance(type, (types.Bit, types.BitArray)):
            raise_from_node(node, "required a bit or bit register")
        return value

    def _resolve_qarg(
        self, node: ast.Expression, context: State
    ) -> Union[Qubit, QuantumRegister, List[Qubit]]:
        value, type = self._resolve_generic(node, context, strict=True)
        if not isinstance(type, (types.Qubit, types.HardwareQubit, types.QubitArray)):
            raise_from_node(node, "required a qubit or qubit register")
        return value

    def _resolve_condition(
        self, node: ast.Expression, context: State
    ) -> Union[Tuple[ClassicalRegister, int], Tuple[Clbit, bool]]:
        lhs, rhs = resolve_condition(node, context)
        if not isinstance(lhs, (Clbit, ClassicalRegister)):
            name = context.unique_name()
            lhs = ClassicalRegister(name=_escape_qasm2(name), bits=lhs)
            context.circuit.add_register(lhs)
            context.symbol_table.insert(Symbol(name, lhs, types.BitArray, Scope.NONE))
        return (lhs, rhs)

    # Everything below is the implementation of the visitor itself.  The general `visit` method is
    # derived from the base class.

    def generic_visit(self, node, context=None):
        raise_from_node(node, f"node of type {node.__class__.__name__} is not supported")

    def visit_Program(self, node: ast.Program, context: State) -> State:
        for statement in node.statements:
            context = self.visit(statement, context)
        return context

    def visit_Include(self, node: ast.Include, context: State) -> State:
        if node.filename != "stdgates.inc":
            raise_from_node(node, "non-stdgates imports not currently supported")
        for name, (builder, n_arguments, n_qubits) in _STDGATES.items():
            context = self._define_gate(name, builder, n_arguments, n_qubits, node, context)
        return context

    def visit_QubitDeclaration(self, node: ast.QubitDeclaration, context: State) -> State:
        context.addressing_mode.set_virtual_mode(node)
        name = node.qubit.name
        if node.size is None:
            bit = Qubit()
            context.circuit.add_bits([bit])
            symbol = Symbol(name, bit, types.Qubit(), Scope.GLOBAL, node)
        else:
            size = self._resolve_constant_int(node.size, context)
            register = QuantumRegister(size, name=_escape_qasm2(name))
            context.circuit.add_register(register)
            symbol = Symbol(name, register, types.QubitArray(size), Scope.GLOBAL, node)
        context.symbol_table.insert(symbol)
        return context

    def visit_QuantumGateDefinition(self, node: ast.QuantumGateDefinition, context: State) -> State:
        with GateScope(context) as inner:
            parameters = [Parameter(name.name) for name in node.arguments]
            for parameter in parameters:
                inner.symbol_table.insert(
                    Symbol(parameter.name, parameter, types.Angle(), Scope.GATE, node)
                )
                inner.all_parameters.add(parameter)
            bits = [Qubit() for _ in node.qubits]
            inner.circuit.add_bits(bits)
            for name, bit in zip(node.qubits, bits):
                inner.symbol_table.insert(Symbol(name.name, bit, types.Qubit(), Scope.GATE, node))
            for statement in node.body:
                self.visit(statement, inner)
        return self._define_gate(
            node.name.name,
            GateBuilder(node.name.name, inner.circuit),
            len(parameters),
            len(bits),
            node,
            context,
        )

    def visit_QuantumGate(self, node: ast.QuantumGate, context: State) -> State:
        if node.duration is not None:
            raise_from_node(node, "gates with durations are not supported.")
        if (gate_symbol := context.symbol_table.get(node.name.name, node)) is None:
            raise_from_node(node, f"gate '{node.name.name}' is not defined.")
        if not isinstance(gate_symbol.type, types.Gate):
            message = f"'{node.name.name}' is a '{gate_symbol.type.pretty()}', not a gate."
            if (span := gate_symbol.definer.span) is not None:
                message += f" Definition on line {span.start_line}"
            raise_from_node(node, message)
        gate_builder = gate_symbol.data
        arguments = [self._resolve_angle(argument, context) for argument in node.arguments]
        gate = gate_builder(*arguments)
        for modifier in reversed(node.modifiers):
            gate = self._apply_gate_modifier(modifier, gate, context)
        for i, qubits in enumerate(
            self._broadcast_gate([self._resolve_qarg(qarg, context) for qarg in node.qubits], node)
        ):
            if i > 0:
                gate = gate.copy()
            context.circuit.append(gate, qubits, [])
        return context

    def visit_QuantumPhase(self, node: ast.QuantumPhase, context: State) -> State:
        gate = QuantumCircuit(global_phase=self._resolve_angle(node.argument, context)).to_gate()
        for modifier in reversed(node.modifiers):
            gate = self._apply_gate_modifier(modifier, gate, context)
        if not node.qubits:
            context.circuit.global_phase += gate.definition.global_phase
            return context
        for i, qubits in enumerate(
            self._broadcast_gate([self._resolve_qarg(qarg, context) for qarg in node.qubits], node)
        ):
            if i > 0:
                gate = gate.copy()
            context.circuit.append(gate, qubits, [])
        return context

    def visit_QuantumMeasurementStatement(
        self, node: ast.QuantumMeasurementStatement, context: State
    ) -> State:
        if node.target is None:
            raise_from_node(node, "measurements must save their result in Qiskit")
        measured = self._resolve_qarg(node.measure.qubit, context)
        target = self._resolve_carg(node.target, context)
        context.circuit.measure(measured, target)
        return context

    def visit_QuantumBarrier(self, node: ast.QuantumBarrier, context: State) -> State:
        context.circuit.barrier(*[self._resolve_qarg(qarg, context) for qarg in node.qubits])
        return context

    def visit_QuantumReset(self, node: ast.QuantumReset, context: State) -> State:
        context.circuit.reset(self._resolve_qarg(node.qubits, context))
        return context

    def visit_ClassicalDeclaration(self, node: ast.ClassicalDeclaration, context: State) -> State:
        if context.scope is not Scope.GLOBAL:
            raise_from_node(node, "only global declarations are supported")
        if not isinstance(node.type, ast.BitType):
            type_name = node.type.__class__.__name__[:-4].lower()  # Cheeky quick hack.
            raise_from_node(node, f"declarations of type '{type_name}' are not supported")
        name = node.identifier.name
        if node.type.size is None:
            bit = Clbit()
            context.circuit.add_bits([bit])
            symbol = Symbol(name, bit, types.Bit(), context.scope, node)
        else:
            size = self._resolve_constant_int(node.type.size, context)
            register = ClassicalRegister(size, name=_escape_qasm2(name))
            context.circuit.add_register(register)
            symbol = Symbol(name, register, types.BitArray(size), context.scope, node)
        context.symbol_table.insert(symbol)
        if node.init_expression is not None:
            if not isinstance(node.init_expression, ast.QuantumMeasurement):
                raise_from_node(
                    node.init_expression, "initialisation of classical bits is not supported"
                )
            measured = self._resolve_qarg(node.init_expression.qubit, context)
            target = symbol.data
            context.circuit.measure(measured, target)
        return context

    def visit_IODeclaration(self, node: ast.IODeclaration, context: State) -> State:
        if node.io_identifier is ast.IOKeyword.output:
            raise_from_node(node, "the 'output' keyword is not supported")
        if isinstance(node.type, ast.FloatType):
            size = (
                None
                if node.type.size is None
                else self._resolve_constant_int(node.type.size, context)
            )
            type = types.Float(size=size)
        elif isinstance(node.type, ast.AngleType):
            size = (
                None
                if node.type.size is None
                else self._resolve_constant_int(node.type.size, context)
            )
            type = types.Angle(size=size)
        else:
            raise_from_node(node, "only 'float' and 'angle' inputs are supported")
        name = node.identifier.name
        parameter = Parameter(name)
        symbol = Symbol(name, parameter, type, Scope.GLOBAL, node)
        context.symbol_table.insert(symbol)
        context.all_parameters.add(parameter)
        return context

    def visit_BreakStatement(self, node: ast.BreakStatement, context: State) -> State:
        context.circuit.break_loop()
        return context

    def visit_ContinueStatement(self, node: ast.ContinueStatement, context: State) -> State:
        context.circuit.continue_loop()
        return context

    def visit_BranchingStatement(self, node: ast.BranchingStatement, context: State) -> State:
        condition = self._resolve_condition(node.condition, context)
        with context.circuit.if_test(condition) as else_:
            with LocalScope(context) as inner:
                for statement in node.if_block:
                    self.visit(statement, inner)
        if not node.else_block:
            return context
        with else_:
            with LocalScope(context) as inner:
                for statement in node.else_block:
                    self.visit(statement, inner)
        return context

    def visit_WhileLoop(self, node: ast.WhileLoop, context: State) -> State:
        condition = self._resolve_condition(node.while_condition, context)
        with context.circuit.while_loop(condition):
            with LocalScope(context) as inner:
                for statement in node.block:
                    self.visit(statement, inner)
        return context

    def visit_ForInLoop(self, node: ast.ForInLoop, context: State) -> State:
        if not isinstance(node.type, (ast.IntType, ast.UintType)):
            raise_from_node(node, "only integer loop variables are supported")
        indexset, indextype = self._resolve_generic(node.set_declaration, context, strict=True)
        if not isinstance(indextype, (types.Range, types.Sequence)):
            raise_from_node(
                node.set_declaration, "only ranges and discrete integer sets are supported"
            )
        if isinstance(indextype, types.Range):
            # indexset is a slice.  Convert to range.
            if indexset.start is None or indexset.stop is None:
                raise_from_node(node.set_declaration, "for-loop ranges must have a start and end")
            indexset = (
                range(indexset.start, indexset.stop)
                if indexset.step is None
                else range(indexset.start, indexset.stop, indexset.step)
            )
        var_type = types.Int() if isinstance(node.type, ast.IntType) else types.Uint()
        with context.circuit.for_loop(indexset) as parameter:
            with LocalScope(context) as inner:
                name = node.identifier.name
                symbol = Symbol(name, parameter, var_type, Scope.LOCAL, node)
                inner.symbol_table.insert(symbol)
                inner.all_parameters.add(parameter)
                for statement in node.block:
                    self.visit(statement, inner)
        return context

    def visit_DelayInstruction(self, node: ast.DelayInstruction, context: State) -> State:
        duration, unit = self._resolve_constant_duration(node.duration, context)
        if not node.qubits:
            context.circuit.delay(duration, unit=unit)
            return context
        for qarg in node.qubits:
            context.circuit.delay(duration, self._resolve_qarg(qarg, context), unit=unit)
        return context

    def visit_AliasStatement(self, node: ast.AliasStatement, context: State) -> State:
        bits, type = self._resolve_generic(node.value, context, strict=True)
        name = node.target.name
        inner_name = _escape_qasm2(name)
        if context.scope is not Scope.GLOBAL:
            inner_name = context.unique_name(inner_name)
        if isinstance(type, types.BitArray):
            register = ClassicalRegister(name=inner_name, bits=bits)
        elif isinstance(type, types.QubitArray):
            register = QuantumRegister(name=inner_name, bits=bits)
        else:
            raise_from_node(
                node.value,
                f"aliases must be of registers of either clbits or qubits, not '{type.pretty()}'",
            )
        context.circuit.add_register(register)
        context.symbol_table.insert(Symbol(name, register, type, context.scope, node))
        return context
