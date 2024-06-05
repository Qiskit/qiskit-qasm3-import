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

"""QASM visitors for the expression tree, resolving it into objects that are useful for interaction
with Qiskit.

Ideally, this would just have a single large ``ExpressionResolver`` or something like that, but
since Qiskit's type system can't represent anywhere close to what OpenQASM 3 can represent, the
output types would be rather confusing.  For now, it's much easier to split into a handful of more
specialised resolvers that can bail out early when they encounter things that are outside their
scope, rather than trying to build a whole new internal IR to handle everything at once."""

# In a few places, we need to check that the types of two `types.Type` instances are
# specifically the same.
# pylint: disable=unidiomatic-typecheck

# The binary operator rules are just complicated.  Early return is probably the easiest.
# pylint: disable=too-many-return-statements,too-many-branches


__all__ = ["ValueResolver", "resolve_condition"]

from typing import Any, Iterable, Tuple, Union, cast

from openqasm3 import ast
from openqasm3.visitor import QASMVisitor
from qiskit.circuit import Clbit, Qubit

from . import types
from .exceptions import raise_from_node
from .data import Symbol, Scope
from . import state


_IntegerT = Union[types.Never, types.Int, types.Uint]
_NumericT = Union[types.Uint, types.Int, types.Float]


def join_integer_types(left: _IntegerT, right: _IntegerT) -> _IntegerT:
    if isinstance(left, types.Never):
        return right
    if isinstance(right, types.Never):
        return left
    if not (
        isinstance(left, (types.Int, types.Uint)) and isinstance(right, (types.Int, types.Uint))
    ):
        return types.Error()
    const = left.const and right.const
    size = None if left.size is None or right.size is None else max((left.size, right.size))
    if isinstance(left, types.Uint) and isinstance(right, types.Uint):
        return types.Uint(const, size)
    return types.Int(const, size)


def join_numeric_types(left: _NumericT, right: _NumericT) -> _NumericT:
    const = left.const and right.const
    if isinstance(left, types.Float) and isinstance(right, types.Float):
        size = None if left.size is None or right.size is None else max((left.size, right.size))
        return types.Float(const, size)
    if isinstance(left, types.Float):
        return types.Float(const, left.size)
    if isinstance(right, types.Float):
        return types.Float(const, right.size)
    # Cast because this can't return the never type since neither of the inputs are never.
    return cast(_NumericT, join_integer_types(left, right))


class ValueResolver(QASMVisitor):
    """A resolver for value-like objects that have exact Qiskit representations.  Everything handled
    here should resolve into a single Qiskit object (or occasionally a compound of such objects,
    like lists of bits).

    These resolvers are intended to be short-lived. Their instance state is the instance of
    :class:`.State` created in :meth:`.ConvertVisitor.convert()`.
    """

    __slots__ = ("_context", "_strict")

    # pylint: disable=no-self-use

    def __init__(self, context: state.State, strict: bool = True):
        self._context = context
        self._strict = strict

    def resolve(self, node: ast.Expression) -> Tuple[Any, types.Type]:
        """The entry point to the resolver, resolving the AST node into a 2-tuple of a relevant
        Qiskit type, and the :class:`.Type` that it is an instance of."""

        return self.visit(node)

    def visit(self, node: ast.QASMNode, context: None = None) -> Tuple[Any, types.Type]:
        value, type = super().visit(node)
        if isinstance(type, types.Error):
            raise_from_node(node, "type error")
        return value, type

    def generic_visit(self, node: ast.QASMNode, context: None = None):
        raise_from_node(node, f"'{node.__class__.__name__}' cannot be resolved into a Qiskit value")

    def visit_Identifier(self, node: ast.Identifier):
        cxt = self._context
        if (symbol := cxt.symbol_table.get(node.name, node)) is not None:
            return symbol.data, symbol.type
        if (index := state.physical_qubit_index(node.name)) is None:
            raise_from_node(node, f"Undefined symbol '{node.name}'.")

        cxt.addressing_mode.set_physical_mode(node)
        num_qubits = cxt.circuit.num_qubits
        new_identifiers = [ast.Identifier(name=f"${i}") for i in range(num_qubits, index + 1)]
        new_bits = [Qubit() for _ in new_identifiers]
        hardware_qubit = types.HardwareQubit()
        for name, bit in zip(new_identifiers, new_bits):
            cxt.symbol_table.insert(Symbol(name.name, bit, hardware_qubit, Scope.GLOBAL, None))
        cxt.circuit.add_bits(new_bits)
        return new_bits[-1], hardware_qubit

    def visit_IntegerLiteral(self, node: ast.IntegerLiteral):
        return node.value, types.Int(const=True)

    def visit_FloatLiteral(self, node: ast.FloatLiteral):
        return node.value, types.Float(const=True)

    def visit_BooleanLiteral(self, node: ast.BooleanLiteral):
        return node.value, types.Bool(const=True)

    def visit_BitstringLiteral(self, node: ast.BitstringLiteral):
        return node.value, types.Uint(const=True, size=node.width)

    def visit_DurationLiteral(self, node: ast.DurationLiteral):
        return (node.value, node.unit.name), types.Duration(const=True)

    def visit_DiscreteSet(self, node: ast.DiscreteSet):
        if not node.values:
            return (), types.Sequence(types.Never())
        set_type: _IntegerT = types.Never()
        values = []
        for expr in node.values:
            expr_value, expr_type = self.visit(expr)
            if not isinstance(expr_type, (types.Int, types.Uint)) or not expr_type.const:
                raise_from_node(
                    expr, f"sequence values must be const int or uint, not '{expr_type.pretty()}'"
                )
            set_type = join_integer_types(set_type, expr_type)
            values.append(expr_value)
        return tuple(values), types.Sequence(set_type)

    def visit_RangeDefinition(self, node: ast.RangeDefinition):
        start, start_type = (None, types.Never()) if node.start is None else self.visit(node.start)
        end, end_type = (None, types.Never()) if node.end is None else self.visit(node.end)
        step, step_type = (None, types.Never()) if node.step is None else self.visit(node.step)
        range_type = join_integer_types(start_type, end_type)
        if not (range_type == types.Never() or range_type.const) or not (
            step_type == types.Never() or step_type.const
        ):
            raise_from_node(node, "can only handle constant ranges")
        if end is not None:
            # OpenQASM 3 ranges are double-end inclusive.  This isn't perfect, but good enough for
            # anything we're actually supporting.
            positive = step_type == types.Never() or step > 0
            end = end + 1 if positive else end - 1
        return slice(start, end, step), types.Range(range_type)

    def visit_Concatenation(self, node: ast.Concatenation):
        lhs_value, lhs_type = self.visit(node.lhs)
        rhs_value, rhs_type = self.visit(node.rhs)
        if not (
            isinstance(lhs_type, (types.BitArray, types.QubitArray))
            and isinstance(rhs_type, (types.BitArray, types.QubitArray))
        ):
            return None, types.Error()
        if type(lhs_type) != type(rhs_type):
            return None, types.Error()
        out_value = tuple(lhs_value) + tuple(rhs_value)
        return out_value, type(lhs_type)(len(out_value))

    def visit_UnaryExpression(self, node: ast.UnaryExpression):
        # In all this, we're only supporting things that we can actually output; `~` for example is
        # supported on `Bit` and `BitArray`, but Qiskit doesn't have any representation of the
        # literals for those or the actual operation on `Clbit` / `ClassicalRegister`, so we can
        # just error out if we see that.
        value, type = self.visit(node.expression)
        if node.op is ast.UnaryOperator["-"]:
            if not isinstance(type, (types.Int, types.Angle, types.Float)):
                raise_from_node(
                    node,
                    f"unary '-' is supported for int, angle and float, not '{type.pretty()}'",
                )
            return (-value), type
        raise_from_node(node, f"unhandled unary operator '{node.op.name}'")

    def visit_BinaryExpression(self, node: ast.BinaryExpression):
        if node.op.name not in ("+", "-", "*", "/"):
            raise_from_node(node, f"unsupported binary operation '{node.op.name}'")
        lhs_value, lhs_type = self.visit(node.lhs)
        rhs_value, rhs_type = self.visit(node.rhs)
        out_type: Union[None, types.Type] = None

        # First, handle the simple implicit promotion rules for the standard numeric types (i.e. not
        # angle).
        numeric_t = (types.Int, types.Uint, types.Float)
        if isinstance(lhs_type, numeric_t) and isinstance(rhs_type, numeric_t):
            out_type = join_numeric_types(lhs_type, rhs_type)

        if node.op.name in ("+", "-"):
            if isinstance(lhs_type, types.Angle) and isinstance(rhs_type, types.Angle):
                const = lhs_type.const and rhs_type.const
                size = (
                    None
                    if lhs_type.size is None or rhs_type.size is None
                    else max((lhs_type.size, rhs_type.size))
                )
                out_type = types.Angle(const, size)
            elif not self._strict and (
                (isinstance(lhs_type, types.Angle) and isinstance(rhs_type, types.Float))
                or (isinstance(rhs_type, types.Angle) and isinstance(lhs_type, types.Float))
            ):
                const = lhs_type.const and rhs_type.const
                size = lhs_type.size if isinstance(lhs_type, types.Angle) else rhs_type.size
                out_type = types.Angle(const, size)
            if out_type is not None:
                out_value = (
                    lhs_value + rhs_value
                    if node.op is ast.BinaryOperator["+"]
                    else lhs_value - rhs_value
                )
                return out_value, out_type

        elif node.op is ast.BinaryOperator["/"]:
            if isinstance(lhs_type, types.Angle):
                if isinstance(rhs_type, (types.Int, types.Uint)):
                    const = lhs_type.const and rhs_type.const
                    out_type = types.Angle(const, lhs_type.size)
                elif isinstance(rhs_type, types.Angle):
                    const = lhs_type.const and rhs_type.const
                    out_type = types.Uint(const, None)
                # We allow `angle / float` in non-strict mode, because the Qiskit OQ3 exporter does
                # not / cannot handle `ParameterExpression` well on output, and often outputs things
                # like that.  That's invalid OQ3, but it's better to support Qiskit's dodgy output
                # than to complain to the user about it.
                elif not self._strict and isinstance(rhs_type, types.Float):
                    const = lhs_type.const and rhs_type.const
                    out_type = types.Angle(const, lhs_type.size)
            if out_type is not None:
                out_value = (
                    lhs_value // rhs_value
                    if isinstance(out_type, (types.Int, types.Uint)) and out_type.const
                    else lhs_value / rhs_value
                )
                return out_value, out_type

        elif node.op is ast.BinaryOperator["*"]:
            # We allow `angle * float` in non-strict mode, because the Qiskit OQ3 exporter does not
            # / cannot handle `ParameterExpression` well on output, and often outputs things like
            # that.  That's invalid OQ3, but it's better to support Qiskit's dodgy output than to
            # complain to the user about it.
            other_types = (
                (types.Int, types.Uint) if self._strict else (types.Int, types.Uint, types.Float)
            )
            if (isinstance(lhs_type, types.Angle) and isinstance(rhs_type, other_types)) or (
                isinstance(rhs_type, types.Angle) and isinstance(lhs_type, other_types)
            ):
                const = lhs_type.const and rhs_type.const
                size = lhs_type.size if isinstance(lhs_type, types.Angle) else rhs_type.size
                out_type = types.Angle(const, size)
            if out_type is not None:
                out_value = lhs_value * rhs_value
                return out_value, out_type

        return None, types.Error()

    def _index_collection(
        self,
        collection: Any,
        collection_type: types.Type,
        indexer: ast.IndexElement,
        base: ast.QASMNode,
    ) -> Tuple[Any, types.Type]:
        if not isinstance(collection_type, (types.BitArray, types.QubitArray)):
            raise_from_node(
                base,
                f"only indexing (qu)bit arrays is supported, not '{collection_type.pretty()}'",
            )
        if isinstance(indexer, ast.DiscreteSet):
            set_values, set_type = self.visit(indexer)
            if not set_values:
                return [], type(collection_type)(0)
            if not (isinstance(set_type.base, (types.Int, types.Uint)) and set_type.base.const):
                raise_from_node(
                    indexer, f"only const (u)int can be indices, not '{set_type.pretty()}'"
                )
            return [collection[x] for x in set_values], type(collection_type)(len(set_values))
        if len(indexer) != 1:
            raise_from_node(base, "only 1D indexers are supported")
        index_value, index_type = self.visit(indexer[0])
        if isinstance(index_type, types.Range):
            value = collection[index_value]
            return value, type(collection_type)(len(value))
        if isinstance(index_type, (types.Int, types.Uint)):
            value = collection[index_value]
            return value, (types.Bit() if isinstance(value, Clbit) else types.Qubit())
        raise_from_node(base, f"unsupported index type: '{index_type.pretty()}'")

    def visit_IndexExpression(self, node: ast.IndexExpression):
        return self._index_collection(*self.visit(node.collection), node.index, node)

    def visit_IndexedIdentifier(self, node: ast.IndexedIdentifier):
        collection, collection_type = self.visit(node.name)
        for index in node.indices:
            collection, collection_type = self._index_collection(
                collection, collection_type, index, node
            )
        return collection, collection_type


def resolve_condition(
    node: ast.Expression, context: state.State
) -> Union[Tuple[Clbit, bool], Tuple[Iterable[Clbit], int]]:
    """A resolver for conditions that can be converted into Qiskit's very basic equality form
    of either ``Clbit == bool`` or ``ClassicalRegister == int``.

    This effectively just handles very special outer cases, then delegates the rest of the work to a
    :class:`.ValueResolver`."""

    value_resolver = ValueResolver(context, strict=True)

    if isinstance(node, ast.BinaryExpression):
        if node.op not in (ast.BinaryOperator["=="], ast.BinaryOperator["!="]):
            raise_from_node(node, f"unhandled binary operator '{node.op.name}'")
        lhs_value, lhs_type = value_resolver.visit(node.lhs)
        rhs_value, rhs_type = value_resolver.visit(node.rhs)
        bad_type_message = (
            "conditions must be 'bit == const bool' or 'bitarray == const int',"
            f" not '{lhs_type.pretty()} {node.op.name} {rhs_type.pretty()}'"
        )
        if isinstance(lhs_type, (types.Bit, types.BitArray)):
            bit_value, bit_type = lhs_value, lhs_type
            cmp_value, cmp_type = rhs_value, rhs_type
        elif isinstance(rhs_type, (types.Bit, types.BitArray)):
            bit_value, bit_type = rhs_value, rhs_type
            cmp_value, cmp_type = lhs_value, lhs_type
        else:
            raise_from_node(node, bad_type_message)
        if isinstance(bit_type, types.Bit):
            if not (isinstance(cmp_type, types.Bool) and cmp_type.const):
                raise_from_node(node, bad_type_message)
            if node.op is ast.BinaryOperator["!="]:
                cmp_value = not cmp_value
            return (bit_value, cmp_value)
        if not (isinstance(cmp_type, (types.Int, types.Uint)) and cmp_type.const):
            raise_from_node(node, bad_type_message)
        if node.op is ast.BinaryOperator["!="]:
            raise_from_node(node, "only '==' is supported in register comparisons")
        return (bit_value, cmp_value)
    if isinstance(node, ast.UnaryExpression):
        if node.op not in (ast.UnaryOperator["~"], ast.UnaryOperator["!"]):
            raise_from_node(node, f"unhandled unary operator '{node.op.name}'")
        value, type = value_resolver.visit(node.expression)
        if isinstance(type, types.Bit):
            return (value, False)
    else:
        value, type = value_resolver.visit(node)
        if isinstance(type, types.Bit):
            return (value, True)
    raise_from_node(
        node,
        f"conditions must be 'bit == const bool' or 'bitarray == const int', not '{type.pretty()}'",
    )
