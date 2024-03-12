# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may obtain a copy of this license
# in the LICENSE file in the root directory of this source tree or at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this copyright notice, and modified
# files need to carry a notice indicating that they have been altered from the originals.

import operator

import pytest

from openqasm3 import ast
from qiskit.circuit import Clbit, Parameter, Qubit

from qiskit_qasm3_import import types, ConversionError
from qiskit_qasm3_import.data import Symbol, Scope
from qiskit_qasm3_import.state import State
from qiskit_qasm3_import.expression import ValueResolver


def _make_context(symbols=None):
    context = State(Scope.GLOBAL)
    if symbols is not None:
        for sym in symbols:
            context.symbol_table.insert(sym)
    return context


@pytest.mark.parametrize(
    ("node", "type"),
    (
        pytest.param(ast.IntegerLiteral(value=1), types.Int(const=True), id="int"),
        pytest.param(ast.FloatLiteral(value=2.3), types.Float(const=True), id="float"),
        pytest.param(ast.BooleanLiteral(value=False), types.Bool(const=True), id="bool"),
        pytest.param(
            ast.BitstringLiteral(value=3, width=4), types.Uint(const=True, size=4), id="bitstring"
        ),
    ),
)
def test_literal_nodes(node, type):
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert resolved_value == node.value
    assert resolved_type == type


@pytest.mark.parametrize(("value", "unit"), ((1.2, "dt"), (4, "us")))
def test_duration_literal(value, unit):
    node = ast.DurationLiteral(value, ast.TimeUnit[unit])
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert resolved_value == (value, unit)
    assert resolved_type == types.Duration(const=True)


def test_identifier():
    symbols = [
        Symbol("a", 1, types.Int(const=True), Scope.GLOBAL),
        Symbol("b", True, types.Bool(const=True), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    for symbol in symbols:
        name = symbol.name
        resolved_value, resolved_type = resolver.resolve(ast.Identifier(name=name))
        assert resolved_value == symbol.data
        assert resolved_type == symbol.type
    with pytest.raises(ConversionError, match="Undefined symbol 'c'"):
        resolver.resolve(ast.Identifier(name="c"))


def test_physical_qubit_identifier():
    resolver = ValueResolver(_make_context())
    for name in ("$0", "$123"):
        q0, q0_type = resolver.resolve(ast.Identifier(name=name))
        assert isinstance(q0, Qubit)
        assert isinstance(q0_type, types.HardwareQubit)


def test_discrete_set_empty():
    node = ast.DiscreteSet(values=[])
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert not tuple(resolved_value)
    assert resolved_type == types.Sequence(types.Never())


def test_discrete_set_int_literals():
    values = (1, 2, 3)
    node = ast.DiscreteSet(values=[ast.IntegerLiteral(value=x) for x in values])
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert resolved_value == values
    assert resolved_type == types.Sequence(types.Int(const=True))


def test_discrete_set_bitstring_literals():
    values = (1, 2, 3)
    node = ast.DiscreteSet(values=[ast.BitstringLiteral(value=x, width=x) for x in values])
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert resolved_value == values
    assert resolved_type == types.Sequence(types.Uint(const=True, size=3))


def test_discrete_set_mixed_literals():
    values = (1, 2, 3)
    node_values = [
        ast.BitstringLiteral(value=1, width=1),
        ast.IntegerLiteral(value=2),
        ast.BitstringLiteral(value=3, width=5),
    ]
    node = ast.DiscreteSet(values=node_values)
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert resolved_value == values
    assert resolved_type == types.Sequence(types.Int(const=True, size=None))


def test_discrete_set_resolves_expressions():
    values = (1, 2, 3)
    symbols = [
        Symbol("a", 1, types.Int(const=True), Scope.GLOBAL),
        Symbol("b", -2, types.Int(const=True), Scope.GLOBAL),
        Symbol("c", 6, types.Int(const=True), Scope.GLOBAL),
        Symbol("d", 2, types.Int(const=True), Scope.GLOBAL),
    ]
    node = ast.DiscreteSet(
        values=[
            ast.Identifier("a"),
            ast.UnaryExpression(op=ast.UnaryOperator["-"], expression=ast.Identifier("b")),
            ast.BinaryExpression(
                op=ast.BinaryOperator["/"], lhs=ast.Identifier("c"), rhs=ast.Identifier("d")
            ),
        ]
    )
    resolved_value, resolved_type = ValueResolver(_make_context(symbols)).resolve(node)
    assert resolved_value == values
    assert resolved_type == types.Sequence(types.Int(const=True))


def test_discrete_set_forbids_disallowed_types():
    symbols = [Symbol("a", Parameter("a"), types.Float(const=False), Scope.GLOBAL)]
    node = ast.DiscreteSet(values=[ast.Identifier("a")])
    resolver = ValueResolver(_make_context(symbols))
    with pytest.raises(ConversionError, match="sequence values must be"):
        resolver.resolve(node)


def test_range_empty():
    node = ast.RangeDefinition(None, None, None)
    resolved_value, resolved_type = ValueResolver(_make_context()).resolve(node)
    assert resolved_value == slice(None, None, None)
    assert resolved_type == types.Range(types.Never())


def test_range_one_ended():
    symbols = [
        Symbol("a", 3, types.Int(const=True), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))

    start = ast.RangeDefinition(start=ast.Identifier("a"), end=None, step=None)
    resolved_value, resolved_type = resolver.resolve(start)
    assert resolved_value == slice(3, None, None)
    assert resolved_type == types.Range(types.Int(const=True))

    end = ast.RangeDefinition(start=None, end=ast.Identifier("a"), step=None)
    resolved_value, resolved_type = resolver.resolve(end)
    # Note OQ3 has double-inclusive ranges, unlike Python.
    assert resolved_value == slice(None, 4, None)
    assert resolved_type == types.Range(types.Int(const=True))


def test_range_two_ended():
    symbols = [
        Symbol("low", 3, types.Uint(const=True, size=6), Scope.GLOBAL),
        Symbol("high", 7, types.Int(const=True, size=4), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))

    node = ast.RangeDefinition(
        start=ast.Identifier("low"),
        end=ast.Identifier("high"),
        step=None,
    )

    resolved_value, resolved_type = resolver.resolve(node)
    assert resolved_value == slice(3, 8, None)
    assert resolved_type == types.Range(types.Int(const=True, size=6))


def test_range_step_does_not_change_type():
    symbols = [
        Symbol("low", 3, types.Uint(const=True, size=6), Scope.GLOBAL),
        Symbol("high", 7, types.Uint(const=True, size=4), Scope.GLOBAL),
        Symbol("step", -2, types.Int(const=True, size=6), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))

    node = ast.RangeDefinition(
        start=ast.Identifier("high"),
        end=ast.Identifier("low"),
        step=ast.Identifier("step"),
    )

    resolved_value, resolved_type = resolver.resolve(node)
    assert resolved_value == slice(7, 2, -2)
    # The step being an `int` shouldn't affect the inferred type of the iterator.
    assert resolved_type == types.Range(types.Uint(const=True, size=6))


def test_range_rejects_non_const():
    symbols = [
        Symbol("a", Parameter("a"), types.Int(const=False), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))

    node = ast.RangeDefinition(start=None, end=ast.Identifier("a"), step=None)
    with pytest.raises(ConversionError, match="can only handle constant ranges"):
        resolver.resolve(node)

    node = ast.RangeDefinition(
        start=ast.IntegerLiteral(2), end=ast.IntegerLiteral(3), step=ast.Identifier("a")
    )
    with pytest.raises(ConversionError, match="can only handle constant ranges"):
        resolver.resolve(node)


@pytest.mark.parametrize(
    ("bit", "array_type"), ((Clbit, types.BitArray), (Qubit, types.QubitArray))
)
def test_concatenation(bit, array_type):
    bits = (bit(), bit())
    symbols = [
        Symbol("a", [bits[0]], array_type(1), Scope.GLOBAL),
        Symbol("b", [bits[1]], array_type(1), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.Concatenation(lhs=ast.Identifier("a"), rhs=ast.Identifier("b"))
    resolved_value, resolved_type = resolver.resolve(node)
    assert tuple(resolved_value) == bits
    assert resolved_type == array_type(2)


def test_concatenation_rejects_mixed_bits():
    symbols = [
        Symbol("a", [Clbit()], types.BitArray(1), Scope.GLOBAL),
        Symbol("b", [Qubit()], types.QubitArray(1), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.Concatenation(lhs=ast.Identifier("a"), rhs=ast.Identifier("b"))
    with pytest.raises(ConversionError, match="type error"):
        resolver.resolve(node)


def test_concatenation_rejects_bad_types():
    symbols = [
        Symbol("a", [Clbit()], types.BitArray(1), Scope.GLOBAL),
        Symbol("b", 1, types.Int(const=True), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.Concatenation(lhs=ast.Identifier("a"), rhs=ast.Identifier("b"))
    with pytest.raises(ConversionError, match="type error"):
        resolver.resolve(node)


def test_unary_minus():
    symbols = [
        Symbol("a", 1, types.Int(const=True), Scope.GLOBAL),
        Symbol("b", Parameter("b"), types.Float(const=False), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    for symbol in symbols:
        name = symbol.name
        node = ast.UnaryExpression(ast.UnaryOperator["-"], ast.Identifier(name))
        resolved_value, resolved_type = resolver.resolve(node)
        assert resolved_value == -symbol.data
        assert resolved_type == symbol.type

        node = ast.UnaryExpression(ast.UnaryOperator["-"], node)
        resolved_value, resolved_type = resolver.resolve(node)
        assert resolved_value == symbol.data
        assert resolved_type == symbol.type


def test_unary_minus_rejects_bad_types():
    resolver = ValueResolver(_make_context())
    with pytest.raises(ConversionError, match="unary '-' is supported"):
        node = ast.UnaryExpression(ast.UnaryOperator["-"], ast.BooleanLiteral(value=True))
        resolver.resolve(node)


@pytest.mark.parametrize(
    ("op", "left_type", "right_type", "out_type"),
    (
        ("+", types.Int(True, 3), types.Int(True, 4), types.Int(True, 4)),
        ("+", types.Int(True, 4), types.Int(False, 3), types.Int(False, 4)),
        ("+", types.Int(True, 4), types.Int(False, None), types.Int(False, None)),
        ("+", types.Float(True, None), types.Float(False, 64), types.Float(False, None)),
        ("+", types.Int(True, None), types.Float(True, 64), types.Float(True, 64)),
        ("+", types.Float(True, None), types.Uint(False, 32), types.Float(False, None)),
        ("+", types.Angle(True, None), types.Angle(False, 8), types.Angle(False, None)),
        ("-", types.Uint(True, 4), types.Int(False, 3), types.Int(False, 4)),
        ("-", types.Uint(True, 4), types.Uint(False, None), types.Uint(False, None)),
        ("-", types.Int(True, None), types.Float(True, 64), types.Float(True, 64)),
        ("-", types.Float(True, None), types.Uint(False, 32), types.Float(False, None)),
        ("-", types.Float(True, 64), types.Float(True, 64), types.Float(True, 64)),
        ("-", types.Angle(True, 4), types.Angle(True, 8), types.Angle(True, 8)),
        ("/", types.Int(True, 5), types.Uint(True, None), types.Int(True, None)),
        ("/", types.Float(True, 64), types.Int(False, None), types.Float(False, 64)),
        ("/", types.Angle(True, 8), types.Int(False, None), types.Angle(False, 8)),
        ("/", types.Angle(True, 8), types.Angle(True, 4), types.Uint(True)),
        ("/", types.Angle(False, 8), types.Angle(True, 4), types.Uint(False)),
        ("/", types.Float(False, 64), types.Float(True, None), types.Float(False, None)),
        ("*", types.Int(True, 5), types.Uint(True, None), types.Int(True, None)),
        ("*", types.Float(True, 64), types.Int(False, None), types.Float(False, 64)),
        ("*", types.Angle(True, 8), types.Int(False, None), types.Angle(False, 8)),
        ("*", types.Float(False, 64), types.Float(True, None), types.Float(False, None)),
        ("*", types.Int(True, None), types.Float(True, 64), types.Float(True, 64)),
        ("*", types.Float(True, None), types.Uint(False, 32), types.Float(False, None)),
    ),
    ids=lambda x: x.pretty() if isinstance(x, types.Type) else x,
)
def test_binary_operator(op, left_type, right_type, out_type):
    a = 144 if left_type.const else Parameter("a")
    b = 36 if right_type.const else Parameter("b")
    symbols = [
        Symbol("a", a, left_type, Scope.GLOBAL),
        Symbol("b", b, right_type, Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.BinaryExpression(ast.BinaryOperator[op], ast.Identifier("a"), ast.Identifier("b"))
    value = getattr(operator, {"+": "add", "-": "sub", "*": "mul", "/": "truediv"}[op])(a, b)
    resolved_value, resolved_type = resolver.resolve(node)
    assert resolved_value == value
    assert resolved_type == out_type


@pytest.mark.parametrize(
    ("op", "left_type", "right_type"),
    (
        ("+", types.Duration(True), types.Float(True, 64)),
        ("+", types.Int(True, 4), types.Angle(True, 4)),
        ("-", types.Angle(True, 4), types.Float(True, 64)),
        ("-", types.Uint(True, 4), types.Angle(False, 3)),
        ("*", types.Angle(True, 4), types.Angle(True, 4)),
        ("*", types.Angle(True, 4), types.Float(True, None)),
        ("*", types.Float(True, None), types.Angle(True, 4)),
        ("/", types.Angle(True, 4), types.Float(True, None)),
        ("/", types.Float(True, 64), types.Angle(True, None)),
        ("/", types.Int(True, 4), types.Angle(True, None)),
    ),
    ids=lambda x: x.pretty() if isinstance(x, types.Type) else x,
)
def test_binary_operator_type_error(op, left_type, right_type):
    a, b = Parameter("a"), Parameter("b")
    symbols = [
        Symbol("a", a, left_type, Scope.GLOBAL),
        Symbol("b", b, right_type, Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.BinaryExpression(ast.BinaryOperator[op], ast.Identifier("a"), ast.Identifier("b"))
    with pytest.raises(ConversionError, match="type error"):
        resolver.resolve(node)


@pytest.mark.parametrize(
    ("bit", "array_type"), ((Clbit, types.BitArray), (Qubit, types.QubitArray))
)
@pytest.mark.parametrize(
    "index_node",
    (
        ast.DiscreteSet([ast.IntegerLiteral(1), ast.IntegerLiteral(2), ast.IntegerLiteral(3)]),
        [ast.RangeDefinition(ast.IntegerLiteral(1), ast.IntegerLiteral(3), None)],
    ),
    ids=["set", "range"],
)
def test_index_expression_collection(bit, array_type, index_node):
    bits = [bit() for _ in [None] * 5]
    node = ast.IndexExpression(ast.Identifier("a"), index_node)
    symbols = [
        Symbol("a", bits, array_type(len(bits)), Scope.GLOBAL),
    ]
    resolved_value, resolved_type = ValueResolver(_make_context(symbols)).resolve(node)
    assert tuple(resolved_value) == (bits[1], bits[2], bits[3])
    assert resolved_type == array_type(3)


@pytest.mark.parametrize(
    ("bit", "scalar_type", "array_type"),
    ((Clbit, types.Bit, types.BitArray), (Qubit, types.Qubit, types.QubitArray)),
)
def test_index_expression_scalar(bit, scalar_type, array_type):
    bits = [bit() for _ in [None] * 5]
    node = ast.IndexExpression(ast.Identifier("a"), [ast.IntegerLiteral(2)])
    symbols = [
        Symbol("a", bits, array_type(len(bits)), Scope.GLOBAL),
    ]
    resolved_value, resolved_type = ValueResolver(_make_context(symbols)).resolve(node)
    assert resolved_value == bits[2]
    assert resolved_type == scalar_type()


@pytest.mark.parametrize(
    ("bit", "array_type"), ((Clbit, types.BitArray), (Qubit, types.QubitArray))
)
def test_index_expression_empty(bit, array_type):
    bits = [bit() for _ in [None] * 5]
    node = ast.IndexExpression(ast.Identifier("a"), ast.DiscreteSet([]))
    symbols = [
        Symbol("a", bits, array_type(len(bits)), Scope.GLOBAL),
    ]
    resolved_value, resolved_type = ValueResolver(_make_context(symbols)).resolve(node)
    assert not tuple(resolved_value)
    assert resolved_type == array_type(0)


def test_index_expression_only_collections():
    symbols = [
        Symbol("a", 1, types.Int(), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.IndexExpression(ast.Identifier("a"), [ast.IntegerLiteral(0)])
    with pytest.raises(ConversionError, match="only indexing .* is supported"):
        resolver.resolve(node)


def test_index_expression_only_1d():
    symbols = [
        Symbol("a", [Qubit(), Qubit()], types.QubitArray(2), Scope.GLOBAL),
    ]
    resolver = ValueResolver(_make_context(symbols))
    node = ast.IndexExpression(ast.Identifier("a"), [ast.IntegerLiteral(0), ast.IntegerLiteral(0)])
    with pytest.raises(ConversionError, match="only 1D indexers are supported"):
        resolver.resolve(node)


@pytest.mark.parametrize(
    ("bit", "array_type"), ((Clbit, types.BitArray), (Qubit, types.QubitArray))
)
def test_indexed_identifier(bit, array_type):
    bits = [bit() for _ in [None] * 5]
    symbols = [
        Symbol("a", bits, array_type(len(bits)), Scope.GLOBAL),
    ]
    node = ast.IndexedIdentifier(
        ast.Identifier("a"),
        indices=[
            ast.DiscreteSet([ast.IntegerLiteral(1), ast.IntegerLiteral(2), ast.IntegerLiteral(3)]),
            [ast.RangeDefinition(ast.IntegerLiteral(1), ast.IntegerLiteral(2), None)],
        ],
    )
    resolved_value, resolved_type = ValueResolver(_make_context(symbols)).resolve(node)
    assert tuple(resolved_value) == (bits[2], bits[3])
    assert resolved_type == array_type(2)


@pytest.mark.parametrize(
    ("bit", "array_type"), ((Clbit, types.BitArray), (Qubit, types.QubitArray))
)
def test_indexed_identifier_no_collections(bit, array_type):
    bits = [bit() for _ in [None] * 5]
    symbols = [
        Symbol("a", bits, array_type(len(bits)), Scope.GLOBAL),
    ]
    node = ast.IndexedIdentifier(ast.Identifier("a"), indices=[])
    resolved_value, resolved_type = ValueResolver(_make_context(symbols)).resolve(node)
    assert tuple(resolved_value) == tuple(bits)
    assert resolved_type == array_type(5)
