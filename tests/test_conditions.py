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

import pytest

from openqasm3 import ast
from qiskit.circuit import Clbit, ClassicalRegister, Parameter

from qiskit_qasm3_import import types, ConversionError
from qiskit_qasm3_import.data import Symbol, Scope
from qiskit_qasm3_import.expression import resolve_condition
from qiskit_qasm3_import.converter import State


def _equal(left: ast.Expression, right: ast.Expression):
    return ast.BinaryExpression(ast.BinaryOperator["=="], left, right)


def _make_context(symbols=None):
    context = State()
    if symbols is not None:
        for sym in symbols:
            context.symbol_table.insert(sym)
    return context


@pytest.mark.parametrize("result", (True, False))
def test_bit_resolution(result):
    bit = Clbit()
    symbols = [
        Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(ast.Identifier("a"), ast.BooleanLiteral(result))
    assert resolve_condition(node, context) == (bit, result)

    node = _equal(ast.BooleanLiteral(result), ast.Identifier("a"))
    assert resolve_condition(node, context) == (bit, result)


@pytest.mark.parametrize("result", (True, False))
def test_bit_negative_resolution(result):
    bit = Clbit()
    symbols = [
        Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = ast.BinaryExpression(
        ast.BinaryOperator["!="], ast.Identifier("a"), ast.BooleanLiteral(not result)
    )
    assert resolve_condition(node, context) == (bit, result)

    node = ast.BinaryExpression(
        ast.BinaryOperator["!="],
        ast.BooleanLiteral(not result),
        ast.Identifier("a"),
    )
    assert resolve_condition(node, context) == (bit, result)


def test_implicit_bit():
    bit = Clbit()
    symbols = [
        Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = ast.Identifier("a")
    assert resolve_condition(node, context) == (bit, True)


@pytest.mark.parametrize("op", ("~", "!"))
def test_implicit_negated_bit(op):
    bit = Clbit()
    symbols = [
        Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = ast.UnaryExpression(ast.UnaryOperator[op], ast.Identifier("a"))
    assert resolve_condition(node, context) == (bit, False)


def test_incorrect_unary_operator():
    bit = Clbit()
    symbols = [
        Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = ast.UnaryExpression(ast.UnaryOperator["-"], ast.Identifier("a"))
    with pytest.raises(ConversionError, match="unhandled unary operator"):
        resolve_condition(node, context)


def test_incorrect_binary_operator():
    bit = Clbit()
    symbols = [
        Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = ast.BinaryExpression(ast.BinaryOperator["-"], ast.Identifier("a"), ast.Identifier("a"))
    with pytest.raises(ConversionError, match="unhandled binary operator"):
        resolve_condition(node, context)


def test_reject_nonbit_condition():
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(ast.BooleanLiteral(True), _make_context({}))


def test_index_to_bit():
    register = ClassicalRegister(3)
    symbols = [
        Symbol("a", register, types.BitArray(5), Scope.GLOBAL),
        Symbol("b", 1, types.Int(const=True), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(
        ast.IndexExpression(ast.Identifier("a"), [ast.Identifier("b")]), ast.BooleanLiteral(True)
    )
    assert resolve_condition(node, context) == (register[1], True)


def test_non_bit_comparison():
    symbols = [
        Symbol("a", Parameter("a"), types.Int(const=False), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(ast.IntegerLiteral(1), ast.Identifier("a"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, context)
    node = _equal(ast.Identifier("a"), ast.IntegerLiteral(1))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, context)


def test_reject_compare_bit_to_non_bool():
    symbols = [
        Symbol("a", Clbit(), types.Bit(), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(ast.IntegerLiteral(1), ast.Identifier("a"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, context)
    node = _equal(ast.Identifier("a"), ast.IntegerLiteral(1))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, context)


def test_creg_to_int():
    creg = ClassicalRegister(3)
    symbols = [
        Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(ast.Identifier("a"), ast.IntegerLiteral(1))
    assert resolve_condition(node, context) == (creg, 1)
    node = _equal(ast.IntegerLiteral(1), ast.Identifier("a"))
    assert resolve_condition(node, context) == (creg, 1)


def test_creg_rejects_unequal():
    creg = ClassicalRegister(3)
    symbols = [
        Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = ast.BinaryExpression(
        ast.BinaryOperator["!="], ast.Identifier("a"), ast.IntegerLiteral(1)
    )
    with pytest.raises(ConversionError, match="only '==' is supported"):
        resolve_condition(node, context)


def test_creg_rejects_nonconst():
    creg = ClassicalRegister(3)
    symbols = [
        Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
        Symbol("b", Parameter("b"), types.Int(const=False), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(ast.Identifier("a"), ast.Identifier("b"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, context)


def test_creg_rejects_noninteger():
    creg = ClassicalRegister(3)
    symbols = [
        Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
        Symbol("b", Parameter("b"), types.Float(const=True), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(ast.Identifier("a"), ast.Identifier("b"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, context)


def test_implicit_creg():
    creg = ClassicalRegister(3)
    symbols = [
        Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
    ]
    context = _make_context(symbols)
    node = _equal(
        ast.IndexExpression(
            ast.Identifier("a"), ast.DiscreteSet([ast.IntegerLiteral(0), ast.IntegerLiteral(2)])
        ),
        ast.IntegerLiteral(3),
    )
    assert resolve_condition(node, context) == ([creg[0], creg[2]], 3)
