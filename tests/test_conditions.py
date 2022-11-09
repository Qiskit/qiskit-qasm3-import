import pytest

from openqasm3 import ast
from qiskit.circuit import Clbit, ClassicalRegister, Parameter

from qiskit_qasm3_import import types, ConversionError
from qiskit_qasm3_import.data import Symbol, Scope
from qiskit_qasm3_import.expression import resolve_condition


def _equal(left: ast.Expression, right: ast.Expression):
    return ast.BinaryExpression(ast.BinaryOperator["=="], left, right)


@pytest.mark.parametrize("result", (True, False))
def test_bit_resolution(result):
    bit = Clbit()
    symbols = {
        "a": Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    }
    node = _equal(ast.Identifier("a"), ast.BooleanLiteral(result))
    assert resolve_condition(node, symbols) == (bit, result)

    node = _equal(ast.BooleanLiteral(result), ast.Identifier("a"))
    assert resolve_condition(node, symbols) == (bit, result)


@pytest.mark.parametrize("result", (True, False))
def test_bit_negative_resolution(result):
    bit = Clbit()
    symbols = {
        "a": Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    }
    node = ast.BinaryExpression(
        ast.BinaryOperator["!="], ast.Identifier("a"), ast.BooleanLiteral(not result)
    )
    assert resolve_condition(node, symbols) == (bit, result)

    node = ast.BinaryExpression(
        ast.BinaryOperator["!="],
        ast.BooleanLiteral(not result),
        ast.Identifier("a"),
    )
    assert resolve_condition(node, symbols) == (bit, result)


def test_implicit_bit():
    bit = Clbit()
    symbols = {
        "a": Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    }
    node = ast.Identifier("a")
    assert resolve_condition(node, symbols) == (bit, True)


def test_implicit_negated_bit():
    bit = Clbit()
    symbols = {
        "a": Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    }
    node = ast.UnaryExpression(ast.UnaryOperator["~"], ast.Identifier("a"))
    assert resolve_condition(node, symbols) == (bit, False)


def test_incorrect_unary_operator():
    bit = Clbit()
    symbols = {
        "a": Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    }
    node = ast.UnaryExpression(ast.UnaryOperator["-"], ast.Identifier("a"))
    with pytest.raises(ConversionError, match="unhandled unary operator"):
        resolve_condition(node, symbols)


def test_incorrect_binary_operator():
    bit = Clbit()
    symbols = {
        "a": Symbol("a", bit, types.Bit(), Scope.GLOBAL),
    }
    node = ast.BinaryExpression(ast.BinaryOperator["-"], ast.Identifier("a"), ast.Identifier("a"))
    with pytest.raises(ConversionError, match="unhandled binary operator"):
        resolve_condition(node, symbols)


def test_reject_nonbit_condition():
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(ast.BooleanLiteral(True), {})


def test_index_to_bit():
    register = ClassicalRegister(3)
    symbols = {
        "a": Symbol("a", register, types.BitArray(5), Scope.GLOBAL),
        "b": Symbol("b", 1, types.Int(const=True), Scope.GLOBAL),
    }
    node = _equal(
        ast.IndexExpression(ast.Identifier("a"), [ast.Identifier("b")]), ast.BooleanLiteral(True)
    )
    assert resolve_condition(node, symbols) == (register[1], True)


def test_non_bit_comparison():
    symbols = {
        "a": Symbol("a", Parameter("a"), types.Int(const=False), Scope.GLOBAL),
    }
    node = _equal(ast.IntegerLiteral(1), ast.Identifier("a"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, symbols)
    node = _equal(ast.Identifier("a"), ast.IntegerLiteral(1))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, symbols)


def test_reject_compare_bit_to_non_bool():
    symbols = {
        "a": Symbol("a", Clbit(), types.Bit(), Scope.GLOBAL),
    }
    node = _equal(ast.IntegerLiteral(1), ast.Identifier("a"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, symbols)
    node = _equal(ast.Identifier("a"), ast.IntegerLiteral(1))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, symbols)


def test_creg_to_int():
    creg = ClassicalRegister(3)
    symbols = {
        "a": Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
    }
    node = _equal(ast.Identifier("a"), ast.IntegerLiteral(1))
    assert resolve_condition(node, symbols) == (creg, 1)
    node = _equal(ast.IntegerLiteral(1), ast.Identifier("a"))
    assert resolve_condition(node, symbols) == (creg, 1)


def test_creg_rejects_unequal():
    creg = ClassicalRegister(3)
    symbols = {
        "a": Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
    }
    node = ast.BinaryExpression(
        ast.BinaryOperator["!="], ast.Identifier("a"), ast.IntegerLiteral(1)
    )
    with pytest.raises(ConversionError, match="only '==' is supported"):
        resolve_condition(node, symbols)


def test_creg_rejects_nonconst():
    creg = ClassicalRegister(3)
    symbols = {
        "a": Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
        "b": Symbol("b", Parameter("b"), types.Int(const=False), Scope.GLOBAL),
    }
    node = _equal(ast.Identifier("a"), ast.Identifier("b"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, symbols)


def test_creg_rejects_noninteger():
    creg = ClassicalRegister(3)
    symbols = {
        "a": Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
        "b": Symbol("b", Parameter("b"), types.Float(const=True), Scope.GLOBAL),
    }
    node = _equal(ast.Identifier("a"), ast.Identifier("b"))
    with pytest.raises(ConversionError, match="conditions must be"):
        resolve_condition(node, symbols)


def test_implicit_creg():
    creg = ClassicalRegister(3)
    symbols = {
        "a": Symbol("a", creg, types.BitArray(len(creg)), Scope.GLOBAL),
    }
    node = _equal(
        ast.IndexExpression(
            ast.Identifier("a"), ast.DiscreteSet([ast.IntegerLiteral(0), ast.IntegerLiteral(2)])
        ),
        ast.IntegerLiteral(3),
    )
    assert resolve_condition(node, symbols) == ([creg[0], creg[2]], 3)
