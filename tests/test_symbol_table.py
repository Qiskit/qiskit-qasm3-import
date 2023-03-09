import pytest

from qiskit_qasm3_import.data import Symbol, Scope
from qiskit_qasm3_import.state import SymbolTable
from qiskit_qasm3_import import types, ConversionError


def test_insert_symbol_twice_global_scope():
    symtab = SymbolTable(Scope.GLOBAL)
    s = Symbol("x", 1, types.Int, Scope.GLOBAL)
    symtab.insert(s)
    with pytest.raises(
        ConversionError,
        match="Symbol 'x' already inserted in symbol table in this scope: Scope.GLOBAL",
    ):
        symtab.insert(s)
