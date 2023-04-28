import openqasm3
from qiskit import QuantumCircuit

from .converter import ConvertVisitor


def convert(node: openqasm3.ast.Program) -> QuantumCircuit:
    """Convert a parsed OpenQASM 3 program in AST form, into a Qiskit
    :class:`~qiskit.circuit.QuantumCircuit`."""
    return ConvertVisitor().convert(node).circuit


def parse(string: str, /) -> QuantumCircuit:
    """Wrapper around :func:`.convert`, which first parses the OpenQASM 3 program into AST form, and
    then converts the output to Qiskit format."""
    return convert(openqasm3.parse(string))
