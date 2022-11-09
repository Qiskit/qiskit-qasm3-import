from typing import NoReturn

from openqasm3 import ast


class ConversionError(Exception):
    """Raised when an error occurs converting from the AST representation into a
    :class:`~qiskit.circuit.QuantumCircuit`.  This is often due to OpenQASM 3 constructs that have
    no equivalent in Qiskit."""


def raise_from_node(node: ast.QASMNode, message: str) -> NoReturn:
    """Raise a :exc:`.ConversionError` caused by the given `node`."""
    if node.span is not None:
        message = f"{node.span.start_line},{node.span.start_column}: {message}"
    raise ConversionError(message)
