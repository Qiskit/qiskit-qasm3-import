from typing import NoReturn, Optional

from openqasm3 import ast


class ConversionError(Exception):
    """Raised when an error occurs converting from the AST representation into a
    :class:`~qiskit.circuit.QuantumCircuit`.  This is often due to OpenQASM 3 constructs that have
    no equivalent in Qiskit."""

    def __init__(self, message, node: Optional[ast.QASMNode] = None):
        if node is not None and node.span is not None:
            message = f"{node.span.start_line},{node.span.start_column}: {message}"
        self.message = message
        super().__init__(message)


def raise_from_node(node: ast.QASMNode, message: str) -> NoReturn:
    """Raise a :exc:`.ConversionError` caused by the given `node`."""
    raise ConversionError(message, node)
