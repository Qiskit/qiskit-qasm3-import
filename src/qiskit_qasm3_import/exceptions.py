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
