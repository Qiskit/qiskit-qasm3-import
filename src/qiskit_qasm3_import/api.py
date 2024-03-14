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
