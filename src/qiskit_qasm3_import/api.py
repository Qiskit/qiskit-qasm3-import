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

from __future__ import annotations

import typing

import openqasm3
from qiskit import QuantumCircuit

from .converter import ConvertVisitor

if typing.TYPE_CHECKING:
    from qiskit.circuit import annotation


def convert(
    node: openqasm3.ast.Program,
    *,
    annotation_handlers: dict[str, annotation.OpenQASM3Serializer] | None = None,
) -> QuantumCircuit:
    """Convert a parsed OpenQASM 3 program in AST form, into a Qiskit
    :class:`~qiskit.circuit.QuantumCircuit`.

    :param annotation_handlers: A mapping whose values are the (de)serializers of custom annotation
        objects, and whose associated key is a parent of all the namespaces that that deserializer
        can handle.  The corresponding Qiskit functionality was only added in Qiskit 2.1.

    .. versionadded:: 0.6.0
        The ``annotation_handlers`` parameter.
    """
    return ConvertVisitor(annotation_handlers=annotation_handlers).convert(node).circuit


def parse(
    string: str,
    /,
    *,
    annotation_handlers: dict[str, annotation.OpenQASM3Serializer] | None = None,
) -> QuantumCircuit:
    """Wrapper around :func:`.convert`, which first parses the OpenQASM 3 program into AST form, and
    then converts the output to Qiskit format.

    :param annotation_handlers: A mapping whose values are the (de)serializers of custom annotation
        objects, and whose associated key is a parent of all the namespaces that that deserializer
        can handle.  The corresponding Qiskit functionality was only added in Qiskit 2.1.

    .. versionadded:: 0.6.0
        The ``annotation_handlers`` parameter.
    """
    return convert(openqasm3.parse(string), annotation_handlers=annotation_handlers)
