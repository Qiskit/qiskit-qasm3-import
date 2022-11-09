OpenQASM 3 Importer for Qiskit
==============================

.. module:: qiskit_qasm3_import

This project is a temporary importer for parsing simple OpenQASM 3 programmes
and converting them into Qiskit :class:`~qiskit.circuit.QuantumCircuit` objects.

We fully expect Qiskit's capability to grow in this field, and for a similar
importer to be merged into mainline Qiskit.  This project is a stop-gap measure,
while various technical details are decided on the Qiskit side, as switching
parsing frameworks and so on is far more complex for code brought into Qiskit,
as the stability guarantees are much stronger.

This package providers two public methods: :func:`.parse` and :func:`.convert`.
The complete path of taking a string representation of an OpenQASM 3 programme
to :class:`~qiskit.circuit.QuantumCircuit` is :func:`.parse`, while
:func:`.convert` takes an AST :class:`~openqasm3.ast.Program` node from `the
reference OpenQASM 3 Python package <https://pypi.org/project/openqasm3>`__, and
converts that to a :class:`~qiskit.circuit.QuantumCircuit`.

.. autofunction:: parse

.. autofunction:: convert

The converter in this module is quite limited, since Qiskit does not yet support
all the features of OpenQASM 3.  When something unsupported is encountered
during an attempted import, a :exc:`.ConversionError` will be raised.

.. autoexception:: ConversionError


.. currentmodule:: qiskit_qasm3_import.convert

The internals of the :func:`.convert` function use a tree visitor, which
subclasses :class:`openqasm3.visitor.QASMVisitor`.  This is principally an
internal implementation detail.

.. autoclass:: ConvertVisitor
   :members: convert
