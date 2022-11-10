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

.. contents::


API
---

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


Internals
---------

Conversion mechanism
....................


.. currentmodule:: qiskit_qasm3_import.converter

The internals of the :func:`.convert` function use a tree visitor, which
subclasses :class:`openqasm3.visitor.QASMVisitor`.  This is principally an
internal implementation detail.

.. autoclass:: ConvertVisitor
   :members: convert


.. currentmodule:: qiskit_qasm3_import.expression

In addition, there is a subvisitor to handle constant-folding and the resolution
of symbols and expressions into equivalent Qiskit types.

.. autoclass:: ValueResolver
   :members: resolve

Conditions are handled slightly differently; to avoid needing to support the
entire type system of OpenQASM 3 before Terra does, a special-case function
handles the allowable components of the expression tree in conditions before
delegating the rest of the calculation to :meth:`.ExpressionResolver.resolve`
above.

.. autofunction:: resolve_condition



.. module:: qiskit_qasm3_import.types

Type system
...........

Internally we use a reduced and modified type system to represent the
intermediate values we are working over.  Everything is a subclass of the
abstract base class.

.. autoclass:: Type
   :members: pretty

There are several different types within this.  Some of these represent the
types of real values that have corresponding Qiskit or just standard Python
objects, and some are types used during inference or to represent errors.

.. autoclass:: Bit
.. autoclass:: BitArray
.. autoclass:: Qubit
.. autoclass:: QubitArray
.. autoclass:: Bool
.. autoclass:: Int
.. autoclass:: Uint
.. autoclass:: Float
.. autoclass:: Angle
.. autoclass:: Duration
.. autoclass:: Range
.. autoclass:: Sequence
.. autoclass:: Gate
.. autoclass:: Error
.. autoclass:: Never
