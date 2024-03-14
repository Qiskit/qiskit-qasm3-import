# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may obtain a copy of this license
# in the LICENSE file in the root directory of this source tree or at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this copyright notice, and modified
# files need to carry a notice indicating that they have been altered from the originals.

import math

import numpy as np
import pytest

from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Clbit,
    Qubit,
)
from qiskit.quantum_info import Operator
from qiskit.transpiler import TranspileLayout, Layout

from qiskit_qasm3_import import parse, ConversionError


def test_readme_circuit():
    # No real test here as there's too much variance in the control-flow builders, and it's tricky
    # to make things like the `Parameter` instances do the equality checks we want.  This is just to
    # test that the circuit in the README can parse and convert.
    source = """
OPENQASM 3.0;
// The 'stdgates.inc' is supported, and the gates are only available if it
// has correctly been included.
include "stdgates.inc";

// Parametrised inputs are supported.
input float[64] a;

qubit[3] q;
bit[2] mid;
bit[3] out;

// Aliasing and re-aliasing are supported.
let aliased = q[0:1];

// Parametrised gates that make use of the stdlib.
gate my_gate(a) c, t {
  gphase(a / 2);
  ry(a) c;
  cx c, t;
}

// Gate modifiers work as well; this gate is equivalent to `p(-a) c;`.
gate my_phase(a) c {
  ctrl @ inv @ gphase(a) c;
}

// We handle mathematical expressions on gate creation and complex indexing
// of temporary collections.
my_gate(a * 2) aliased[0], q[{1, 2}][0];
measure q[0] -> mid[0];
measure q[1] -> mid[1];

while (mid == "00") {
  reset q[0];
  reset q[1];
  my_gate(a) q[0], q[1];
  // We support the builtin mathematical symbols.
  my_phase(a - pi/2) q[1];
  mid[0] = measure q[0];
  mid[1] = measure q[1];
}

// The condition resolver can also handle simple cases that don't look
// _exactly_ like equality conditions.
if (mid[0]) {
  // There is limited support for aliasing within nested scopes.
  let inner_alias = q[{0, 1}];
  reset inner_alias;
}

out = measure q;
    """
    parse(source)


def test_readme_circuit_physical_qubits():
    source = """
OPENQASM 3.0;
include "stdgates.inc";

input float[64] a;

bit[2] mid;
bit[3] out;

gate my_gate(a) c, t {
  gphase(a / 2);
  ry(a) c;
  cx c, t;
}

gate my_phase(a) c {
  ctrl @ inv @ gphase(a) c;
}

my_gate(a * 2) $0, $1;
measure $0 -> mid[0];
measure $1 -> mid[1];

while (mid == "00") {
  reset $0;
  reset $1;
  my_gate(a) $0, $1;
  my_phase(a - pi/2) $1;
  mid[0] = measure $0;
  mid[1] = measure $1;
}

out[0] = measure $0;
out[1] = measure $1;
out[2] = measure $2;
    """
    parse(source)


def test_include_rejects_non_stdgates():
    source = "include 'unknown.qasm';"
    with pytest.raises(ConversionError, match="non-stdgates imports not currently supported"):
        parse(source)


def test_include_stdgates():
    source = """
        include 'stdgates.inc';
        qubit q;
        h q;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()])
    expected.h(0)
    assert qc == expected


def test_stdgates_not_implicitly_included():
    source = """
        qubit q;
        h q;
    """
    with pytest.raises(ConversionError, match="gate 'h' is not defined"):
        parse(source)


def test_undefined_symbol():
    source = """
       gate my_gate q {
          U(0, new_symbol, 0) q;
       }
    """
    with pytest.raises(ConversionError, match="Undefined symbol 'new_symbol'"):
        parse(source)


def test_qubit_declarations():
    source = """
        qubit q;
        qubit[5] q1;
        qreg q2[3];
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()], QuantumRegister(5, "q1"), QuantumRegister(3, "q2"))
    assert len(qc.qubits) == 9
    assert qc.qregs == expected.qregs
    for left, right in zip(qc.qubits, expected.qubits):
        assert qc.find_bit(left) == expected.find_bit(right)


def test_undeclared_physical_qubit():
    source = """
        reset $1;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit(), Qubit()])
    expected.reset(1)
    assert len(qc.qubits) == len(expected.qubits)
    assert qc.qregs == expected.qregs
    assert qc == expected


def test_undeclared_physical_qubits_with_gaps():
    """We should output a circuit that has as many qubits as the highest physical qubit used, since
    Qiskit only represents physical qubits by integer indices."""
    source = """
        include "stdgates.inc";
        bit[2] c;
        h $3;
        cx $5, $3;
        c[0] = measure $3;
        c[1] = measure $5;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit() for _ in range(6)], ClassicalRegister(2, "c"))
    expected.h(3)
    expected.cx(5, 3)
    expected.measure([3, 5], [0, 1])
    assert qc == expected

    # Note we have to use the 'Qubit' instances of the parsed circuit when comparing layouts, since
    # that's outside the context of the full comparison.
    expected_layout = TranspileLayout(
        initial_layout=Layout.from_qubit_list(qc.qubits),
        input_qubit_mapping={bit: i for i, bit in enumerate(qc.qubits)},
        final_layout=None,
    )
    assert qc.layout == expected_layout


def test_undeclared_physical_qubits_in_control_flow():
    source = """
        include "stdgates.inc";
        bit[2] c;
        if (c[0]) {
            h $7;
        }
        while (c == 0) {
            while (!c[0]) {
                h $3;
                cx $3, $9;
                c[0] = measure $3;
                c[1] = measure $9;
            }
        }
        h $9;
    """
    qc = parse(source)
    cr = ClassicalRegister(2, "c")
    expected = QuantumCircuit([Qubit() for _ in range(10)], cr)
    with expected.if_test((cr[0], True)):
        expected.h(7)
    with expected.while_loop((cr, 0)):
        with expected.while_loop((cr[0], False)):
            expected.h(3)
            expected.cx(3, 9)
            expected.measure([3, 9], [0, 1])
    expected.h(9)
    assert qc == expected

    expected_layout = TranspileLayout(
        initial_layout=Layout.from_qubit_list(qc.qubits),
        input_qubit_mapping={bit: i for i, bit in enumerate(qc.qubits)},
        final_layout=None,
    )
    assert qc.layout == expected_layout


def test_physical_qubit_stdgates():
    source = """
        include 'stdgates.inc';
        h $0;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()])
    expected.h(0)
    assert qc == expected


def test_clbit_declarations():
    source = """
        bit c;
        bit[5] c1;
        creg c2[3];
    """
    qc = parse(source)
    expected = QuantumCircuit([Clbit()], ClassicalRegister(5, "c1"), ClassicalRegister(3, "c2"))
    assert len(qc.clbits) == 9
    assert qc.cregs == expected.cregs
    for left, right in zip(qc.clbits, expected.clbits):
        assert qc.find_bit(left) == expected.find_bit(right)


def test_bad_declaration_types_raise():
    source = """
        int x;
    """
    with pytest.raises(ConversionError, match="declarations of type 'int' are not supported"):
        parse(source)


def test_clbit_declaration_measurement():
    source = """
        qubit q;
        bit c = measure q;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit(), Clbit()])
    expected.measure(0, 0)
    assert qc == expected


def test_declaration_initializers_raise():
    source = """
        bit c = 0;
    """
    with pytest.raises(ConversionError, match="initialisation of classical bits is not supported"):
        parse(source)


def test_output_keyword_rejected():
    source = """
        output bit c;
    """
    with pytest.raises(ConversionError, match="the 'output' keyword is not supported"):
        parse(source)


def test_input_allows_only_supported_types():
    source = """
        input bit c;
    """
    with pytest.raises(ConversionError, match="only .* inputs are supported"):
        parse(source)


def test_input_defines_parameter():
    source = """
        input float a;
        qubit q;
        U(a, a, a) q;
    """
    qc = parse(source)
    assert len(qc.parameters) == 1
    assert qc.parameters[0].name == "a"
    p = qc.parameters[0]
    expected = QuantumCircuit([Qubit()])
    expected.u(p, p, p, 0)
    assert qc == expected


def test_invalid_register_names_are_escaped():
    """Terra as of 0.22 only allows registers with valid OQ2 identifiers as names.  This restriction
    may be relaxed following Qiskit/qiskit-terra#9100."""
    source = """
        bit[2] _prefix_invalid;
        qubit[2] invalid_char_π;
        _prefix_invalid = measure invalid_char_π;
    """
    qc = parse(source)
    assert len(qc.cregs) == 1
    assert len(qc.clbits) == 2
    assert len(qc.qregs) == 1
    assert len(qc.qubits) == 2
    expected = QuantumCircuit(qc.qregs[0], qc.cregs[0])
    expected.measure(expected.qubits, expected.clbits)
    assert qc == expected


def test_register_sizes_fold_constants():
    source = """
        bit[2 * 4 + 8] c;
        qubit[8 / 2 - 1] q;
    """
    qc = parse(source)
    assert qc.cregs == [ClassicalRegister(16, "c")]
    assert qc.qregs == [QuantumRegister(3, "q")]


def test_register_sizes_reject_bad_types():
    source = """
        bit[3.0] c;
    """
    with pytest.raises(ConversionError, match="required a constant integer"):
        parse(source)


def test_only_global_declarations():
    source = """
        for int x in {2, 3} {
            bit[1] c;
        }
    """
    with pytest.raises(ConversionError, match="only global declarations are supported"):
        parse(source)


def test_no_rebinding_in_global_scope():
    source = """
        include 'stdgates.inc';
        qubit p;
    """
    with pytest.raises(ConversionError, match="already inserted in symbol table"):
        parse(source)


def test_basic_gate_definition():
    source = """
        include 'stdgates.inc';

        qubit[2] q;

        gate my_gate q0, q1 {
            h q0;
            cx q0, q1;
        }

        my_gate q[0], q[1];
    """
    qc = parse(source)
    assert len(qc.data) == 1
    assert qc.data[0].operation.name == "my_gate"
    assert qc.data[0].qubits == tuple(qc.qubits)

    expected = QuantumCircuit([Qubit(), Qubit()])
    expected.h(0)
    expected.cx(0, 1)
    assert qc.data[0].operation.definition == expected


def test_parameter_shadows_global_1():
    source = """
        include 'stdgates.inc';

        qubit q;

        gate my_gate(p) q0 {
            U(0, p, 0) q0;
        }

        my_gate(4.5) q;
    """
    qc = parse(source)

    expected = QuantumCircuit([Qubit()])
    expected.u(0, 4.5, 0, 0)
    assert qc.data[0].operation.definition == expected


def test_parameter_shadows_global_2():
    source = """
        include 'stdgates.inc';

        qubit q;

        gate my_gate(p) q0 {
            U(0, p, 0) q0;
            p q0;
        }
    """
    with pytest.raises(ConversionError, match="not a gate"):
        parse(source)


def test_parameter_shadows_builtin():
    source = """
        include 'stdgates.inc';

        qubit q;

        gate my_gate(pi) q0 {
            U(0, pi, 0) q0;
        }

        my_gate(4.5) q;
    """
    qc = parse(source)

    expected = QuantumCircuit([Qubit()])
    expected.u(0, 4.5, 0, 0)
    assert qc.data[0].operation.definition == expected


def test_input_shadows_builtin():
    source = """
        input float euler;
        qubit q;
        U(euler, euler, euler) q;
    """
    with pytest.raises(
        ConversionError, match="Symbol 'euler' already inserted in symbol table in this scope"
    ):
        parse(source)
    # qc = parse(source)
    # assert len(qc.parameters) == 1
    # assert qc.parameters[0].name == "euler"
    # p = qc.parameters[0]
    # expected = QuantumCircuit([Qubit()])
    # expected.u(p, p, p, 0)
    # assert qc == expected


def test_parametrised_gate_definition():
    source = """
        include 'stdgates.inc';

        qubit[2] q;

        gate my_gate(p) q0, q1 {
            U(0, p, 0) q0;
            cx q0, q1;
        }

        my_gate(4.5) q[0], q[1];
    """
    qc = parse(source)
    assert len(qc.data) == 1
    assert qc.data[0].operation.name == "my_gate"
    assert qc.data[0].qubits == tuple(qc.qubits)

    expected = QuantumCircuit([Qubit(), Qubit()])
    expected.u(0, 4.5, 0, 0)
    expected.cx(0, 1)
    assert qc.data[0].operation.definition == expected


def test_gate_cannot_redefine():
    source = """
        gate x q {
        }
        gate x q {
        }
    """
    with pytest.raises(ConversionError, match="'x' is already defined"):
        parse(source)


def test_global_phase():
    source = """
        qubit q;
        gphase(2);
    """
    qc = parse(source)
    assert len(qc.data) == 0
    assert qc.global_phase == 2


def test_broadcast_global_phase():
    source = """
        qubit[2] q;
        ctrl @ gphase(2) q;
    """
    qc = parse(source)
    assert len(qc.data) == 2
    assert qc.data[0].operation == qc.data[1].operation
    expected = QuantumCircuit(2)
    expected.p(2, expected.qubits)
    assert Operator(qc) == Operator(expected)


def test_inverse_global_phase():
    source = """
        qubit q;
        inv @ gphase(1.5);
    """
    qc = parse(source)
    assert len(qc.data) == 0
    assert qc.global_phase == 2 * math.pi - 1.5


def test_gate_definition_scope_limited():
    source = """
        input float x;
        gate my_gate q {
            U(x, 0, 0) q;
        }
    """
    with pytest.raises(ConversionError, match="not visible in the scope of a"):
        parse(source)


def test_gate_call_rejects_nongate():
    source = """
        input float x;
        qubit q;
        x q;
    """
    with pytest.raises(ConversionError, match="'x' is a 'float', not a gate"):
        parse(source)


def test_control_modifier():
    source = """
        include "stdgates.inc";
        qubit[3] q;

        ctrl @ x q[0], q[1];
        ctrl(2) @ x q[0], q[1], q[2];
        ctrl @ ctrl @ x q[0], q[1], q[2];

        negctrl @ x q[0], q[1];
    """
    qc = parse(source)
    expected = QuantumCircuit(QuantumRegister(3, "q"))
    expected.cx(0, 1)
    expected.ccx(0, 1, 2)
    expected.ccx(0, 1, 2)
    expected.cx(0, 1, ctrl_state=0)
    assert qc == expected


def test_pow_modifier():
    source = """
        include "stdgates.inc";
        qubit q;
        pow(2) @ z q;
    """
    qc = parse(source)
    assert Operator(qc) == Operator(np.eye(2))


def test_pow_modifier_rejects_bad_types():
    source = """
        include "stdgates.inc";
        qubit q;
        pow(12dt) @ z q;
    """
    with pytest.raises(ConversionError, match="required a constant floating-point number"):
        parse(source)


def test_gate_broadcast():
    source = """
        include "stdgates.inc";
        qubit[2] q;
        qubit[2] p0;
        cx q[0], p0;
        cx q, p0;
        cx q, p0[0];
    """
    qc = parse(source)
    q, p = QuantumRegister(2, "q"), QuantumRegister(2, "p0")
    expected = QuantumCircuit(q, p)
    expected.cx(q[0], p)
    expected.cx(q, p)
    expected.cx(q, p[0])
    assert qc == expected
    # We need to be safe in the event of mutation.  We can be safe either if the object is
    # immutable, or if it's mutable but only present once in the circuit.
    num_safe_mutate = 0
    unsafe_mutate_indices = []
    refs = set()
    for i, instruction in enumerate(qc.data):
        # Be careful: the `mutable` attribute is only Qiskit 0.45+.
        if getattr(instruction.operation, "mutable", True):
            if id(instruction.operation) in refs:
                unsafe_mutate_indices.append(i)
            else:
                num_safe_mutate += 1
                refs.add(id(instruction.operation))
        else:
            num_safe_mutate += 1
    assert not unsafe_mutate_indices
    assert num_safe_mutate == len(qc.data)


def test_gate_broadcast_rejects_bad_lengths():
    source = """
        include "stdgates.inc";
        qubit[2] q;
        qubit[3] p0;
        cx q, p0;
    """
    with pytest.raises(ConversionError, match="mismatched lengths in gate broadcast"):
        parse(source)


def test_gate_rejects_bad_types():
    source = """
        input angle q;
        U(0, 0, 0) q;
    """
    with pytest.raises(ConversionError, match="required a qubit"):
        parse(source)


def test_gate_rejects_bad_parameter_types():
    source = """
        qubit q;
        U(12dt, 0, 0) q;
    """
    with pytest.raises(ConversionError, match="required an angle-like value"):
        parse(source)


def test_gate_rejects_incorrect_parameters():
    source = """
        gate my_gate(p) q {
        }
        qubit q;
        my_gate q;
    """
    with pytest.raises(ConversionError, match="incorrect number of parameters in call"):
        parse(source)

    source = """
        gate my_gate(p) q {
        }
        qubit q;
        my_gate(0.2, 0.3) q;
    """
    with pytest.raises(ConversionError, match="incorrect number of parameters in call"):
        parse(source)


def test_basic_measurement():
    source = """
        bit[2] c;
        qubit[2] q;
        c = measure q;
        c[0] = measure q[0];
        measure q[1] -> c[1];
    """
    qc = parse(source)
    expected = QuantumCircuit(ClassicalRegister(2, "c"), QuantumRegister(2, "q"))
    expected.measure([0, 1], [0, 1])
    expected.measure(0, 0)
    expected.measure(1, 1)
    assert qc == expected


def test_measure_rejects_no_store():
    source = """
        qubit q;
        measure q;
    """
    with pytest.raises(ConversionError, match="measurements must save their result"):
        parse(source)


def test_measure_rejects_bad_types():
    source = """
        input angle q;
        bit c;
        c = measure q;
    """
    with pytest.raises(ConversionError, match="required a qubit"):
        parse(source)

    source = """
        input angle c;
        qubit q;
        c = measure q;
    """
    with pytest.raises(ConversionError, match="required a bit"):
        parse(source)


def test_barrier():
    source = """
        qubit q;
        qubit[2] qr;
        barrier q;
        barrier qr;
        barrier;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()], QuantumRegister(2, name="qr"))
    expected.barrier(0)
    expected.barrier([1, 2])
    expected.barrier()
    assert qc == expected


def test_barrier_rejects_bad_types():
    source = """
        bit c;
        barrier c;
    """
    with pytest.raises(ConversionError, match="required a qubit"):
        parse(source)


def test_reset():
    source = """
        qubit q;
        qubit[2] qr;
        reset q;
        reset qr;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()], QuantumRegister(2, name="qr"))
    expected.reset(0)
    expected.reset([1, 2])
    assert qc == expected


def test_reset_rejects_bad_types():
    source = """
        bit c;
        reset c;
    """
    with pytest.raises(ConversionError, match="required a qubit"):
        parse(source)


def test_delay():
    source = """
        qubit q;
        qubit[2] qr;
        delay[10dt] q;
        delay[1s] qr;
        delay[1.5ms];
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()], QuantumRegister(2, name="qr"))
    expected.delay(10, 0, unit="dt")
    expected.delay(1, [1, 2], unit="s")
    expected.delay(1.5, unit="ms")
    assert qc == expected


def test_delay_rejects_bad_arguments():
    source = """
        qubit q;
        delay[5] q;
    """
    with pytest.raises(ConversionError, match="required a constant duration"):
        parse(source)


def test_delay_rejects_bad_types():
    source = """
        bit c;
        delay[5dt] c;
    """
    with pytest.raises(ConversionError, match="required a qubit"):
        parse(source)


def test_break():
    source = """
        bit c;
        while (c) {
            break;
        }
    """
    qc = parse(source)
    expected = QuantumCircuit([Clbit()])
    with expected.while_loop((expected.clbits[0], True)):
        expected.break_loop()
    assert qc == expected


def test_continue():
    source = """
        bit c;
        while (c) {
            continue;
        }
    """
    qc = parse(source)
    expected = QuantumCircuit([Clbit()])
    with expected.while_loop((expected.clbits[0], True)):
        expected.continue_loop()
    assert qc == expected


@pytest.mark.parametrize(
    ("condition", "value"),
    (("c", True), ("c == true", True), ("c == false", False), ("~c", False), ("true != c", False)),
)
def test_if_bit(condition, value):
    source = f"""
        qubit q;
        bit c;
        if ({condition}) {{
            U(0, 0, 0) q;
        }}
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit(), Clbit()])
    with expected.if_test((0, value)):
        expected.u(0, 0, 0, 0)
    assert qc == expected


@pytest.mark.parametrize(
    ("condition", "value"),
    (('cr == "00"', 0), ('"00" == cr', 0), ('cr == "11"', 3)),
)
def test_if_register(condition, value):
    source = f"""
        qubit q;
        bit[2] cr;
        if ({condition}) {{
            U(0, 0, 0) q;
        }}
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()], ClassicalRegister(2, "cr"))
    with expected.if_test((expected.cregs[0], value)):
        expected.u(0, 0, 0, 0)
    assert qc == expected


def test_if_implicit_register():
    source = """
        qubit q;
        bit[2] cr;
        if (cr[1:-1:0] == "00") {
            U(0, 0, 0) q;
        }
    """
    qc = parse(source)
    assert len(qc.cregs) == 2
    expected = QuantumCircuit([Qubit()], *qc.cregs)
    with expected.if_test((expected.cregs[1], 0)):
        expected.u(0, 0, 0, 0)
    assert qc == expected


def test_if_else():
    source = """
        qubit q;
        bit c;
        if (c) {
            U(0, 0, 0) q;
        } else {
            U(1.5, 1, -1) q;
        }
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit(), Clbit()])
    with expected.if_test((0, True)) as else_:
        expected.u(0, 0, 0, 0)
    with else_:
        expected.u(1.5, 1, -1, 0)
    assert qc == expected


def test_if_else_does_not_share_scope():
    source = """
        include 'stdgates.inc';
        qubit[2] q;
        bit c;
        if (c) {
            let q1 = q[0:0];
            x q1;
        } else {
            x q1;
        }
    """
    with pytest.raises(ConversionError, match="Undefined symbol 'q1'"):
        parse(source)


def test_if_does_not_leak_scope():
    source = """
        include 'stdgates.inc';
        qubit[2] q;
        bit c;
        if (c) {
            let q1 = q[0:0];
            x q1;
        }
        x q1;
    """
    with pytest.raises(ConversionError, match="Undefined symbol 'q1'"):
        parse(source)


@pytest.mark.parametrize(
    ("condition", "value"),
    (("c", True), ("c == true", True), ("c == false", False), ("~c", False), ("true != c", False)),
)
def test_while_bit(condition, value):
    source = f"""
        qubit q;
        bit c;
        while ({condition}) {{
            U(0, 0, 0) q;
        }}
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit(), Clbit()])
    with expected.while_loop((0, value)):
        expected.u(0, 0, 0, 0)
    assert qc == expected


@pytest.mark.parametrize(
    ("condition", "value"),
    (('cr == "00"', 0), ('"00" == cr', 0), ('cr == "11"', 3)),
)
def test_while_register(condition, value):
    source = f"""
        qubit q;
        bit[2] cr;
        while ({condition}) {{
            U(0, 0, 0) q;
        }}
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()], ClassicalRegister(2, "cr"))
    with expected.while_loop((expected.cregs[0], value)):
        expected.u(0, 0, 0, 0)
    assert qc == expected


def test_for_loop_rejects_bad_type():
    source = """
        for float f in [0:1] {
        }
    """
    with pytest.raises(ConversionError, match="only integer loop variables are supported"):
        parse(source)


def test_for_loop_rejects_bad_collection():
    source = """
        bit c;
        for int i in c {
        }
    """
    with pytest.raises(ConversionError, match="only ranges and discrete integer sets"):
        parse(source)


def test_for_loop_range():
    source = """
        for int i in [0:2] {
        }
    """
    qc = parse(source)
    assert qc.data[0].operation.params[0] == range(0, 3)


def test_for_loop_rejects_bad_range():
    source = """
        for int i in [:] {
        }
    """
    with pytest.raises(ConversionError, match="for-loop ranges must have a start and end"):
        parse(source)


def test_alias():
    source = """
        qubit[3] q;
        bit[3] c;
        let q1 = q;
        let q2 = q[1:1] ++ q[{0, 2}];

        let c1 = c;
        let c2 = c[2:-1:1];
    """
    qc = parse(source)
    q = qc.qregs[0]
    c = qc.cregs[0]
    assert tuple(qc.qregs) == (
        q,
        QuantumRegister(name="q1", bits=list(q)),
        QuantumRegister(name="q2", bits=[q[1], q[0], q[2]]),
    )
    assert tuple(qc.cregs) == (
        c,
        ClassicalRegister(name="c1", bits=list(c)),
        ClassicalRegister(name="c2", bits=[c[2], c[1]]),
    )


def test_alias_in_scope():
    source = """
        include 'stdgates.inc';
        qubit[2] q;
        bit c;
        if (c) {
            let q1 = q[0:0];
            x q1;
        } else {
            let q1 = q[1:1];
            x q1;
        }
    """
    qc = parse(source)
    assert len(qc.qregs) == 3
    assert tuple(qc.qregs[0]) == tuple(qc.qubits)
    assert tuple(qc.qregs[1]) == (qc.qubits[0],)
    assert tuple(qc.qregs[2]) == (qc.qubits[1],)


def test_alias_rejects_bad_types():
    source = """
        let q = 1;
    """
    with pytest.raises(ConversionError, match="aliases must be of registers"):
        parse(source)


def test_reject_mixed_addressing_mode_virtual_first():
    source = """
    qubit q1;
    reset $0;
    """
    with pytest.raises(
        ConversionError, match="Physical qubit referenced in virtual addressing mode"
    ):
        parse(source)


def test_reject_mixed_addressing_mode_hardware_first():
    source = """
    reset $0;
    qubit q1;
    """
    with pytest.raises(ConversionError, match="Virtual qubit declared in physical addressing mode"):
        parse(source)


def test_reject_mixed_addressing_mode_virtual_register():
    source = """
    reset $0;
    qubit[3] q;
    """
    with pytest.raises(ConversionError, match="Virtual qubit declared in physical addressing mode"):
        parse(source)


def test_reject_mixed_addressing_mode_local_scope():
    source = """
    include "stdgates.inc";

    bit[1] mid;

    while(mid == "0") {
      h $0;
      mid[0] = measure $0;
    }

    qubit q;
    """
    with pytest.raises(ConversionError, match="Virtual qubit declared in physical addressing mode"):
        parse(source)


def test_reject_hardware_qubit_in_gate_body_1():
    source = """
        include 'stdgates.inc';

        gate my_gate q {
           h $0;
        }
    """
    with pytest.raises(ConversionError, match="hardware qubits not allowed in gate definitions."):
        parse(source)


def test_reject_hardware_qubit_in_gate_body_2():
    source = """
        include 'stdgates.inc';

        h $0;
        gate my_gate q {
           h $0;
        }
    """
    with pytest.raises(ConversionError, match="hardware qubits not allowed in gate definitions."):
        parse(source)


def test_hardware_mode_and_user_gates():
    source = """
        include 'stdgates.inc';

        reset $0;

        gate my_gate(pi) q0 {
            U(0, pi, 0) q0;
        }

        my_gate(4.5) $0;
    """
    qc = parse(source)
    expected = QuantumCircuit([Qubit()])
    expected.u(0, 4.5, 0, 0)
    assert qc.data[1].operation.definition == expected
