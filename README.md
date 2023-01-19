# Importer from OpenQASM 3 to Qiskit

[![License](https://img.shields.io/github/license/Qiskit/qiskit-qasm3-import.svg?style=flat)](https://opensource.org/licenses/Apache-2.0)[![Release](https://img.shields.io/github/release/Qiskit/qiskit-qasm3-import.svg?style=flat)](https://github.com/Qiskit/qiskit-qasm3-import/releases)[![Downloads](https://img.shields.io/pypi/dm/qiskit-qasm3-import.svg?style=flat)](https://pypi.org/project/qiskit-qasm3-import/)

This repository provides the Python package `qiskit_qasm3_import`, which is a
basic and temporary importer from OpenQASM 3 into Qiskit's `QuantumCircuit`.

Qiskit itself accepts this package as an optional dependency if it is installed.
In that case, Qiskit exposes the functions `qiskit.qasm3.load` and
`qiskit.qasm3.loads`, which are wrappers around `qiskit_qasm3_import.parse`.
This project is a stop-gap measure until various technical decisions can be
resolved the correct way; Terra makes strong guarantees of stability and support
in its interfaces, and we are not yet ready to make that commitment for this
project, hence the minimal wrappers.


## Example

The principal entry point to the package is the top-level `parse` function,
which accepts a string containing a complete OpenQASM 3 programme.  This complex
example shows a lot of the capabilities of the importer.

```qasm
OPENQASM 3.0;
// The 'stdgates.inc' include is supported, and the gates are only available
// if it has correctly been included.
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
```

Assuming this program is stored as a string in a variable `program`, we then
import it into a `QuantumCircuit` by doing:

```python
from qiskit_qasm3_import import parse
circuit = parse(program)
```

`circuit` is now a complete `QuantumCircuit`, so we can see exactly what it
turned into:

```python
circuit.draw()
```
```text
       ┌───────────────┐┌─┐   ┌─────────────┐┌──────────┐┌─┐
  q_0: ┤0              ├┤M├───┤0            ├┤0         ├┤M├───
       │  my_gate(2*a) │└╥┘┌─┐│             ││          │└╥┘┌─┐
  q_1: ┤1              ├─╫─┤M├┤1            ├┤1         ├─╫─┤M├
       └──────┬─┬──────┘ ║ └╥┘│             ││  If_else │ ║ └╥┘
  q_2: ───────┤M├────────╫──╫─┤  While_loop ├┤          ├─╫──╫─
              └╥┘        ║  ║ │             ││          │ ║  ║
mid_0: ════════╬═════════╩══╬═╡1            ╞╡0         ╞═╬══╬═
               ║            ║ │             │└──────────┘ ║  ║
mid_1: ════════╬════════════╩═╡0            ╞═════════════╬══╬═
               ║              └─────────────┘             ║  ║
out_0: ════════╬══════════════════════════════════════════╩══╬═
               ║                                             ║
out_1: ════════╬═════════════════════════════════════════════╩═
               ║
out_2: ════════╩═══════════════════════════════════════════════
```


## Installation

Install the latest release of the `qiskit_qasm3_import` package from pip:

```text
pip install qiskit_qasm3_import
```

This will automatically install all the dependencies as well (an OpenQASM 3
parser, for example) if they are not already installed.  Alternatively, you can
install Qiskit Terra directly with this package as an optional dependency by
doing

```text
pip install qiskit-terra[qasm3-import]
```


## Developing

If you're looking to contribute to this project, please first read
[our contributing guidelines](CONTRIBUTING.md).

Set up your development environment by installing the development requirements
with pip:

```bash
pip install -r requirements-dev.txt tox
```

This installs a few more packages than the dependencies of the package at
runtime, because there are some tools we use for testing also included, such as
`tox` and `pytest`.

After the development requirements are installed, you can install an editable
version of the package with

```bash
pip install -e .
```

After this, any changes you make to the library code will immediately be present
when you open a new Python interpreter session.


### Building documentation

After the development requirements have been installed, the command

```bash
tox -e docs
```

will build the HTML documentation, and place it in `docs/_build/html`.  The
documentation state of the `main` branch of this repository is published to
https://qiskit.github.io/qiskit-qasm3-import.


### Code style and linting

The Python components of this repository are formatted using `black`.  You can
run this on the required files by running

```bash
tox -e black
```

The full lint suite can be run with

```bash
tox -e lint
```


## License

This project is licensed under [version 2.0 of the Apache License](LICENSE).
