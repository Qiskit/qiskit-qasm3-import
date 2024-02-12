import qiskit_qasm3_import

project = 'Qiskit OpenQASM 3 Importer'
copyright = '2022, Jake Lishman'
author = 'Jake Lishman'
version = qiskit_qasm3_import.__version__
release = qiskit_qasm3_import.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    'qiskit_sphinx_theme',
]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Document the docstring for the class and the __init__ method together.
autoclass_content = "both"

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"

intersphinx_mapping = {
    "qiskit-terra": ("https://docs.quantum.ibm.com/api/qiskit/", None),
}
