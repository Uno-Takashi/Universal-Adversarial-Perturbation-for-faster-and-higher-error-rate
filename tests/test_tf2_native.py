"""Guard the TF2 port: no module may reach back into the TF1 graph API.

The check walks the AST rather than grepping, so prose in a docstring cannot trip it.
``classifiers.load_frozen_graph`` is the one sanctioned exception: a frozen ``GraphDef``
has no TF2 entry point other than ``tf.compat.v1.wrap_function``.
"""

import ast
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = sorted(p for p in PROJECT_ROOT.glob("*.py"))

# Attribute names that only exist to drive TF1-style graph execution.
FORBIDDEN = {
    "Session",
    "InteractiveSession",
    "placeholder",
    "placeholder_with_default",
    "disable_eager_execution",
    "get_default_session",
    "global_variables_initializer",
}
ALLOWED_COMPAT = {"wrap_function", "GraphDef"}


def _attribute_names(tree):
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_no_tf1_graph_api(module):
    tree = ast.parse(module.read_text())
    used = _attribute_names(tree) & FORBIDDEN
    assert not used, f"{module.name} uses TF1 graph API: {sorted(used)}"


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_compat_v1_is_confined(module):
    tree = ast.parse(module.read_text())
    prefix = "tf.compat.v1."
    compat_uses = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted(node)
        if dotted.startswith(prefix):
            compat_uses.add(dotted[len(prefix) :].split(".")[0])
    unexpected = compat_uses - ALLOWED_COMPAT
    assert not unexpected, f"{module.name} uses unexpected compat.v1 API: {sorted(unexpected)}"


def test_only_classifiers_needs_compat_v1():
    offenders = []
    for module in MODULES:
        tree = ast.parse(module.read_text())
        if any(
            isinstance(node, ast.Attribute) and "compat.v1" in _dotted(node)
            for node in ast.walk(tree)
        ):
            offenders.append(module.name)
    assert offenders == ["classifiers.py"], offenders
