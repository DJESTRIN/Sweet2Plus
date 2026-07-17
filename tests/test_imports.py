"""
Smoke-test suite: verifies that every importable module in the Sweet2Plus
package can be imported without raising an exception.

This does not exercise runtime logic (there is no synthetic test data in
this repo), but it catches an entire class of real bugs cheaply and
quickly: syntax errors, missing/incorrect imports, undefined names used at
module scope, and broken package metadata (see the accompanying
`Sweet2Plus/__init__.py` files, which were previously missing entirely).

A small allow-list of modules is skipped when their dependency genuinely
cannot be installed from PyPI (see SKIP_MODULES below for specifics).
"""
import importlib
import pkgutil

import pytest

import Sweet2Plus

# Modules skipped with a reason. Keep this list as small as possible --
# if a dependency becomes available, or a script is refactored to guard its
# side effects behind `if __name__ == "__main__":`, remove it from here.
SKIP_MODULES = {
    "Sweet2Plus.denoise.RunDeepCAD": (
        "requires the 'deepcad' PyPI package, but that package is an unrelated "
        "namespace squat with no 'deepcad' importable module; the actual "
        "DeepCAD-RT project this code depends on is only distributed on GitHub"
    ),
}


def _discover_modules():
    modules = []
    for _, name, _ in pkgutil.walk_packages(Sweet2Plus.__path__, prefix="Sweet2Plus."):
        modules.append(name)
    return sorted(modules)


@pytest.mark.parametrize("module_name", _discover_modules())
def test_module_imports_cleanly(module_name):
    if module_name in SKIP_MODULES:
        pytest.skip(SKIP_MODULES[module_name])
    importlib.import_module(module_name)
