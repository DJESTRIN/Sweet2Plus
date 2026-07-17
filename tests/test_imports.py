"""
Smoke-test suite: verifies that every importable module in the Sweet2Plus
package can be imported without raising an exception.

This does not exercise runtime logic (there is no synthetic test data in
this repo), but it catches an entire class of real bugs cheaply and
quickly: syntax errors, missing/incorrect imports, undefined names used at
module scope, and broken package metadata (see the accompanying
`Sweet2Plus/__init__.py` files, which were previously missing entirely).

A small allow-list of modules is skipped because they either:
  * require optional/heavy dependencies not in requirements.txt
    (torch-geometric, keras, customtkinter, projectmanager), or
  * execute real analysis/plotting logic at module scope instead of behind
    an `if __name__ == "__main__":` guard, so importing them has side
    effects (file I/O, plotting) rather than just defining symbols.
"""
import importlib
import pkgutil

import pytest

import Sweet2Plus

# Modules skipped with a reason. Keep this list as small as possible --
# if a dependency becomes available, or a script is refactored to guard its
# side effects behind `if __name__ == "__main__":`, remove it from here.
SKIP_MODULES = {
    "Sweet2Plus.core.quickgui": "GUI module requiring customtkinter/tkinter",
    "Sweet2Plus.core.StimulationAnalysis": "imports Sweet2Plus.core.quickgui, which requires customtkinter",
    "Sweet2Plus.signalclassifier.mlp_decoder": "requires keras/tensorflow (not in requirements.txt)",
    "Sweet2Plus.graphics.zprojection_grid": "executes analysis/plotting code at import time (not guarded by __main__)",
    "Sweet2Plus.decoders.NetworkArchitectures": "requires torch-geometric (not in requirements.txt)",
    "Sweet2Plus.decoders.GraphNeuralNetwork": "requires torch-geometric (not in requirements.txt)",
    "Sweet2Plus.decoders.CoderDecoders": "imports Sweet2Plus.decoders.NetworkArchitectures, which requires torch-geometric",
    "Sweet2Plus.denoise.RunDeepCAD": "requires the deepcad package (not published on PyPI)",
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
