"""Comprehensive test runner for NeuroStream test suite.

Discovers and executes all tests in tests/ with fixture resolution.
Usage:
    python scripts/run_tests.py
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

os.environ["NEUROSTREAM_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = str(repo_root / ".cache" / "matplotlib")
os.environ["MNE_DATA"] = str(repo_root / "data" / "mne_data")


def run_all_tests() -> bool:
    test_dir = repo_root / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))

    total = 0
    passed = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    start_time = time.time()
    print(f"=== NeuroStream Test Suite ({len(test_files)} test files) ===\n")

    for tf in test_files:
        mod_name = tf.stem
        # Import module
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(mod_name, tf)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[ERROR] Could not import {tf.name}: {e}")
            errors.append((tf.name, traceback.format_exc()))
            continue

        # Collect fixtures in module
        fixtures = {}
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if callable(obj) and getattr(obj, "_is_fixture", False):
                fixtures[attr_name] = obj

        # Collect test functions and classes
        test_items = []
        for attr_name in dir(mod):
            if attr_name.startswith("test_"):
                obj = getattr(mod, attr_name)
                if callable(obj):
                    test_items.append((attr_name, obj, None))
            elif attr_name.startswith("Test"):
                cls = getattr(mod, attr_name)
                if isinstance(cls, type):
                    instance = cls()
                    for m_name in dir(instance):
                        if m_name.startswith("test_"):
                            m = getattr(instance, m_name)
                            if callable(m):
                                test_items.append((f"{attr_name}.{m_name}", m, instance))

        for name, fn, inst in test_items:
            total += 1
            sig = inspect.signature(fn)
            kwargs = {}
            temp_dirs = []

            try:
                for param in sig.parameters.values():
                    p_name = param.name
                    if p_name == "self":
                        continue
                    elif p_name == "tmp_path":
                        td = Path(tempfile.mkdtemp())
                        temp_dirs.append(td)
                        kwargs[p_name] = td
                    elif p_name in fixtures:
                        f_fn = fixtures[p_name]
                        # Check if fixture itself has dependencies
                        f_sig = inspect.signature(f_fn)
                        f_kwargs = {}
                        for fp in f_sig.parameters.values():
                            if fp.name in fixtures:
                                f_kwargs[fp.name] = fixtures[fp.name]()
                            elif fp.name == "tmp_path":
                                td = Path(tempfile.mkdtemp())
                                temp_dirs.append(td)
                                f_kwargs["tmp_path"] = td
                        kwargs[p_name] = f_fn(**f_kwargs)
                    else:
                        pass

                # Execute test
                fn(**kwargs)
                passed += 1
                print(f"  ✓ {tf.name}::{name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {tf.name}::{name} -> FAILED: {e}")
                errors.append((f"{tf.name}::{name}", traceback.format_exc()))
            finally:
                for td in temp_dirs:
                    shutil.rmtree(td, ignore_errors=True)

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"Results: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print(f"=======================================================\n")

    if errors:
        print("Failures:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
        return False
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

