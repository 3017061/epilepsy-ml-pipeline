"""Smoke-test the project environment.

Checks:
- required packages import
- `src.data.loader` imports and exposes `load_epileptic_seizure`
- attempts a non-network local load of the epileptic dataset (expected to fail if data is absent)

Exit codes:
- 0: success
- 2: import/module error
- 3: unexpected runtime error
"""

from pathlib import Path
import sys
import importlib

print('Working directory:', Path.cwd())

def print_version(pkg, attr='__version__'):
    try:
        p = importlib.import_module(pkg)
        v = getattr(p, attr, None)
        print(f"{pkg}:", v)
    except Exception as e:
        print(f"{pkg}: import failed ({e})")

# Print versions for key deps
print_version('numpy')
print_version('pandas')
print_version('sklearn', attr='__version__')
print_version('openml')

# Ensure repo root is importable
sys.path.insert(0, str(Path.cwd()))

try:
    loader = importlib.import_module('src.data.loader')
    print('Imported loader:', getattr(loader, '__file__', '<unknown>'))
except Exception as e:
    print('Failed to import src.data.loader:', e)
    sys.exit(2)

if not hasattr(loader, 'load_epileptic_seizure'):
    print('Missing function: load_epileptic_seizure')
    sys.exit(2)

print('Found load_epileptic_seizure(). Calling with openml_lookup=False (no network).')
try:
    df, meta = loader.load_epileptic_seizure(openml_lookup=False)
    print('Local CSV load succeeded. Data shape:', getattr(df, 'shape', None))
except FileNotFoundError:
    print('Local CSV not found (this is OK if you did not download the dataset).')
except Exception as e:
    print('Loader raised unexpected error:', repr(e))
    sys.exit(3)

print('Smoke test passed')
sys.exit(0)
