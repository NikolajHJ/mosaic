"""Load only the wrapper code, using the teacher interpreter's dependencies.

Adding the coordinator's site-packages to PYTHONPATH can expose incompatible
NumPy/Torch extension modules when coordinator and worker use different Python
versions. Load this package explicitly; leave dependency resolution to the
teacher environment. The caller's interpreter and sys.path are unchanged.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def main():
    directory = Path(__file__).resolve().parent
    spec = spec_from_file_location(
        "mosaic_defog", directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    package = module_from_spec(spec)
    sys.modules["mosaic_defog"] = package
    spec.loader.exec_module(package)
    from mosaic_defog._worker import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
