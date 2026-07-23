"""Make the project's ``src/`` importable from the ``plotting/`` directory.

The plotting/analysis scripts were migrated out of ``src/`` but still import
project modules (``definitions``, ``models``, ``envs``, ``algos``) using
top-level absolute imports.  Importing this module first inserts the sibling
``src/`` directory onto ``sys.path`` so those imports keep working when a script
is run from ``plotting/`` (e.g. ``python plotting/plot_radar.py``).
"""
import os
import sys

_SRC = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
