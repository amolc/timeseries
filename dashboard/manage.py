#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'timeseries_dashboard.settings')

    # Ensure dashboard modules resolve before project-root modules with same name.
    current_dir = str(Path(__file__).resolve().parent)
    project_root = str(Path(current_dir).parent)
    for p in (current_dir, project_root):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, project_root)
    sys.path.insert(0, current_dir)

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
