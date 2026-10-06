"""`python -m tuningfork`: the CLI, for a client config that names a Python, not a script."""

from tuningfork.cli import app

if __name__ == "__main__":
    app(prog_name="tuningfork")
