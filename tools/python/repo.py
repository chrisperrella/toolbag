import os
import subprocess
import sys
import tempfile


def install_requirements() -> bool:
    lines = []
    requirements = {
        "colorama": ("0.4.6", "import colorama"),
        "toml": ("0.10.2", "import toml"),
        "isort": ("5.13.2", "import isort"),
        "black": ("24.10.0", "import black"),
    }
    for name in requirements:
        version, import_line = requirements[name]
        try:
            exec(import_line)
        except ImportError:
            lines.append(f"{name}=={version}")
    if len(lines) > 0:
        with tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".txt") as file:
            file.write("\n".join(lines) + "\n")
            file.close()
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", file.name])
            if result.returncode != 0:
                print("Failed to install requirements!")
                return False
    return True


if __name__ == "__main__":
    if not install_requirements():
        sys.exit(1)

    from global_constants import global_constants
    from utils import strip_suffix

    repo_commands = global_constants.lib_path / "repo_commands"
    command_args = args = sys.argv[2:]
    command_name = sys.argv[1] if len(sys.argv) > 1 else ""
    command_path = repo_commands / f"{command_name}.py"
    if command_name == "help":
        commands = [strip_suffix(e) for e in os.listdir(repo_commands) if e.lower().endswith(".py")]
        command_string = "    " + "\n    ".join(commands)
        print(f"Available commands:\n{command_string}")
        sys.exit(0)
    if not command_path.exists() or not command_path.is_file():
        print(repo_commands)
        commands = [strip_suffix(e) for e in os.listdir(repo_commands) if e.lower().endswith(".py")]
        command_string = "    " + "\n    ".join(commands)
        print(f'Command "{command_name}" is not a valid command!')
        print(f"Valid commands are:\n{command_string}")
        sys.exit(1)
    returncode = 0
    try:
        exec(f"from repo_commands.{command_name} import run")
        exec("returncode = run(command_args)")
    except Exception:
        print(
            f'Failed to run command "{command_name}"!',
            exc_info=True,
        )
        returncode = 1
    sys.exit(returncode)
