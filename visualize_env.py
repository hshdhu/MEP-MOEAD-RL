
import argparse
from pathlib import Path

from general.environment import Environment
from utils.draw import plot_environment, plot_environment_image


def visualize_environment(json_path: Path, save_path: Path | None = None) -> None:
    """
    Load environment data from a JSON file and visualize it.

    If save_path is provided, the environment plot will be saved to that file.
    Otherwise, the plot will be displayed interactively.
    """
    env, path = Environment.load(str(json_path))

    if path:
        plot_environment(env)
    else:
        plot_environment_image(env, str(save_path) if save_path else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an environment JSON file.")
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the environment JSON file.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the rendered image instead of displaying it.",
    )
    args = parser.parse_args()

    visualize_environment(args.json_path, args.save)


if __name__ == "__main__":
    main()

