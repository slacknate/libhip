import os
import argparse

from .hip import convert_from_hip


def abs_path(value):
    value = os.path.abspath(value)

    if not os.path.exists(value):
        raise argparse.ArgumentError("Invalid file path! Does not exist!")

    return value


def main():
    parser = argparse.ArgumentParser("hip")
    subparsers = parser.add_subparsers(title="commands")

    topng = subparsers.add_parser("topng")
    topng.add_argument(dest="hip_path", type=abs_path, help="HIP file input path.")

    args, _ = parser.parse_known_args()

    hip_path = getattr(args, "hip_path", None)

    if hip_path is not None:
        convert_from_hip(hip_path)


if __name__ == "__main__":
    main()
