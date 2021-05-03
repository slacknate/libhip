import os
import argparse

from .hip import hip_to_png, png_to_hip


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

    topng = subparsers.add_parser("frompng")
    topng.add_argument(dest="png_path", type=abs_path, help="PNG file input path.")

    args, _ = parser.parse_known_args()

    hip_path = getattr(args, "hip_path", None)
    png_path = getattr(args, "png_path", None)

    if hip_path is not None:
        hip_to_png(hip_path)

    if png_path is not None:
        png_to_hip(png_path)


if __name__ == "__main__":
    main()
