import io
import os
import struct

from PIL import Image

HIP_PREFIX = b"HIP\x00"

HIP_PAL_CHUNK_SIZE = 3
HIP_IMG_CHUNK_SIZE = 2

HIP_NUM_COLORS_INDEX = 2
HIP_COLOR_DEPTH_INDEX = 6
HIP_IMG_WIDTH_INDEX = 7
HIP_IMG_HEIGHT_INDEX = 8


def _unpack_from(fmt, data):
    """
    Helper function to call and return the result of struct.unpack_from
    as well as any remaining packed data that exists in our bytestring following what was unpacked.
    """
    offset = struct.calcsize(fmt)
    unpacked = struct.unpack_from(fmt, data)
    remaining = data[offset:]
    return unpacked, remaining


def _parse_header(hip_contents):
    """
    Parse the header of a HIP file.
    We do basic validation of the header with the HIP_PREFIX constant.
    """
    if not hip_contents.startswith(HIP_PREFIX):
        raise ValueError("Not valid HIP file!")

    remaining = hip_contents[len(HIP_PREFIX):]
    data, remaining = _unpack_from("<IIIIIIIIIIIIIII", remaining)

    # Grab important info our of the header.
    # Note that not all of it is currently used in this script but we grab it anyway.
    num_colors = data[HIP_NUM_COLORS_INDEX]
    palette_size = num_colors * 4
    color_depth = data[HIP_COLOR_DEPTH_INDEX] / 4
    width = data[HIP_IMG_WIDTH_INDEX]
    height = data[HIP_IMG_HEIGHT_INDEX]

    if not color_depth.is_integer():
        raise ValueError(f"Color depth of {color_depth} is invalid!")

    color_depth = int(color_depth)
    return num_colors, color_depth, palette_size, width, height, remaining


def _parse_palette(hip_contents, palette_size):
    """
    Parse the palette data from the HIP file and create a raw palette that Pillow can work with.
    """
    palette = b""

    palette_data = hip_contents[:palette_size]
    remaining = hip_contents[palette_size:]

    while palette_data:
        # Read the palette data in chunks of four bytes.
        # Note that we drop the last byte as it is probably an Alpha channel that we don't care about.
        color = palette_data[0:HIP_PAL_CHUNK_SIZE]
        palette_data = palette_data[HIP_PAL_CHUNK_SIZE + 1:]
        palette += color

    # Transpose the palette so it works with HPL files.
    palette = palette[::-1]
    return palette, remaining


def _parse_image_data(hip_contents, num_colors, image_fp):
    """
    Parse the image data of our HIP file and write the data into our PNG palette image.
    """
    remaining = hip_contents

    # The Pillow Image.putdata() method expects an iterable of pixel values.
    # Seemingly this can be any type of iterable of byte values, but we choose to construct a bytearray
    # featuring the data extracted from the HIP file as a bytearray is memory efficient.
    data = bytearray()

    while remaining:
        # Color palette index we are currently working with.
        # We subtract our "palette index" from the number of colors as we transposed the palette
        # as it exists in the HIP file so it is compatible with HPL files.
        # We need to subtract 1 from the index as a palette index ranges from 0 to num_colors-1.
        palette_index = num_colors - remaining[0] - 1
        # The number of pixels to draw using the given color.
        num_pixels = remaining[1]

        # Create  bytearray of length `num_pixels` where each element is the value `palette_index` and
        # append it to our total image data bytearray.
        data += bytearray((palette_index,) * num_pixels)

        remaining = remaining[HIP_IMG_CHUNK_SIZE:]

    image_fp.putdata(data)


def convert_from_hip(hip_image, out=None):
    """
    Extract an image from a HIP file and save it as a PNG palette image.

    Reference: https://github.com/dantarion/bbtools/blob/master/extractHip.py
    """
    if out is None:
        if not isinstance(hip_image, str):
            raise ValueError("Must provide an output path or fp when not supplying HIP image via file path!")

        out = hip_image.replace(".hip", ".png")

    if isinstance(hip_image, str) and os.path.exists(hip_image):
        with open(hip_image, "rb") as hip_fp:
            hip_contents = hip_fp.read()

    elif isinstance(hip_image, io.BytesIO):
        hip_contents = hip_image.read()

    else:
        raise TypeError(f"Unsupported HIP image type {hip_image}!")

    num_colors, _, palette_size, width, height, remaining = _parse_header(hip_contents)
    palette, remaining = _parse_palette(remaining, palette_size)

    with Image.new("P", (width, height)) as image_fp:
        image_fp.putpalette(palette)
        _parse_image_data(remaining, num_colors, image_fp)
        image_fp.save(out, format="PNG")
