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


def _parse_image_data(hip_contents, num_colors, width, height, image_fp):
    """
    Parse the image data of our HIP file and write the data into our PNG palette image.
    """
    remaining = hip_contents

    # Pixel coordinates.
    x = 0
    y = 0

    while remaining:
        # Color palette index we are currently working with.
        # We subtract our "palette index" from the number of colors as we transposed the palette
        # as it exists in the HIP file so it is compatible with HPL files.
        # We need to subtract 1 from the index as a palette index ranges from 0 to num_colors-1.
        palette_index = num_colors - remaining[0] - 1
        # The number of pixels to draw using the given color.
        num_pixels = remaining[1]

        # Draw our pixels in our PNG image one at a time. Painstaking, yes, but it works.
        for _ in range(num_pixels):
            # Ensure we are not attempting to create an invalid image.
            # We do not check against the width here as we use modular math to limit the x values.
            if y >= height:
                raise ValueError("Cannot exceed image height!")

            # We do not set the pixel value to a color, but rather to the color palette index.
            # The color will be set correctly from the palette.
            image_fp.putpixel((x, y), palette_index)

            # Move to the next column in the row. Once we reach the end of the row we head back to column 0.
            x += 1
            x %= width
            # Once we have filled a row of length `width` we move to the next row.
            if x == 0:
                y += 1

        remaining = remaining[HIP_IMG_CHUNK_SIZE:]


def extract_img(hip_path):
    """
    Extract an image from a HIP file and save it as a PNG palette image.

    Reference: https://github.com/dantarion/bbtools/blob/master/extractHip.py
    """
    out = hip_path.replace(".hip", ".png")

    with open(hip_path, "rb") as hip_fp:
        hip_contents = hip_fp.read()

    num_colors, _, palette_size, width, height, remaining = _parse_header(hip_contents)
    palette, remaining = _parse_palette(remaining, palette_size)

    with Image.new("P", (width, height)) as image_fp:
        image_fp.putpalette(palette)
        _parse_image_data(remaining, num_colors, width, height, image_fp)
        image_fp.save(out)
