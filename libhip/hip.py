import io
import os
import struct

from PIL import Image

HIP_PREFIX = b"HIP\x00"

RAW_A_SIZE = 1
RAW_RGB_SIZE = 3

HIP_PAL_IMG_CHUNK_SIZE = 2
HIP_STD_IMG_CHUNK_SIZE = 5

HIP_FILE_SIZE_INDEX = 1
HIP_NUM_COLORS_INDEX = 2

HIP_PAL_IMG_WIDTH_INDEX = 0
HIP_PAL_IMG_HEIGHT_INDEX = 1

HIP_STD_IMG_WIDTH_INDEX = 3
HIP_STD_IMG_HEIGHT_INDEX = 4


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
        raise ValueError("Not valid HIP file! Missing HIP file header prefix!")

    remaining = hip_contents[len(HIP_PREFIX):]
    header, remaining = _unpack_from("<IIIIIII", remaining)

    hip_file_size = header[HIP_FILE_SIZE_INDEX]
    if hip_file_size != len(hip_contents):
        raise ValueError("Not valid HIP file! File size mismatch!")

    # Grab important info our of the header.
    # Note that not all of it is currently used in this script but we grab it anyway.
    num_colors = header[HIP_NUM_COLORS_INDEX]

    # If a number of colors is called out then this HIP file represents a palette image.
    # The pixel data described in this file are palette indices.
    if num_colors:
        palette_header, remaining = _unpack_from("<IIIIIIII", remaining)
        width = palette_header[HIP_PAL_IMG_WIDTH_INDEX]
        height = palette_header[HIP_PAL_IMG_HEIGHT_INDEX]

    # Otherwise this HIP file describes raw RGBA pixel data and we have no palette.
    else:
        width = header[HIP_STD_IMG_WIDTH_INDEX]
        height = header[HIP_STD_IMG_HEIGHT_INDEX]

    return num_colors, width, height, remaining


def _parse_palette(num_colors, hip_contents):
    """
    Parse the palette data from the HIP file and create a raw palette that Pillow can work with.
    Separate the RGB and Alpha channels as we cannot create a palette image with an alpha channel
    in the palette data. The transparency information needs to be included in a tRNS header.
    """
    palette_size = num_colors * (RAW_RGB_SIZE + RAW_A_SIZE)

    palette_data = bytearray()
    alpha_data = bytearray()

    remaining = hip_contents[:palette_size]

    while remaining:
        rgb = remaining[:RAW_RGB_SIZE]
        remaining = remaining[RAW_RGB_SIZE:]

        alpha = remaining[:RAW_A_SIZE]
        remaining = remaining[RAW_A_SIZE:]

        palette_data += rgb
        alpha_data += alpha

    if len(palette_data) != len(alpha_data) * 3:
        raise ValueError("Mismatch between RGB and transparency data!")

    # Transpose the palette data so it works with PNG palette images.
    return palette_data[::-1], alpha_data[::-1], hip_contents[palette_size:]


def _parse_palette_image_data(width, height, num_colors, hip_contents):
    """
    Parse the palette image data of our HIP file.
    """
    total_pixels = width * height
    chunk_index = 0

    # The Pillow Image.putdata() method expects an iterable of pixel values.
    # Seemingly this can be any type of iterable of byte values, but we choose to construct a bytearray
    # featuring the data extracted from the HIP file as a bytearray is memory efficient.
    data = bytearray()

    while True:
        chunk_offset = chunk_index * HIP_PAL_IMG_CHUNK_SIZE

        # Read our color data in 2 byte chunks.
        # The data contains the color and number of pixels to render which are that color.
        color_data = hip_contents[chunk_offset:chunk_offset+HIP_PAL_IMG_CHUNK_SIZE]

        # If the color data chunk is empty then we have parsed the full image.
        if not color_data:
            break

        # Color palette index we are currently working with.
        # We subtract our "palette index" from the number of colors as we transposed the palette
        # as it exists in the HIP file so it is compatible with HPL files.
        # We need to subtract 1 from the index as a palette index ranges from 0 to num_colors-1.
        palette_index = num_colors - color_data[0] - 1
        # The number of pixels to draw using the given color.
        num_pixels = color_data[1]

        # Create  bytearray of length `num_pixels` where each element is the value `palette_index` and
        # append it to our total image data bytearray.
        data += bytearray((palette_index,) * num_pixels)

        chunk_index += 1

    if len(data) != total_pixels:
        raise ValueError("Image data length mismatch!")

    return data


def _parse_palette_image(num_colors, width, height, remaining, out):
    """
    Parse a HIP image that represents a PNG palette image.
    """
    palette, alpha, remaining = _parse_palette(num_colors, remaining)
    image_data = _parse_palette_image_data(width, height, num_colors, remaining)

    with Image.new("P", (width, height)) as image_fp:
        image_fp.putpalette(palette)
        image_fp.putdata(image_data)
        # Setting the transparency kwarg to our alpha raw data creates a tRNS header.
        # The kwarg expects an instance of `bytes()`.
        image_fp.save(out, format="PNG", transparency=bytes(alpha))


def _parse_standard_image_data(width, height, hip_contents):
    """
    Parse the raw RGBA image data from our HIP file.
    """
    total_pixels = width * height
    chunk_index = 0

    # We are parsing raw RGBA data. PIL expects RGBA images to provide
    # pixel data to the Image.putdata() method as RGBA tuples.
    data = []

    while True:
        chunk_offset = chunk_index * HIP_STD_IMG_CHUNK_SIZE

        # Read our color data in 5 byte chunks.
        # The data contains the color and number of pixels to render which are that color.
        color_data = hip_contents[chunk_offset:chunk_offset+HIP_STD_IMG_CHUNK_SIZE]

        # If the color data chunk is empty then we have parsed the full image.
        if not color_data:
            break

        # Note that HIP standard image files store there color data in the format BGRA.
        bgr = color_data[:RAW_RGB_SIZE]
        a = color_data[RAW_RGB_SIZE:RAW_RGB_SIZE+RAW_A_SIZE]
        # The last byte of the chunk is the number of pixels that is the given color.
        num_pixels = color_data[HIP_STD_IMG_CHUNK_SIZE-1]

        # Convert our color from BGRA to RGBA.
        rgba = tuple(bgr[::-1] + a)

        # Extend our data with the number of pixels that are the given color.
        data.extend([rgba] * num_pixels)

        chunk_index += 1

    if len(data) != total_pixels:
        raise ValueError("Image data length mismatch!")

    return data


def _parse_standard_image(width, height, hip_contents, out):
    """
    Parse a HIP image that represents a PNG RGBA image.
    """
    image_data = _parse_standard_image_data(width, height, hip_contents)

    with Image.new("RGBA", (width, height)) as image_fp:
        image_fp.putdata(image_data)
        image_fp.save(out, format="PNG")


def hip_to_png(hip_image, out=None):
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

    elif isinstance(hip_image, str) and not os.path.exists(hip_image):
        raise ValueError(f"HIP image {hip_image} does not exist!")

    elif isinstance(hip_image, io.BytesIO):
        hip_contents = hip_image.read()

    else:
        raise TypeError(f"Unsupported HIP image type {hip_image}!")

    num_colors, width, height, remaining = _parse_header(hip_contents)

    if num_colors > 0:
        _parse_palette_image(num_colors, width, height, remaining, out)

    else:
        _parse_standard_image(width, height, remaining, out)
