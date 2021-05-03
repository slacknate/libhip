import io
import os
import struct
import contextlib

from PIL import Image

HIP_PREFIX = b"HIP\x00"

PALETTE_IMAGE_TYPE = "P"
RAW_IMAGE_TYPE = "RGBA"

RAW_A_SIZE = 1
RAW_RGB_SIZE = 3
BITS_PER_COLOR_CHANNEL = 8

HIP_MAX_COLORS = 256
MAX_BYTE_VALUE = 255

HIP_PAL_IMG_CHUNK_SIZE = 2
HIP_RAW_IMG_CHUNK_SIZE = 5

HIP_FILE_SIZE_INDEX = 1
HIP_NUM_COLORS_INDEX = 2

HIP_PAL_IMG_WIDTH_INDEX = 0
HIP_PAL_IMG_HEIGHT_INDEX = 1

HIP_RAW_IMG_WIDTH_INDEX = 3
HIP_RAW_IMG_HEIGHT_INDEX = 4

INT_SIZE = struct.calcsize("<I")

HEADER_SIZE = INT_SIZE * 7
PALETTE_HEADER_SIZE = INT_SIZE * 8

# TODO: we probably want to support:
#       24 bit color (i.e. RGB with no transparency)
#       palette images with less than 256 colors


@contextlib.contextmanager
def output_image(hip_output):
    """
    Helper context manager that either wraps `open()` or simply yields an `io.BytesIO`.
    Also provides basic type validation.
    """
    if isinstance(hip_output, str):
        with open(hip_output, "wb") as hpl_fp:
            yield hpl_fp

    elif isinstance(hip_output, io.BytesIO):
        yield hip_output

    else:
        raise TypeError(f"Unsupported output image type {hip_output}!")


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
        if num_colors < HIP_MAX_COLORS:
            raise ValueError(f"Image contains {num_colors} colors which is less than {HIP_MAX_COLORS}!")

        palette_header, remaining = _unpack_from("<IIIIIIII", remaining)
        width = palette_header[HIP_PAL_IMG_WIDTH_INDEX]
        height = palette_header[HIP_PAL_IMG_HEIGHT_INDEX]

    # Otherwise this HIP file describes raw RGBA pixel data and we have no palette.
    else:
        width = header[HIP_RAW_IMG_WIDTH_INDEX]
        height = header[HIP_RAW_IMG_HEIGHT_INDEX]

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
        bgr = remaining[:RAW_RGB_SIZE]
        remaining = remaining[RAW_RGB_SIZE:]

        alpha = remaining[:RAW_A_SIZE]
        remaining = remaining[RAW_A_SIZE:]

        palette_data += bgr
        alpha_data += alpha

    if len(palette_data) != len(alpha_data) * (RAW_RGB_SIZE / RAW_A_SIZE):
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


def _parse_raw_image_data(width, height, hip_contents):
    """
    Parse the raw RGBA image data from our HIP file.
    """
    total_pixels = width * height
    chunk_index = 0

    # We are parsing raw RGBA data. PIL expects RGBA images to provide
    # pixel data to the Image.putdata() method as RGBA tuples.
    data = []

    while True:
        chunk_offset = chunk_index * HIP_RAW_IMG_CHUNK_SIZE

        # Read our color data in 5 byte chunks.
        # The data contains the color and number of pixels to render which are that color.
        color_data = hip_contents[chunk_offset:chunk_offset+HIP_RAW_IMG_CHUNK_SIZE]

        # If the color data chunk is empty then we have parsed the full image.
        if not color_data:
            break

        # Note that HIP raw image files store there color data in the format BGRA.
        bgr = color_data[:RAW_RGB_SIZE]
        a = color_data[RAW_RGB_SIZE:RAW_RGB_SIZE+RAW_A_SIZE]
        # The last byte of the chunk is the number of pixels that is the given color.
        num_pixels = color_data[HIP_RAW_IMG_CHUNK_SIZE-1]

        # Convert our color from BGRA to RGBA.
        rgba = tuple(bgr[::-1] + a)

        # Extend our data with the number of pixels that are the given color.
        data.extend([rgba] * num_pixels)

        chunk_index += 1

    if len(data) != total_pixels:
        raise ValueError("Image data length mismatch!")

    return data


def _parse_raw_image(width, height, hip_contents, out):
    """
    Parse a HIP image that represents a PNG RGBA image.
    """
    image_data = _parse_raw_image_data(width, height, hip_contents)

    with Image.new("RGBA", (width, height)) as image_fp:
        image_fp.putdata(image_data)
        image_fp.save(out, format="PNG")


def hip_to_png(hip_image, out=None):
    """
    Extract an image from a HIP file and save it as a PNG image.
    The resultant image may be a palette image or a raw RGBA image as HIP images could be either.

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
        _parse_raw_image(width, height, remaining, out)


def _build_header(image_dimensions, image_type, image_data_size, palette_data_size=0):
    """
    Build a valid HIP file header.
    """
    palette_header_size = PALETTE_HEADER_SIZE if image_type == PALETTE_IMAGE_TYPE else 0
    file_size = len(HIP_PREFIX) + HEADER_SIZE + palette_header_size + palette_data_size + image_data_size

    # Not sure what this value is but it seems static.
    unknown_00 = 0x125

    if image_type == PALETTE_IMAGE_TYPE:
        # Not sure what this value is but it seems static for palette images.
        unknown_05 = 0x0
        num_colors = HIP_MAX_COLORS
        # Not sure why but palette images seem to have the image dimensions stored in a sub-header of some kind.
        width1 = 0
        height1 = 0

    else:
        # Not sure what this value is but it seems static for raw images.
        unknown_05 = 0x110
        # Raw images do not feature a palette, and thus do not set the number of colors.
        num_colors = 0
        width1, height1 = image_dimensions

    color_depth = num_colors // BITS_PER_COLOR_CHANNEL

    # Fill in our required prefix and meta data.
    header = HIP_PREFIX
    header += struct.pack("<IIIIIII", unknown_00, file_size, num_colors, width1, height1, unknown_05, color_depth)

    if image_type == PALETTE_IMAGE_TYPE:
        # The aforementioned palette image sub-header that contains the image dimensions.
        width2, height2 = image_dimensions
        # TODO: We currently pack a bunch of zeros in here but it seems like this should be real data?
        #       Not even sure it matters but felt a note should be made here in case it does.
        header += struct.pack("<IIIIIIII", width2, height2, 0, 0, 0, 0, 0, 0)

    return header


def _build_palette(palette_data, alpha):
    """
    Build a HIP image palette.
    The data is structured in the format: BGRA.
    """
    pixel_index = 0
    hip_palette_data = bytearray()
    palette_data = palette_data[::-1]
    alpha = alpha[::-1]

    if len(palette_data) != len(alpha) * (RAW_RGB_SIZE / RAW_A_SIZE):
        raise ValueError("Mismatch between RGB and transparency data!")

    while True:
        bgr_offset = pixel_index * RAW_RGB_SIZE
        a_offset = pixel_index * RAW_A_SIZE

        bgr = palette_data[bgr_offset:bgr_offset+RAW_RGB_SIZE]
        a = alpha[a_offset:a_offset+RAW_A_SIZE]

        # If there is no remaining palette data from the source then we are done.
        if not bgr and not a:
            break

        # Check to ensure we have not de-synced during the parse process.
        if bool(bgr) != bool(a):
            raise ValueError("Mismatch between RGB and transparency data!")

        hip_palette_data += bgr
        hip_palette_data += a

        pixel_index += 1

    if len(hip_palette_data) != HIP_MAX_COLORS * (RAW_RGB_SIZE + RAW_A_SIZE):
        raise ValueError("Mismatch between RGB and transparency data!")

    return hip_palette_data


def _build_palette_image(image_data):
    """
    Build HIP palette image data.
    The data is structed as two byte chunks which are: Palette Index, Pixel Count
    """
    count = 0
    current_index = -1
    last_index = HIP_MAX_COLORS - 1
    hip_image_data = bytearray()

    for palette_index in image_data:
        if (palette_index != current_index and count > 0) or count >= MAX_BYTE_VALUE:
            hip_image_data += bytearray((last_index - current_index, count))
            count = 0

        current_index = palette_index
        count += 1

    if count:
        hip_image_data += bytearray((last_index - current_index, count))

    return hip_image_data


def _save_palette_image(image_fp, out):
    """
    Build a HIP palette image and write it out to the given destination.
    """
    # Get image size.
    size = image_fp.size
    # Get image data.
    image = bytearray(image_fp.getdata())
    # Get color information from the palette.
    palette_data = bytearray(image_fp.getdata().getpalette())
    # Get transparency information from the tRNS header.
    alpha = bytearray(image_fp.info["transparency"])

    hip_palette_data = _build_palette(palette_data, alpha)
    hip_image_data = _build_palette_image(image)
    header = _build_header(size, PALETTE_IMAGE_TYPE, len(hip_image_data), palette_data_size=len(hip_palette_data))

    with output_image(out) as out_fp:
        out_fp.write(header)
        out_fp.write(hip_palette_data)
        out_fp.write(hip_image_data)


def _build_raw_image(image_data):
    """
    Build HIP palette image data.
    The data is structed as five byte chunks which are: BGRA (color), Pixel Count
    """
    count = 0
    pixel_index = 0
    current_color = b""
    hip_image_data = bytearray()

    # Pillow presents the image data of a raw RGBA PNG image as color tuples.
    for color_data in image_data:
        bgr = color_data[:RAW_RGB_SIZE][::-1]
        a = color_data[RAW_RGB_SIZE:RAW_RGB_SIZE+RAW_A_SIZE]
        color = bgr + a

        if (color != current_color and count > 0) or count >= MAX_BYTE_VALUE:
            hip_image_data += bytearray(current_color) + bytearray((count,))
            count = 0

        current_color = color
        pixel_index += 1
        count += 1

    if count:
        hip_image_data += bytearray(current_color) + bytearray((count,))

    return hip_image_data


def _save_raw_image(image_fp, out):
    """
    Build a HIP raw image and write it out to the given destination.
    """
    # Get image size.
    size = image_fp.size
    # Get image data.
    image = image_fp.getdata()

    hip_image_data = _build_raw_image(image)
    header = _build_header(size, RAW_IMAGE_TYPE, len(hip_image_data))

    with output_image(out) as out_fp:
        out_fp.write(header)
        out_fp.write(hip_image_data)


def png_to_hip(png_image, out=None):
    """
    Convert a PNG image to an equivalent HIP image.
    Can take a palette image or a raw RGBA image as input.
    """
    if out is None:
        if not isinstance(png_image, str):
            raise ValueError("Must provide an output path or fp when not supplying PNG image via file path!")

        out = png_image.replace(".png", ".hip")

    with Image.open(png_image) as image_fp:
        image_type = image_fp.mode

        if image_type == PALETTE_IMAGE_TYPE:
            _save_palette_image(image_fp, out)

        elif image_type == RAW_IMAGE_TYPE:
            _save_raw_image(image_fp, out)

        else:
            raise ValueError(f"Unknown image type {image_type}!")
