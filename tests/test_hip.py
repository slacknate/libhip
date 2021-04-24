import os
import unittest
import contextlib

from libhip.hip import hip_to_png

TEST_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


@contextlib.contextmanager
def test_file(file_name):
    file_path = os.path.join(TEST_DIRECTORY, file_name)

    try:
        yield file_path

    finally:
        os.remove(file_path)


def read_file(file_name):
    with open(file_name, "rb") as ref_fp:
        return ref_fp.read()


SRC_PAL_HIP = os.path.join(TEST_DIRECTORY, "src_pal.hip")

REF_PAL_PNG = os.path.join(TEST_DIRECTORY, "ref_pal.png")
REF_PAL_PNG_DATA = read_file(REF_PAL_PNG)

SRC_RAW_HIP = os.path.join(TEST_DIRECTORY, "src_raw.hip")

REF_RAW_PNG = os.path.join(TEST_DIRECTORY, "ref_raw.png")
REF_RAW_PNG_DATA = read_file(REF_RAW_PNG)


class HPLPaletteTests(unittest.TestCase):
    def test_hip_to_png_palette(self):
        with test_file("hip_to_png_pal.png") as hip_to_png_pal:
            hip_to_png(SRC_PAL_HIP, out=hip_to_png_pal)
            hip_to_png_pal_data = read_file(hip_to_png_pal)
            self.assertEqual(hip_to_png_pal_data, REF_PAL_PNG_DATA)

    def test_hip_to_png_raw(self):
        with test_file("hip_to_png_raw.png") as hip_to_png_raw:
            hip_to_png(SRC_RAW_HIP, out=hip_to_png_raw)
            hip_to_png_raw_data = read_file(hip_to_png_raw)
            self.assertEqual(hip_to_png_raw_data, REF_RAW_PNG_DATA)
