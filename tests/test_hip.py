import os
import unittest
import contextlib

from libhip.hip import HIPImage

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
SRC_PAL_PNG = os.path.join(TEST_DIRECTORY, "src_pal.png")

REF_PAL_HIP = os.path.join(TEST_DIRECTORY, "ref_pal.hip")
REF_PAL_HIP_DATA = read_file(REF_PAL_HIP)
REF_PAL_PNG = os.path.join(TEST_DIRECTORY, "ref_pal.png")
REF_PAL_PNG_DATA = read_file(REF_PAL_PNG)

SRC_RAW_HIP = os.path.join(TEST_DIRECTORY, "src_raw.hip")
SRC_RAW_PNG = os.path.join(TEST_DIRECTORY, "src_raw.png")

REF_RAW_HIP = os.path.join(TEST_DIRECTORY, "ref_raw.hip")
REF_RAW_HIP_DATA = read_file(REF_RAW_HIP)
REF_RAW_PNG = os.path.join(TEST_DIRECTORY, "ref_raw.png")
REF_RAW_PNG_DATA = read_file(REF_RAW_PNG)


class HIPImageTests(unittest.TestCase):
    def test_hip_to_png_palette(self):
        with test_file("hip_to_png_pal.png") as hip_to_png_pal:
            image = HIPImage()
            image.load_hip(SRC_PAL_HIP)
            image.save_png(hip_to_png_pal)
            hip_to_png_pal_data = read_file(hip_to_png_pal)
            self.assertEqual(hip_to_png_pal_data, REF_PAL_PNG_DATA)

    def test_hip_to_png_raw(self):
        with test_file("hip_to_png_raw.png") as hip_to_png_raw:
            image = HIPImage()
            image.load_hip(SRC_RAW_HIP)
            image.save_png(hip_to_png_raw)
            hip_to_png_raw_data = read_file(hip_to_png_raw)
            self.assertEqual(hip_to_png_raw_data, REF_RAW_PNG_DATA)

    def test_png_palette_to_hip(self):
        with test_file("png_pal_to_hip.png") as png_palette_to_hip:
            image = HIPImage()
            image.load_png(SRC_PAL_PNG)
            image.save_hip(png_palette_to_hip)
            png_pal_to_hip_data = read_file(png_palette_to_hip)
            self.assertEqual(png_pal_to_hip_data, REF_PAL_HIP_DATA)

    def test_png_raw_to_hip(self):
        with test_file("png_raw_to_hip.png") as png_raw_to_hip:
            image = HIPImage()
            image.load_png(SRC_RAW_PNG)
            image.save_hip(png_raw_to_hip)
            png_raw_to_hip_data = read_file(png_raw_to_hip)
            self.assertEqual(png_raw_to_hip_data, REF_RAW_HIP_DATA)
