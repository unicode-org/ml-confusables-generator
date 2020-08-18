# Copyright (C) 2020 and later: Google, Inc.

import os
from qahirah import CAIRO, Colour, Vector
import shutil
import time
import unittest
from unittest.mock import MagicMock, patch, call
from vis_gen import VisualGenerator

class TestVisualGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test data
        cls.tmp_dir = '.tmp' + str(time.time())
        cls.src_file = os.path.join(cls.tmp_dir, 'src.txt')
        cls.code_point_list = ['U+0044', 'U+0065', 'U+006E', 'U+0069',
                                   'U+0073']
        cls.char_list = ['D', 'e', 'n', 'i', 's']

        # Build temporary testing directory
        print("Building temporary directory {}.".format(cls.tmp_dir))
        os.mkdir(cls.tmp_dir)

        #
        print("Building temporary source file {}.".format(cls.src_file))
        with open(cls.src_file, 'w+') as f:
            for code_point in cls.code_point_list:
                f.write(code_point)
                f.write('\n')

    @classmethod
    def tearDownClass(cls):
        print("Deleting temporary directory and file for testing.")
        shutil.rmtree(cls.tmp_dir)

    def test_default_init(self):
        """Test default initialization. When default initialization value
        changes, or any private attribute does not match public attribute, this
        test will fail."""
        vg = VisualGenerator()

        # vg.font_size
        self.assertEqual(vg._font_size, 16)
        # vg.image_size
        self.assertEqual(vg._image_size, 20)
        # vg.font_name)
        self.assertEqual(vg._font_name, "Noto Sans CJK SC")
        # vg.font_style
        self.assertEqual(vg._font_style, '')
        # vg.antilias
        self.assertEqual(vg.antialias, 'Default')
        self.assertEqual(vg._antialias, CAIRO.ANTIALIAS_DEFAULT)
        # vg.out_dir
        self.assertEqual(vg._out_dir, 'data')
        # vg._grayscale
        self.assertFalse(vg._grayscale)

    def test_font_size_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(font_size=10)
        self.assertEqual(vg._font_size, 10)

        # Test setter after initialization
        vg.font_size = 12
        self.assertEqual(vg._font_size, 12)
        vg.font_size = 14.5
        self.assertEqual(vg._font_size, 14.5)

        # Test exceptions
        with self.assertRaises(ValueError):
            vg = VisualGenerator(font_size=0)
        with self.assertRaises(ValueError):
            vg = VisualGenerator(font_size=-2.5)
        with self.assertRaises(ValueError):
            vg = VisualGenerator(font_size=40)
        with self.assertRaises(ValueError):
            vg.font_size = 0
        with self.assertRaises(ValueError):
            vg.font_size = -2.5
        with self.assertRaises(ValueError):
            vg.font_size = vg.image_size + 1

    def test_image_size_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(image_size=22)
        self.assertEqual(vg._image_size, 22)

        # Test setter after initialization
        vg.image_size = 24
        self.assertEqual(vg._image_size, 24)
        vg.image_size = 25.2
        self.assertEqual(vg._image_size, 25.2)

        # Test exceptions
        with self.assertRaises(ValueError):
            vg = VisualGenerator(image_size=0)
        with self.assertRaises(ValueError):
            vg = VisualGenerator(image_size=-4)
        with self.assertRaises(ValueError):
            vg = VisualGenerator(image_size=2)
        with self.assertRaises(ValueError):
            vg.image_size = 0
        with self.assertRaises(ValueError):
            vg.image_size = -4
        with self.assertRaises(ValueError):
            vg.image_size = vg.font_size - 1

    def test_font_name_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(font_name='Noto Serif CJK SC')
        self.assertEqual(vg._font_name, 'Noto Serif CJK SC')
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Serif CJK SC')
        self.assertEqual(vg._font_face.ft_face.style_name, 'Regular')

        # Test setter after initialization
        vg.font_name = 'Noto Serif CJK TC'
        self.assertEqual(vg._font_name, 'Noto Serif CJK TC')
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Serif CJK TC')
        self.assertEqual(vg._font_face.ft_face.style_name, 'Regular')

        # Test exception
        with self.assertRaises(ValueError):
            vg = VisualGenerator(font_name='Non Exists Font Name')
        with self.assertRaises(ValueError):
            vg.font_name = 'Non Exists Font Name'

        # Test interference
        vg.font_style = 'Light'
        self.assertEqual(vg._font_face.ft_face.style_name, 'Light')
        vg.font_name = 'Noto Sans CJK JP'
        self.assertEqual(vg._font_face.ft_face.style_name, 'Light')


    def test_font_style_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(font_style='Bold')
        self.assertEqual(vg._font_style, 'Bold')
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Sans CJK SC')
        self.assertEqual(vg._font_face.ft_face.style_name, 'Bold')

        # Test setter after initialization
        vg.font_style = 'Thin'
        self.assertEqual(vg._font_style, 'Thin')
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Sans CJK SC')
        self.assertEqual(vg._font_face.ft_face.style_name, 'Thin')

        # TODO: Add checking so that exception raises when style not found!
        # Test that non-exist font style results in default style
        vg.font_style='123'
        self.assertEqual(vg._font_style, '123')
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Sans CJK SC')
        self.assertEqual(vg._font_face.ft_face.style_name, 'Regular')

        # Test interference
        vg.font_name = 'Noto Serif CJK SC'
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Serif CJK SC')
        vg.font_style = 'SemiBold'
        self.assertEqual(vg._font_face.ft_face.family_name, 'Noto Serif CJK SC')

    def test_antialias_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(antialias='Fast')
        self.assertEqual(vg._antialias, CAIRO.ANTIALIAS_FAST)

        # Test setter after initialization
        vg = VisualGenerator(antialias='None')
        self.assertEqual(vg._antialias, CAIRO.ANTIALIAS_NONE)
        vg = VisualGenerator(antialias='Best')
        self.assertEqual(vg._antialias, CAIRO.ANTIALIAS_BEST)
        vg = VisualGenerator(antialias='Good')
        self.assertEqual(vg._antialias, CAIRO.ANTIALIAS_GOOD)
        vg = VisualGenerator(antialias='Default')
        self.assertEqual(vg._antialias, CAIRO.ANTIALIAS_DEFAULT)

        # Test Exception
        with self.assertRaises(ValueError):
            vg = VisualGenerator(antialias='Wild')
        with self.assertRaises(ValueError):
            vg.antialias = "Mild"

    def test_out_dir_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(out_dir='new')
        self.assertEqual(vg._out_dir, 'new')

        # Test setter after initialization
        vg.out_dir = 'newer'
        self.assertEqual(vg._out_dir, 'newer')

    def test_grayscale_setter(self):
        # Test setter in initialization
        vg = VisualGenerator(grayscale=True)
        self.assertTrue(vg._grayscale)
        self.assertEqual(vg._cairo_format, CAIRO.FORMAT_A8)
        self.assertEqual(vg._canvas_color.r, Colour.grey(0, 0).r)
        self.assertEqual(vg._canvas_color.g, Colour.grey(0, 0).g)
        self.assertEqual(vg._canvas_color.b, Colour.grey(0, 0).b)
        self.assertEqual(vg._canvas_color.a, Colour.grey(0, 0).a)
        self.assertEqual(vg._text_color.r, Colour.grey(0, 1).r)
        self.assertEqual(vg._text_color.g, Colour.grey(0, 1).g)
        self.assertEqual(vg._text_color.b, Colour.grey(0, 1).b)
        self.assertEqual(vg._text_color.a, Colour.grey(0, 1).a)

        # Test setter after initialization
        vg.grayscale = False
        self.assertFalse(vg._grayscale)
        self.assertEqual(vg._cairo_format, CAIRO.FORMAT_RGB24)
        self.assertEqual(vg._canvas_color.r, Colour.grey(255, 1).r)
        self.assertEqual(vg._canvas_color.g, Colour.grey(255, 1).g)
        self.assertEqual(vg._canvas_color.b, Colour.grey(255, 1).b)
        self.assertEqual(vg._canvas_color.a, Colour.grey(255, 1).a)
        self.assertEqual(vg._text_color.r, Colour.grey(0, 1).r)
        self.assertEqual(vg._text_color.g, Colour.grey(0, 1).g)
        self.assertEqual(vg._text_color.b, Colour.grey(0, 1).b)
        self.assertEqual(vg._text_color.a, Colour.grey(0, 1).a)

    def test_generate_dataset_from_file(self):
        # Mock two methods:
        # public: self.generate_dataset_from_list
        # private: self._get_out_dir_abs_and_check
        vg = VisualGenerator()
        with patch.object(vg, 'generate_dataset_from_list') as d_from_l, \
             patch.object(vg, '_get_out_dir_abs_and_check',
                          return_value="out_dir") as out_dir_func:
            vg.generate_dataset_from_file(self.src_file, ['Regular', 'Bold'],
                                          ['Default', 'None'])

        # Assert both methods are called exactly once
        d_from_l.assert_called_once_with(self.char_list, ['Regular', 'Bold'],
                                         ['Default', 'None'])
        out_dir_func.assert_called_once()

    def test_generate_dataset_from_list(self):
        # Mock two methods:
        # public: self.visualzie_list
        # private: self._get_out_dir_abs_and_check
        vg = VisualGenerator()
        with patch.object(vg, 'visualize_list') as v_l, \
             patch.object(vg, '_get_out_dir_abs_and_check',
                          return_value="out_dir") as out_dir_func:
            vg.generate_dataset_from_list(self.char_list, ['Regular', 'Bold'],
                                          ['Default', 'None'])

        # Assert that visualize_list is called exactly 4 times
        # Assert that out_dir is checked exactly once
        calls = [call(self.char_list)] * 4
        v_l.assert_has_calls(calls, any_order=True)
        out_dir_func.assert_called_once()

    def test_visualize_list(self):
        # Mock two methods:
        # public: self.visualzie_single
        # private: self._get_out_dir_abs_and_check
        vg = VisualGenerator()
        with patch.object(vg, 'visualize_single') as v_s, \
             patch.object(vg, '_get_out_dir_abs_and_check',
                          return_value="out_dir") as out_dir_func:
            vg.visualize_list(self.char_list, x=3, y=4)

        # Assert that visualize_single is called for each character
        # Assert that out_dir is checked exactly once
        calls = [call(char, False, x=3, y=4) for char in self.char_list]
        v_s.assert_has_calls(calls, any_order=True)
        out_dir_func.assert_called_once()

    def test_visualize_range(self):
        # Mock two methods:
        # public: self.visualzie_single
        # private: self._get_out_dir_abs_and_check
        vg = VisualGenerator()
        with patch.object(vg, 'visualize_single') as v_s, \
            patch.object(vg, '_get_out_dir_abs_and_check',
                         return_value="out_dir") as out_dir_func:
            vg.visualize_range('\u4000', '\u4003', x=5, y=2)

        # Assert that visualize_single is called for each code point in range
        # Assert that out_dir is checked exactly once
        calls = [call(char, False, x=5, y=2) for char in ['\u4000', '\u4001',
                                                          '\u4002', '\u4003']]
        v_s.assert_has_calls(calls, any_order=True)
        out_dir_func.assert_called_once()

        # Test exception
        with self.assertRaises(TypeError):
            vg.visualize_range('4000', '\u4002')
        with self.assertRaises(ValueError):
            vg.visualize_range('\u4004', '\u4000')

    def test_visualize_single(self):
        # This method is very hard to test. We only make sure that the final
        # image is created.
        vg = VisualGenerator(out_dir=self.tmp_dir)
        vg.visualize_single('a')
        vg.font_style = 'Bold'
        vg.visualize_single('b')
        vg.font_style = None
        vg.visualize_single('c')

        # Assert that image files exist
        file_path = os.path.join(self.tmp_dir,
                                 'U+0061_Noto Sans CJK SC_Default.png')
        self.assertTrue(os.path.isfile(file_path))
        file_path = os.path.join(self.tmp_dir,
                                 'U+0062_Noto Sans CJK SC_Bold_Default.png')
        self.assertTrue(os.path.isfile(file_path))
        file_path = os.path.join(self.tmp_dir,
                                 'U+0063_Noto Sans CJK SC_Default.png')
        self.assertTrue(os.path.isfile(file_path))

        # Mock 4 private methods:
        # self._check_create_out_dir_abs
        # self._center_text
        # self._position_text
        # self._get_filename
        with patch.object(vg, '_center_text') as c_t, \
             patch.object(vg, '_position_text') as p_t, \
             patch.object(vg, '_get_filename',
                          return_value='DontCare') as g_f, \
             patch.object(vg, '_check_create_out_dir_abs') as out_dir_func:
            vg.visualize_single('d')
        c_t.assert_called_once()
        p_t.assert_not_called()
        g_f.assert_called_once_with('d')
        out_dir_func.assert_called_once()

        with patch.object(vg, '_center_text') as c_t, \
             patch.object(vg, '_position_text') as p_t, \
             patch.object(vg, '_get_filename',
                          return_value='DontCare') as g_f, \
             patch.object(vg, '_check_create_out_dir_abs') as out_dir_func:
            vg.visualize_single('e', False, x=10, y=10)
        c_t.assert_not_called()
        p_t.assert_called_once()
        g_f.assert_called_once_with('e')
        out_dir_func.assert_not_called()

        # Test exception
        with self.assertRaises(ValueError):
            vg.visualize_single('f', x=10, y=None)

if __name__ == "__main__":
    unittest.main(verbosity=2)