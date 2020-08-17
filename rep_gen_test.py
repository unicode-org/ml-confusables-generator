# Copyright (C) 2020 and later: Google, Inc.

import numpy as np
import os
import random
import shutil
import string
import time
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.modules['tensorflow'] = MagicMock()
sys.modules['dataset_builder'] = MagicMock()
sys.modules['model_builder'] = MagicMock()

from rep_gen import RepresentationGenerator
import dataset_builder as mock_custom_dataset
import model_builder as mock_custom_model

class TestRepresentationGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test data
        cls.tmp_dir = '.tmp' + str(time.time())
        cls.config_file = os.path.join(cls.tmp_dir, 'config.ini')

        # Build temporary testing directory
        print("Building temporary directory {}.".format(cls.tmp_dir))
        os.mkdir(cls.tmp_dir)

        # Build temporary config file
        print("Building temporary source file {}.".format(cls.config_file))
        with open(cls.config_file, 'w+') as f:
            f.write('[GOOMBAS]\n')
            f.write('TOAD = PRINCESS PEACH\n')

    @classmethod
    def tearDownClass(cls):
        print("Deleting temporary directory and file for testing.")
        shutil.rmtree(cls.tmp_dir)

    def test_default_init(self):
        """Test default initialization. When default initialization value
        changes, or any private attribute does not match public attribute, this
        test will fail."""
        # Assert rg._config_path and rg._out_dir are initialized correctly
        rg = RepresentationGenerator()
        self.assertEqual(rg._config_path, 'configs/sample_config.ini')
        self.assertEqual(rg._out_dir, 'embeddings')

        # Assert DatasetBuilder and ModelBuilder is created during
        # initialization.
        mock_custom_dataset.DatasetBuilder.assert_called_with(
            config_path='configs/sample_config.ini', one_hot=False)
        mock_custom_model.ModelBuilder.assert_called_with(
            config_path='configs/sample_config.ini'
        )
        # Assert that model builder is used for getting self._model
        rg._model_builder.get_encoder.assert_called()
        self.assertEqual(rg._model, rg._model_builder.get_encoder())

    def test_config_path_setter(self):
        # Test setter in initialization
        rg = RepresentationGenerator(config_path=self.config_file)
        self.assertEqual(rg._config_path, self.config_file)

        # Test setter after initialization
        rg = RepresentationGenerator()
        rg.config_path = self.config_file
        self.assertEqual(rg._config_path, self.config_file)

        # Test exception
        with self.assertRaises(ValueError):
            RepresentationGenerator(os.path.join(self.config_file,'123'))
        with self.assertRaises(ValueError):
            rg.config_path = os.path.join(self.config_file,'123')

    def test_out_dir_setter(self):
        # Test setter in initialization
        rg = RepresentationGenerator(out_dir=self.tmp_dir)
        self.assertEqual(rg._out_dir, self.tmp_dir)

        # Test setter after initialization
        rg = RepresentationGenerator()
        rg.out_dir = self.tmp_dir
        self.assertEqual(rg._out_dir, self.tmp_dir)

    def test_get_embeddings(self):
        # Assert that dataset_builder
        rg = RepresentationGenerator()
        # Get mocked filename dataset
        mock_img = MagicMock()
        mock_label = MagicMock()
        mock_ds = [(mock_img, mock_label)]

        # Mock function get_filename_dataset.
        with patch.object(rg._dataset_builder, 'get_filename_dataset',
                          return_value=mock_ds) as get_fd:
            codepoints, embeddings = rg.get_embeddings(img_dir=self.tmp_dir)

        get_fd.assert_called_with(self.tmp_dir)
        # Assert prediction is made on mock image
        rg._model.predict.assert_called_with(mock_img)

        # Assert labels (filenames) are handled correctly
        self.assertEqual(mock_label.numpy()[0].decode('utf-8').split('.')[0],
                         codepoints[0])
        # Assert embeddings are result of self._model.predict
        self.assertEqual(rg._model.predict(mock_img)[0], embeddings[0])

    def test_get_embeddings_filename(self):
        """Test filename extraction process (from tf.Tensor to Str). Tensor
        cannot contain string type content, any string by be of bytes type."""

        # Assert that dataset_builder
        rg = RepresentationGenerator()
        # Get mocked filename dataset
        mock_img = MagicMock()
        mock_label = MagicMock()
        mock_ds = [(mock_img, mock_label)]

        # Get fake Tensor content
        letters = string.ascii_uppercase
        filename = ''.join(random.choice(letters) for _ in range(10))
        tensor_content = [str.encode(filename + ".format")]

        # Mock function get_filename_dataset.
        with patch.object(rg._dataset_builder, 'get_filename_dataset',
                          return_value=mock_ds) as get_fd,\
             patch.object(mock_label, 'numpy', return_value=tensor_content):
            codepoints, _ = rg.get_embeddings(img_dir=self.tmp_dir)

        self.assertEqual(codepoints[0], filename)


    def test_write_embeddings_from_image(self):
        # Mock inputs
        img_dir, out_file, char_as_label = MagicMock(), MagicMock(), MagicMock()
        # Mock codepoints and embeddings
        codepoints, embeddings = MagicMock(), MagicMock()
        rg = RepresentationGenerator()

        # Mock two functions: rg.get_embeddings and
        # rg.write_embeddings_from_list
        with patch.object(rg, 'get_embeddings',
                          return_value=(codepoints, embeddings)) as get_emb, \
             patch.object(rg, 'write_embeddings_from_list') as w_emb_f_l:
            rg.write_embeddings_from_image(img_dir=img_dir, out_file=out_file,
                                           char_as_label=char_as_label)

        get_emb.assert_called_with(img_dir=img_dir)
        w_emb_f_l.assert_called_once_with(codepoints, embeddings, out_file,
                                          char_as_label)

    def test_write_embeddings_from_list(self):
        # Use reasonable inputs
        codepoints = ['U+4e00_additional_info', 'U+4e01_additional_info']
        embeddings = [[random.random(), random.random()],
                      [random.random(), random.random()]]
        filename = 'test' + str(time.time())
        vec_file = os.path.join(self.tmp_dir, filename + '_vec.tsv')
        meta_file = os.path.join(self.tmp_dir, filename + '_meta.tsv')

        # Assert that:
        #   1. Output files are directed to out_dir
        #   2. Output files have the correct names
        #   3. Output files have the correct format
        #   4. Output files have the correct value

        # Make sure no vec_file exists in self.tmp_dir
        if os.path.isfile(vec_file):
            os.remove(vec_file)
        if os.path.isfile(meta_file):
            os.remove(meta_file)

        # Write codepoints and embeddings to file
        rg = RepresentationGenerator(out_dir=self.tmp_dir)
        rg.write_embeddings_from_list(codepoints, embeddings, out_file=filename,
                                      char_as_label=False)

        # Read newly generated vec file
        output_embs = np.genfromtxt(fname=vec_file, delimiter="\t")
        # Read newly generated meta file
        with open(meta_file,'r') as f_in:
            output_chars = [label.strip() for label in f_in.readlines()]
        # Check vec file content
        self.assertTrue(np.array_equal(embeddings, output_embs))
        # Check meta file content
        self.assertTrue(codepoints == output_chars)

        os.remove(vec_file)
        os.remove(meta_file)


    def test_write_embeddings_from_list_output_char(self):
        # Use reasonable inputs
        codepoints = ['U+4e00_additional_info', 'U+4e01_additional_info']
        embeddings = [[random.random(), random.random()],
                      [random.random(), random.random()]]
        filename = 'test' + str(time.time())
        vec_file = os.path.join(self.tmp_dir, filename + '_vec.tsv')
        meta_file = os.path.join(self.tmp_dir, filename + '_meta.tsv')

        # Make sure no vec_file exists in self.tmp_dir
        if os.path.isfile(vec_file):
            os.remove(vec_file)
        if os.path.isfile(meta_file):
            os.remove(meta_file)

        # Write codepoints and embeddings to file
        rg = RepresentationGenerator(out_dir=self.tmp_dir)
        rg.write_embeddings_from_list(codepoints, embeddings, out_file=filename,
                                      char_as_label=True)

        # Read newly generated vec file
        output_embs = np.genfromtxt(fname=vec_file, delimiter="\t")
        # Read newly generated meta file
        with open(meta_file, 'r') as f_in:
            output_chars = [label.strip() for label in f_in.readlines()]
        # Check vec file content
        self.assertTrue(np.array_equal(embeddings, output_embs))
        self.assertFalse(codepoints == output_chars)
        self.assertTrue(output_chars == ['一', '丁'])

        os.remove(vec_file)
        os.remove(meta_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
