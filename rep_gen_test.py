# Copyright (C) 2020 and later: Google, Inc.

import numpy as np
import os
import random
import shutil
import time
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.modules['tensorflow'] = MagicMock()
sys.modules['custom_dataset'] = MagicMock()
sys.modules['custom_model'] = MagicMock()

from rep_gen import RepresentationGenerator
import custom_dataset as mock_custom_dataset
import custom_model as mock_custom_model

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
        # Assert that dataset_builder and
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

        # Assert that:
        #   1. Output files are directed to out_dir
        #   2. Output files have the correct names
        #   3. Output files have the correct format
        #   4. Output files have the correct value
        #   5. Argument char_as_label work as expected

        # Write codepoints and embeddings to file
        rg = RepresentationGenerator(out_dir=self.tmp_dir)
        rg.write_embeddings_from_list(codepoints, embeddings, out_file=filename,
                                      char_as_label=True)

        # Read newly generated vec file
        output_embs = np.genfromtxt(
            fname=os.path.join(self.tmp_dir, filename + '_vec.tsv'),
            delimiter="\t")
        # Read newly generated meta file
        with open(os.path.join(self.tmp_dir, filename + '_meta.tsv'),'r') \
            as f_in:
            output_chars = [label.strip() for label in f_in.readlines()]
        # Check vec file content
        self.assertTrue(np.array_equal(embeddings, output_embs))
        # Check meta file cntent
        self.assertFalse(codepoints == output_chars)
        self.assertTrue(output_chars == ['一', '丁'])

        # Write codepoints and embeddings to file
        rg.write_embeddings_from_list(codepoints, embeddings, out_file=filename,
                                      char_as_label=False)

        # Read newly generated meta file
        with open(os.path.join(self.tmp_dir, filename + '_meta.tsv'), 'r') \
            as f_in:
            output_chars = [label.strip() for label in f_in.readlines()]
        self.assertTrue(codepoints == output_chars)



if __name__ == "__main__":
    unittest.main(verbosity=2)