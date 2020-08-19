# Copyright (C) 2020 and later: Unicode, Inc. and others.
# License & terms of use: http://www.unicode.org/copyright.html

import cv2
from distance_metrics import ImgFormat, Distance
import numpy as np
import os
import shutil
import time
import unittest
from unittest.mock import MagicMock, patch, call

import sys
sys.modules['sklearn.decomposition'] = MagicMock()
from rep_cls import RepresentationClustering

class TestRepresentationClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test data
        cls.tmp_dir = '.tmp' + str(time.time())
        cls.embedding_file = os.path.join(cls.tmp_dir, "vec.tsv")
        cls.label_file = os.path.join(cls.tmp_dir, "meta.tsv")
        cls.embeddings = np.random.uniform(low=-3.0, high=3.0, size=(5, 100))
        cls.labels = [chr(num) for num in range(0x4e00, 0x4e05)]
        cls.img_dir = os.path.join(cls.tmp_dir, 'img_dir')
        cls.img_names = ['U+{:04X}'.format(num) + '_info.png' for num in
                         range(0x4e00, 0x4e05)]


        # Build temporary testing directory and add data
        print("Building temporary directory {}.".format(cls.tmp_dir))
        os.mkdir(cls.tmp_dir)
        os.mkdir(cls.img_dir)

        print("Building temporary embedding and label file.")
        np.savetxt(cls.embedding_file, cls.embeddings,
                   delimiter='\t')
        with open(cls.label_file, 'w+') as f_out:
            for label in cls.labels:
                f_out.write(label)
                f_out.write('\n')

        print("Building temporary .png images for testing.")
        img = np.zeros([5,5,3])
        for name in cls.img_names:
            cv2.imwrite(os.path.join(cls.img_dir, name), img,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

    @classmethod
    def tearDownClass(cls):
        print("Deleting temporary directory and file for testing.")
        shutil.rmtree(cls.tmp_dir)


    def test_default_init(self):
        """Test default initialization. When default initialization value
        changes, or any private attribute does not match public attribute, this
        test will fail."""
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir)

        # rc.embedding_file
        self.assertEqual(rc._embedding_file, self.embedding_file)
        # rc.label_file
        self.assertEqual(rc._label_file, self.label_file)
        # rc.img_dir
        self.assertEqual(rc._img_dir, self.img_dir)
        # rc.n_candidates
        self.assertEqual(rc._n_candidates, 100)
        # rc.pca_dimensions:
        self.assertEqual(len(rc._reps), len(rc.pca_dimensions))
        # rc.img_format
        self.assertEqual(rc._img_format, ImgFormat.RGB)
        # rc.primary_distance_type
        self.assertEqual(rc.primary_distance_type, "manhattan")
        # rc.secondary_distance_type
        self.assertEqual(rc.secondary_distance_type, "sum_squared")
        # rc.secondary_filter_threshold
        self.assertEqual(rc.secondary_filter_threshold, 0.1)

    def test_embedding_file_setter(self):
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir)
        self.assertTrue(np.array_equal(rc.embeddings, self.embeddings))

        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file="123",
                                          label_file=self.label_file,
                                          img_dir=self.img_dir)

    def test_label_file_setter(self):
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir)
        self.assertEqual(rc.labels, self.labels)

        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file=self.embedding_file,
                                          label_file="123",
                                          img_dir=self.img_dir)

    def test_img_dir_setter(self):
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir)
        for label, img_name in zip(self.labels, self.img_names):
            self.assertEqual(rc._label_img_map[label],
                             os.path.join(self.img_dir, img_name))

        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file=self.embedding_file,
                                          label_file=self.label_file,
                                          img_dir="123")

    def test_n_candidates_setter(self):
        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file=self.embedding_file,
                                          label_file=self.label_file,
                                          img_dir=self.img_dir,
                                          n_candidates=-2)

        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file=self.embedding_file,
                                          label_file=self.label_file,
                                          img_dir=self.img_dir,
                                          n_candidates=-1.2)

    def test_img_format_setter(self):
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir,
                                      img_format=ImgFormat.A8)
        self.assertEqual(rc.img_format, ImgFormat.A8)

        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file=self.embedding_file,
                                          label_file="123",
                                          img_dir=self.img_dir,
                                          img_format=123)

    def test_pca_dimensions_setter(self):
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir,
                                      pca_dimensions=[10,20,50])
        # Assert models are created with correct dimensions
        for model, dimension in zip(rc._pca_models, rc._pca_dimensions):
            self.assertEqual(model,
                             sys.modules['sklearn.decomposition'].PCA
                             (n_components=dimension))
        # Assert models are fitted correctly
        for model in rc._pca_models:
            model.fit.assert_called_with(rc.embeddings)
        # Assert representations are derived from models
        for rep, model in zip(rc._reps, rc._pca_models):
            self.assertTrue(rep, model.fit(rc.embeddings))
        # Assert private field is set correctly
        self.assertEqual(rc._pca_dimensions, [10,20,50])

        with self.assertRaises(ValueError):
            rc = RepresentationClustering(embedding_file=self.embedding_file,
                                          label_file="123",
                                          img_dir=self.img_dir,
                                          pca_dimensions=[-1])

    def test_get_candidate_pool_for_char(self):
        # This function is hard to test due to configurability of
        # sorting and filtering methods.
        # We only test basic functionality.
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir, n_candidates=3)
        # Change rc._reps to real embeddings
        rc._reps = [rc.embeddings]
        candidate_pool, distances = rc.get_candidate_pool_for_char('\u4e00')

        # Assert that correct number of candidates are kept before secondary
        # filtering

        # Assert no more than n candidates are selected
        self.assertLessEqual(len(candidate_pool), rc.n_candidates)
        for key, value in distances.items():
            # Assert no more than n candidates are considered
            self.assertLessEqual(len(value), rc.n_candidates)
            # Assert the closest character is itself
            self.assertEqual(value[0][0], 0)
            self.assertEqual(value[0][1], '\u4e00')

    def test_filter_candidate_pool(self):
        # This function is hard to test due to configurability of
        # sorting and filtering methods.
        # We only test basic functionality.
        rc = RepresentationClustering(embedding_file=self.embedding_file,
                                      label_file=self.label_file,
                                      img_dir=self.img_dir, n_candidates=3)
        # Change rc._reps to real embeddings
        rc._reps = [rc.embeddings]

        # Fake candidate pool
        candidate_pool = set(['\u4e00', '\u4e01', '\u4e02'])
        confusables = rc.filter_candidate_pool(candidate_pool, '\u4e00')

        # Assert the character itself is not in the confusables
        self.assertFalse('\u4e00' in confusables)


if __name__ == "__main__":
    unittest.main(verbosity=2)
