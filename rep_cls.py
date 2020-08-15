# Copyright (C) 2020 and later: Google, Inc.

from distance_metrics import ImgFormat, Distance, calculate_from_path

import argparse
from argparse import RawDescriptionHelpFormatter
from sklearn.decomposition import PCA
import numpy as np
import os
import bisect

class RepresentationClustering:
    """Representation (embeddings) clustering generator.

    Initialization:
        >>> rc = RepCls(embedding_file='embeddings/full_data_vec.tsv', \
        >>>             label_file='embeddings/full_data_meta.tsv', \
        >>>             img_dir='data/full_data/')
    Configurations:
        >>> rc.n_candidates = 50
        >>> rc.pca_dimensions = [5, 10, 100]
        >>> rc.primary_distance_type = 'manhattan'
        >>> rc.secondary_distance_type = 'sum_squared'
        >>> rc.secondary_filter_threshold = 0.1
    Generate confusable cluster centered at single character
        >>> confusables, distances = rc.get_confusables_for_char('褢')
        >>> ['裹', '裛', '裏', '裏']
    """

    def _init_(self, embedding_file, label_file, img_dir, n_candidates=100,
                 pca_dimensions=[5, 10, 20, 100, 500], img_format=ImgFormat.RGB,
                 primary_distance_type="manhattan",
                 secondary_distance_type="cross_correlation",
                 secondary_filter_threshold=0.1):
        """Specify and read embedding file, label file and image directory.
        Choose number of candidate to consider. Set primary and secondary
        distance metrics. Specify number of components for PCA algorithm. Run
        PCA on embeddings and store results. Specify threshold for secondary
        distance metric.

        Args:
            embedding_file: Str, path to the embedding (_vec) .tsv file.
            label_file: Str, path to the label (_meta) .tsv file.
            img_dir: Str, location of the image folder generated by
                VisualGenerator.
            n_candidates: Int, number of possible candidates to consider
                (default 100).
            pca_dimensions: List of Int, number of principal components
                (default [5, 10, 20, 100, 500]).
            img_format: ImgFormat, format of specified images (Default RGB).
            primary_distance_type: Str, name of the primary distance metric
                (default "manhattan").
            secondary_distance_type: Str, name of the secondary distance metric
                (default "cross_correlation").
            secondary_filter_threshold: Float, only confusables with distance
                UNDER this threshold will be selected (default 0.1).

        Raises:
            ValueError: if number of embeddings and labels do not match.
            ValueError: if no 1-1 mapping between labels and images exists
                (In setters).
            ValueError: if embedding_file or label_file does not exists.
            ValueError: if img_dir does not exists.
            ValueError: if n_candidates is not a positive integer.
            ValueError: if numbers in pca_dimensions are not all positive
                integers.
            TypeError: if img_format is not an ImgFormat object.
        """

        # Read embeddings and labels from file and store in self.embeddings and
        # self.labels
        self.embeddings = None
        self.embedding_file = embedding_file
        self.labels = None
        self.label_file = label_file

        # Set img_dir and names of all images
        # Get mapping from label to image for naive distance calculation
        self._img_names = None
        self._label_img_map = None
        self.img_dir = img_dir

        # Assertion that # of embeddings, labels match
        if len(self.embeddings) != len(self.labels):
            raise ValueError("Embeddings and labels should have the same number"
                             " of entries.")
        # Assertion that there are not less images than embeddings
        if len([name for name in self._img_names if
                os.path.isfile(name)]) < len(self.labels):
            raise ValueError("Not enough image files, make sure img_dir "
                             "contains the whole dataset.")

        # Set number of candidates
        self.n_candidates = n_candidates

        # Get low dimensional embeddings based on PCA
        self.pca_dimensions = pca_dimensions

        # Set image format
        self.img_format = img_format

        # Primary filter setup
        self.primary_distance_type = primary_distance_type
        # Secondary filter setup
        self.secondary_distance_type = secondary_distance_type
        self.secondary_filter_threshold = secondary_filter_threshold

    @property
    def embedding_file(self):
        """
        Returns:
            self._embedding_file: Str, path to the embedding (_vec) .tsv file.
        """
        return self._embedding_file

    @property
    def label_file(self):
        """
        Returns:
            self._label_file: Str, path to the label (_meta) .tsv file.
        """
        return self._label_file

    @property
    def img_dir(self):
        """
        Returns:
            self._img_dir: Str, location of the image folder generated by
                VisualGenerator.
        """
        return self._img_dir

    @property
    def n_candidates(self):
        """
        Returns:
            self._n_candidates: Int, number of candidates to choose from.
        """
        return self._n_candidates

    @property
    def pca_dimensions(self):
        """
        Returns:
            self._pca_dimensions: List of Int, number of principal components.
        """
        return self._pca_dimensions

    @property
    def img_format(self):
        """
        Returns:
            self._img_format: ImgFormat, format of specified images.
        """
        return self._img_format

    @embedding_file.setter
    def embedding_file(self, embedding_file):
        """Check that embedding file exists and read into self.embeddings.
        Raise ValueError if file does not exist. Also set
        self._embedding_file."""
        # Check that embeddings exists
        if not os.path.isfile(embedding_file):
            raise ValueError('File {} does not exist.'.format(embedding_file))
        # Read embeddings into np.ndarray
        print('Reading embeddings from file {}...'.format(embedding_file))
        self.embeddings = np.genfromtxt(fname=embedding_file, delimiter="\t")
        self._embedding_file = embedding_file
        print('Successfully read from file {}.'.format(embedding_file))

    @label_file.setter
    def label_file(self, label_file):
        """Check that label file exists and read into self.labels. Raise
        ValueError if file does not exist. Also set self._label_file."""
        # Check that labels exists
        if not os.path.isfile(label_file):
            raise ValueError('File {} does not exist.'.format(label_file))
        # Read labels into dictionary
        print('Reading labels from file {}...'.format(label_file))
        self.labels = []
        with open(label_file, "r") as f_in:
            for line in f_in:
                self.labels.append(line.split('\n')[0])
        self._label_file = label_file
        print('Successfully read from file {}.'.format(label_file))

    @img_dir.setter
    def img_dir(self, img_dir):
        """Read all file names in img_dir and create mapping from all labels to
        file names. Raise ValueError if specified directory does not exist.
        Also set self._label_img_map."""
        # Check directory exists
        if not os.path.isdir(img_dir):
            raise ValueError('Directory {} does not exist.'.format(img_dir))
        # Read all image names
        self._img_names = [os.path.join(img_dir, name) for name in
                            os.listdir(img_dir)]
        self._img_dir = img_dir

        # Create map form label to image name
        _codepoints = [name.split('/')[-1].split('_')[0] for name in
                       self._img_names]
        _keys = [chr(int('0x' + codepoint[2:], 16)) for codepoint in
                 _codepoints]
        self._label_img_map = dict(zip(_keys, self._img_names))

    @n_candidates.setter
    def n_candidates(self, n_candidates):
        """Set self._n_candidates. Raise ValueError if n_candidates is less or
        equal to zero or is not an integer."""
        if n_candidates <= 0 or type(n_candidates) != int:
            raise ValueError("Number of candidates must be a positive integer.")
        self._n_candidates = n_candidates

    @pca_dimensions.setter
    def pca_dimensions(self, pca_dimensions):
        """Create d PCA models for the d dimensions in pca_dmensions. Fit all
        models with the full embeddings and create reduced embeddings in
        self._reps. If any element in pca_dimensions is less or equal to 0 or
        is not an integer, raise ValueError."""
        for dimension in pca_dimensions:
            if dimension <= 0 or type(dimension) != int:
                raise ValueError('PCA component number must be a positive '
                                 'integer.')
        self._pca_dimensions = pca_dimensions

        # Build PCA models
        self._pca_models = []
        print("Building PCA models.")
        for dimension in pca_dimensions:
            self._pca_models.append(PCA(n_components = dimension))
        # Fit models with full embeddings
        print("Fitting PCA models.")
        for model in self._pca_models:
            model.fit(self.embeddings)

        # Get reduced embeddings as representations
        print("Generating reduced embeddings as representations.")
        self._reps =[]
        for model in self._pca_models:
            reduced_embeddings = model.fit_transform(self.embeddings)
            self._reps.append(reduced_embeddings)

    @img_format.setter
    def img_format(self, img_format):
        """Raise TypeError if img_format is not an ImgFormat enum class."""
        # Distance metrics setup
        if img_format not in list(ImgFormat):
            raise TypeError('Expect img_format to be a member of Format class.')
        self._img_format = img_format

    def get_pca_results(self):
        """Get all embeddings after PCA selection."""
        return self._reps

    def get_confusables_for_char(self, char):
        """Obtain confusables clustered around a certain character. For all
        PCA selections, get the nearest n_candidates characters according to
        their embeddings and add to candidate pool. From candidate pool,
        filter again based on secondary distance methods.

        Args:
            char: Char, single character, must exists in labels

        Returns:
            confusables: List of Char, a list of confusables
            candidate_dis: Dict, mapping from candidates to their respective
                distances
        """
        # Get character index in labels and embeddings
        idx = self.labels.index(char)
        # Get a pool of possible candidates for secondary filter
        candidate_pool = set()
        # Store distances between all confusables and anchor
        candidate_dis = dict()
        for embs in self._reps:
            # Get embedding anchor to compare with others
            emb_anchor = embs[idx]

            # Get primary distance metrics
            embedding_metrics = Distance(ImgFormat.EMBEDDINGS).get_metrics()
            if self.primary_distance_type not in embedding_metrics.keys():
                raise ValueError("Expect primary_distance_type to be one of {}."
                                 .format(embedding_metrics.keys()))
            primary_dis = embedding_metrics[self.primary_distance_type]

            # Get distance from anchor embedding to all other embeddings
            distances = []
            for emb in embs:
                distances.append(primary_dis(emb_anchor, emb))
            label_dis_pairs = list(zip(self.labels, distances))

            # Get top n candidates using the primary distance metric
            topN = []
            for label, dis in label_dis_pairs:
                if len(topN) < self.n_candidates:
                    # Append reversed tuple for sorting
                    bisect.insort(topN, (dis, label))
                else:
                    if dis < topN[self.n_candidates-1][0]:
                        # If the distance is lower than the largest of the
                        # candidates we only keep top N
                        bisect.insort(topN, (dis, label))
                        topN = topN[:self.n_candidates-1]

            # Store all candidate distances
            candidate_dis["PCA" + str(embs.shape[1])] = topN
            candidate_pool = candidate_pool.union(
                set([entry[1] for entry in topN]))

        # Get secondary distance metrics
        image_metrics = Distance(self.img_format).get_metrics()
        if self.secondary_distance_type not in image_metrics.keys():
            raise ValueError("Expect secondary_distance_type to be one of {}."
                             .format(image_metrics.keys()))
        secondary_dis = image_metrics[self.secondary_distance_type]

        # Filter candidate pool to get confusables
        confusables = []
        for candidate in candidate_pool:
            if ord(char) == ord(candidate):
                continue
            if calculate_from_path(secondary_dis, self._label_img_map[char],
                                   self._label_img_map[candidate]) <= \
                self.secondary_filter_threshold:
                confusables.append(candidate)

        return confusables, candidate_dis


if __name__ == "__main__":
    formatter = RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description='Usage: \n',
                                     formatter_class=formatter)
    parser.add_argument('--embedding_file', type=str, required=True, nargs=1,
                        help='Path to the embedding (_vec.tsv) file.')
    parser.add_argument('--label_file', type=str, required=True, nargs=1,
                        help='Path to the label (_meta.tsv) file.')
    parser.add_argument('--img_dir', type=str, required=True, nargs=1,
                        help="Path to image directory generated by VisGen.")
    parser.add_argument('--n_candidates', type=int, default=50, required=False,
                        nargs=1, help="Relative path to output directory.")
    parser.add_argument('--primary_distance_type', type=str,
                        default='manhattan', required=False, nargs=1,
                        help="Name of the primary distance metric.")
    parser.add_argument('--secondary_distance_type', type=str,
                        default='sum_squared', required=False, nargs=1,
                        help="Name of the secondary distance metric.")
    parser.add_argument('--secondary_filter_threshold', type=float, default=0.1,
                        required=False, nargs=1, help="Threshold of the "
                        "secondary distance metric.")
    parser.add_argument('--anchor_char', type=str, required=True, nargs=1,
                        help="The anchor character.")
    args = parser.parse_args()

    rc = RepresentationClustering(
        embedding_file=args.embedding_file, label_file=args.label_file,
        img_dir=args.img_dir, n_candidates=args.n_candidates,
        primary_distance_type=args.primary_distance_type,
        secondary_distance_type=args.secondary_distance_type,
        secondary_filter_threshold=args.secondary_filter_threshold)
    confusables, distances = rc.get_confusables_for_char(args.anchor_char)
    print("Anchor character: {}".format(args.anchor_char))
    print("Confusables: ")
    print(confusables)