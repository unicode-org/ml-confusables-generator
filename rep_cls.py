r"""Representation (embeddings) clustering scripts. Modules for finding confusables."""
from distance_metrics import ImgFormat, Distance, calculate_from_path
from sklearn.decomposition import PCA
import numpy as np
import os
import bisect

class RepCls:
    def __init__(self, embedding_file, label_file, img_dir, n_candidates=50, pca_dimensions=[5, 10, 20, 100, 500],
                 img_format=ImgFormat.RGB, primary_distance_type="manhattan", secondary_distance_type="sum_squared",
                 secondary_filter_threshold=0.1):

        # Read embeddings and labels from file and store in self.embeddings and self.labels
        self.embeddings = None
        self.embedding_file = embedding_file
        self.labels = None
        self.label_file = label_file

        # Set img_dir and names of all images
        # Get mapping from label to image for naive distance calculation
        self.__img_names = None
        self.__label_img_map = None
        self.img_dir = img_dir

        # Assertion that # of embeddings, labels match
        if len(self.embeddings) != len(self.labels):
            raise ValueError("Embeddings and labels should have the same number of entries.")
        # Assertion that there are not less images
        if len([name for name in self.__img_names if os.path.isfile(name)]) < len(self.labels):
            raise ValueError("Not enough image files, make sure img_dir contains the whole dataset.")

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
        return self.__embedding_file

    @property
    def label_file(self):
        return self.__label_file

    @property
    def img_dir(self):
        return self.__img_dir

    @property
    def n_candidates(self):
        return self.__n_candidates

    @property
    def pca_dimensions(self):
        return self.__pca_dimensions

    @property
    def img_format(self):
        return self.__img_format

    @embedding_file.setter
    def embedding_file(self, embedding_file):
        # Check that embeddings exists
        if not os.path.isfile(embedding_file):
            raise ValueError('File {} does not exist.'.format(embedding_file))
        # Read embeddings into np.ndarray
        print('Reading embeddings from file {}...'.format(embedding_file))
        self.embeddings = np.genfromtxt(fname=embedding_file, delimiter="\t")
        self.__embedding_file = embedding_file
        print('Successfully read from file {}.'.format(embedding_file))

    @label_file.setter
    def label_file(self, label_file):
        # Check that labels exists
        if not os.path.isfile(label_file):
            raise ValueError('File {} does not exist.'.format(label_file))
        # Read labels into dictionary
        print('Reading labels from file {}...'.format(label_file))
        self.labels = []
        with open(label_file, "r") as f_in:
            for line in f_in:
                self.labels.append(line.split('\n')[0])
        self.__label_file = label_file
        print('Successfully read from file {}.'.format(label_file))

    @img_dir.setter
    def img_dir(self, img_dir):
        # Check directory exists
        if not os.path.isdir(img_dir):
            raise ValueError('Direcotry {} does not exist.'.format(img_dir))
        # Read all image names
        self.__img_names = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
        self.__img_dir = img_dir

        # Create map form label to image name
        __codepoints = [name.split('/')[-1].split('_')[0] for name in self.__img_names]
        __keys = [chr(int('0x' + codepoint[2:], 16)) for codepoint in __codepoints]
        self.__label_img_map = dict(zip(__keys, self.__img_names))

    @n_candidates.setter
    def n_candidates(self, n_candidates):
        if n_candidates <= 0 or type(n_candidates) != int:
            raise ValueError("Number of candidates must be a positive integer.")
        self.__n_candidates = n_candidates

    @pca_dimensions.setter
    def pca_dimensions(self, pca_dimensions):
        for dimension in pca_dimensions:
            if dimension <= 0 or type(dimension) != int:
                raise ValueError('PCA component number must be a positive integer.')
        self.__pca_dimensions = pca_dimensions

        # Build PCA models
        self.__pca_models = []
        print("Building PCA models.")
        for dimension in pca_dimensions:
            self.__pca_models.append(PCA(n_components = dimension))
        # Fit models with full embeddings
        print("Fitting PCA models.")
        for model in self.__pca_models:
            model.fit(self.embeddings)

        # Get reduced embeddings as representations
        print("Generating reduced embeddings as representations.")
        self.__reps =[]
        for model in self.__pca_models:
            reduced_embeddings = model.fit_transform(self.embeddings)
            self.__reps.append(reduced_embeddings)

    @img_format.setter
    def img_format(self, img_format):
        # Distance metrics setup
        if img_format not in list(ImgFormat):
            raise TypeError('Expect img_format to be a member of Format class.')
        self.__img_format = img_format

    def get_pca_results(self):
        return self.__reps

    def get_confusables_for_char(self, char, output_codepoint=False):
        # Get character index in labels and embeddings
        idx = self.labels.index(char)
        # Get a pool of possible candidates for secondary filter
        candidate_pool = set()
        # Store distances between all confusables and anchor
        candidate_dis = dict()
        for embs in self.__reps:
            # Get embedding anchor to compare with others
            emb_anchor = embs[idx]

            # Get primary distance metrics
            embedding_metrics = Distance(ImgFormat.EMBEDDINGS).get_metrics()
            if self.primary_distance_type not in embedding_metrics.keys():
                raise ValueError("Expect primary_distance_type to be one of {}.".format(embedding_metrics.keys()))
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
                        # If the distance is lower than the largest of the candidates we only keep top N
                        bisect.insort(topN, (dis, label))
                        topN = topN[:self.n_candidates-1]

            # Store all candidate distances
            candidate_dis["PCA" + str(embs.shape[1])] = topN
            candidate_pool = candidate_pool.union(set([entry[1] for entry in topN]))

        # Get secondary distance metrics
        image_metrics = Distance(self.img_format).get_metrics()
        if self.secondary_distance_type not in image_metrics.keys():
            raise ValueError("Expect secondary_distance_type to be one of {}.".format(image_metrics.keys()))
        secondary_dis = image_metrics[self.secondary_distance_type]

        # Filter candidate pool to get confusables

        confusables = []
        for candidate in candidate_pool:
            if ord(char) == ord(candidate):
                continue
            if calculate_from_path(secondary_dis, self.__label_img_map[char], self.__label_img_map[candidate]) \
                <= self.secondary_filter_threshold:
                confusables.append(candidate)

        return confusables, candidate_dis







if __name__ == "__main__":
    rc = RepCls(embedding_file='embeddings/full_data_triplet1.0_vec.tsv',
                label_file='embeddings/full_data_triplet1.0_meta.tsv',
                img_dir='data/full_data/')
    import pdb;pdb.set_trace()

