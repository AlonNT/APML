import pickle
import os
import numpy as np
from scipy.special import logsumexp
from itertools import product

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'


def get_data(data_path: str = 'PoS_data.pickle',
             words_path: str = 'all_words.pickle',
             pos_path: str = 'all_PoS.pickle') -> (list, list, list):
    """
    Loads the data from the pickle files which are located in the directory 'supplementaries',
    and the names are given to this function
    :param data_path: name of the pickle file containing the data.
    :param words_path: name of the pickle file containing all possible words.
    :param pos_path: name of the pickle file containing all possible pos-taggings.
    :return: A tuple containing the above three elements.
    """
    with open(os.path.join('.', 'supplementaries', data_path), 'rb') as f:
        dataset = pickle.load(f)
    with open(os.path.join('.', 'supplementaries', words_path), 'rb') as f:
        words = pickle.load(f)
    with open(os.path.join('.', 'supplementaries', pos_path), 'rb') as f:
        pos_tags = pickle.load(f)

    return dataset, words, pos_tags


def convert_to_numpy(dataset: list, rare_threshold: int = 0) -> (np.ndarray, np.ndarray):
    """
    Create a sparse representation of the data, using numpy's unique function.

    index2word, words_indices, words_count,
    index2pos, pos_indices, pos_count,
    n_samples, max_sentence_length, max_word_length_new, max_pos_length

    Reconstructing the sentences array can be done using
    index2word[words_indices].reshape(n_samples, max_sentence_length)
    It's also possible to view the dataset containing as integers representing the words.
    Zero will be like a NaN in a pandas DataFrame, which means that there is no value there.
    (We can't put np.nan because this is an integer array).
    Reconstructing the sentences array using the indices representing the words
    (instead of the strings themselves) can be done using
    np.arange(len(index2word))[words_indices].reshape(n_samples, max_sentence_length)
    And the same holds for the pos-tags.

    This function also handles the rare words by replacing them with the RARE_WORD symbol.
    Furthermore, it adds START and END symbols for the sentences and the pos-tags.

    :param dataset: A list containing tuples, where each tuple is a single sample.
                    - The first element is the PoS tagging of the sentence.
                    - The second element sentence itself.
    :param rare_threshold: An integer representing the threshold for a word to be regarded as a "rare word".
                           By default it is 0, meaning that nothing will be done with the rare-words.
    :return: sentences_new, pos_tags: numpy arrays containing strings.
                                      The number of rows is the number of samples (i.e. sentence + pos-tag)
                                      in the given dataset. The number of columns equals
                                      the length of the longest sample.
    """
    n_samples = len(dataset)

    # Define the maximal length of a sentence.
    # This is in order to create a 'padded' numpy array that will contain the training-set.
    max_sentence_length = max([len(sample[0]) for sample in dataset])
    max_sentence_length += 2  # each sentence is being added with a START and END symbols.

    # Calculate the maximal length of a word in the dataset, as well as the maximal length of a pos-tag.
    # This is because that in order to create a numpy array containing strings,
    # one must know the maximal length of a string in the array.
    sentences_list = [sample[1] for sample in dataset]
    pos_tags_list = [sample[0] for sample in dataset]
    max_word_length = max([len(word) for sentence in sentences_list for word in sentence] +
                          [len(START_WORD), len(END_WORD)])
    max_pos_length = max([len(pos_tag) for pos_tags in pos_tags_list for pos_tag in pos_tags] +
                         [len(START_STATE), len(END_STATE)])

    # Define two 2D arrays containing strings (with the corresponding maximal size).
    # These will hold the sentences and the pos-tags, and an empty string means
    # that there is nothing there (like a NaN in a pandas DataFrame).
    sentences = np.zeros(shape=(n_samples, max_sentence_length), dtype=np.dtype(('U', max_word_length)))
    pos_tags = np.zeros(shape=(n_samples, max_sentence_length), dtype=np.dtype(('U', max_pos_length)))

    # Since the sentences are in different lengths, we can't initialize the whole padded numpy array directly,
    # and we have to manually add each sentence according to its length.
    for i in range(n_samples):
        sentence_pos_tags = dataset[i][0]
        sentence = dataset[i][1]

        # If the length of the sentence differ from the length of the pos-tagging, something bad happened...
        assert len(sentence) == len(sentence_pos_tags)

        # Add the START & END symbols for both the sentence and its pos-tagging.
        sentence = [START_WORD] + sentence + [END_WORD]
        sentence_pos_tags = [START_STATE] + sentence_pos_tags + [END_STATE]

        # Set the relevant rows in the numpy 2D array.
        sentences[i, :len(sentence)] = sentence
        pos_tags[i, :len(sentence)] = sentence_pos_tags

    # Create the sparse representation of the data, using numpy's unique function.
    index2word, words_indices, words_count = np.unique(sentences, return_inverse=True, return_counts=True)

    # Replace the rare words with the RARE_WORD symbol.
    index2word_new = np.copy(index2word)
    index2word_new[words_count < rare_threshold] = RARE_WORD

    # Now now that we removed a lot of words, the maximal word's length may be less (it's actually 21 v.s. 54).
    # So define the new data-type to be of the new max-length, to increase efficiency.
    max_word_length_new = np.amax(np.char.str_len(index2word_new))
    index2word_new = index2word_new.astype(dtype=np.dtype(('U', max_word_length_new)))

    # Construct the new sentences, replacing the rare words with the RARE_WORD symbol.
    sentences_new = index2word_new[words_indices].reshape(n_samples, max_sentence_length)

    return sentences_new, pos_tags


def split_train_test(sentences: np.ndarray,
                     pos_tags: np.ndarray,
                     train_ratio: float = 0.9) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Split the given dataset to train/test, according to the given train_ratio.
    The split will be random, meaning that the train-data will be sampled randomly from the given dataset.
    See the format of the given sentences and pos_tags in the documentation of the function convert_to_numpy.
    :param sentences: The sentences.
    :param pos_tags: The pos-tags.
    :param train_ratio: A number between 0 and 1, portion of the dataset to be train-data.
    :return: The train-data and the test-data (in the same format as the given dataset).
    """
    n_samples = sentences.shape[0]
    n_train = int(train_ratio * n_samples)

    # Shuffle the data-set, to split to train/test randomly.
    permutation = np.random.permutation(n_samples)
    sentences = sentences[permutation]
    pos_tags = pos_tags[permutation]

    train_sentences = sentences[:n_train]
    train_pos_tags = pos_tags[:n_train]
    test_sentences = sentences[n_train:]
    test_pos_tags = pos_tags[n_train:]

    return train_sentences, train_pos_tags, test_sentences, test_pos_tags


class Baseline(object):
    """
    The baseline model.
    """

    def __init__(self, sentences: np.ndarray, pos_tags: np.ndarray):
        """
        The init function of the baseline Model.
        See the format of the given sentences and pos_tags
        in the documentation of the function convert_to_numpy.
        :param sentences: The sentences.
        :param pos_tags: The pos-tags.
        """
        self.sentences = sentences
        self.pos_tags = pos_tags

        # Set the sparse representation of the dataset, it may be used later.
        self.index2word, self.words_indices, self.words_count = np.unique(sentences,
                                                                          return_inverse=True,
                                                                          return_counts=True)
        self.index2pos, self.pos_indices, self.pos_count = np.unique(pos_tags,
                                                                     return_inverse=True,
                                                                     return_counts=True)

        # Define the sentences and pos-tags arrays as integers instead of strings.
        self.sentences_i = np.arange(len(self.index2word))[self.words_indices].reshape(self.sentences.shape)
        self.pos_tags_i = np.arange(len(self.index2pos))[self.pos_indices].reshape(self.pos_tags.shape)

        # Minus 1 because the empty-string is not really a word, it just indicates that there is no value there.
        self.words_size = len(self.index2word) - 1
        self.pos_size = len(self.index2pos) - 1

        self.word2i = {word: i for (i, word) in enumerate(self.index2word)}
        self.pos2i = {pos: i for (i, pos) in enumerate(self.index2pos)}

        self.pos_prob, self.emission_prob = self.maximum_likelihood_estimation()

    def maximum_likelihood_estimation(self) -> (np.ndarray, np.ndarray):
        """
        Calculate the Maximum Likelihood estimation of the
        multinomial and emission probabilities for the baseline model.
        :return: two numpy arrays - one is the multinomial distribution over the pos-tags,
                 and the other is the emission probabilities.
        """
        pos_tags_counts = self.pos_count[1:]  # Remove the first element - it corresponds to the zero pos-tag.

        # The PoS probabilities are the amount of time each PoS-tag occurred, divided by the total amount of PoS tags.
        pos_prob = pos_tags_counts / pos_tags_counts.sum()
        emission_prob = np.zeros(shape=(self.pos_size, self.words_size), dtype=np.float32)

        # Go over all pos-tags and for each one create a mask of the same shape as the training-set,
        # where the ij-th entry indicates whether the j-th word (in the i-th sentence) is tagged with that PoS tag.
        for i in range(1, self.pos_size + 1):
            pos_mask = (self.pos_tags_i == i)

            # Mask out the words in the training-set where the pos-tag is not i.
            words_at_pos = self.sentences_i * pos_mask

            # Get the set of words the appeared when the pos-tag i appeared,
            # with a count of how many time it happened.
            words_at_pos_unique, words_at_pos_counts = np.unique(words_at_pos, return_counts=True)

            # If the words that appeared at this pos-tag contains the empty-word,
            # it's best to remove it since it's not actually a word.
            if 0 in words_at_pos_unique:
                words_at_pos_unique = words_at_pos_unique[1:]
                words_at_pos_counts = words_at_pos_counts[1:]

            # The emission probability of the j-th word, given that the pos-tag is i,
            # is the amount of times the word j appeared with the pos-tag i,
            # divided by the total number of times the pos-tag i occurred.
            # Subtract 1 because the word2i and pos2i start at 1 (to enable 0 being like NaN).
            emission_prob[i - 1, words_at_pos_unique - 1] = words_at_pos_counts / words_at_pos_counts.sum()

        return pos_prob, emission_prob

    def MAP(self, sentences: np.ndarray) -> np.ndarray:
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: numpy array containing the pos tags.
        """
        n_samples, max_sentence_length = sentences.shape
        pos_tags = np.zeros(shape=(n_samples, max_sentence_length), dtype=self.index2pos.dtype)

        for i in range(n_samples):
            for j in range(max_sentence_length):
                word = sentences[i, j]

                # Since our sentences are padded with empty-strings in the end, we must check where to stop.
                # So if the word is the empty-word, we finished reading the sentence.
                if len(word) == 0:
                    break

                # If we encounter a word we did not see in the training-set, sample a pos-tag according to the
                # distribution we learned from the training-set (regardless of the word).
                if word not in self.word2i:
                    pos_tags[i, j] = np.random.choice(self.index2pos[1:], p=self.pos_prob)
                    continue

                # We subtract 1 from self.word2i[word] because the self.emission_prob's size corresponds to the original
                # number of words/pos-tags, excluding the empty-string (which is not really a word but a padding-word).
                # So the index of the word is the index in the self.index2word, but in the emission_prob array it's one
                # cell to the left.
                # We add 1 to the argmax, because the index of the maximal pos-tag is in the pos_prob array,
                # and from the same reason as above to get the index in the index2pos array 1 must be added.
                pos_tag_index = np.argmax(self.pos_prob * self.emission_prob[:, self.word2i[word] - 1]) + 1
                pos_tags[i, j] = self.index2pos[pos_tag_index]

        return pos_tags


class HMM(object):
    """
    The basic HMM_Model with multinomial transition functions.
    """

    def __init__(self, sentences: np.ndarray, pos_tags: np.ndarray):
        """
        The init function of the basic HMM Model.
        See the format of the given sentences and pos_tags
        in the documentation of the function convert_to_numpy.
        :param sentences: The sentences.
        :param pos_tags: The pos-tags.
        """
        self.sentences = sentences
        self.pos_tags = pos_tags

        # Set the sparse representation of the dataset, it may be used later.
        self.index2word, self.words_indices, self.words_count = np.unique(sentences,
                                                                          return_inverse=True,
                                                                          return_counts=True)
        self.index2pos, self.pos_indices, self.pos_count = np.unique(pos_tags,
                                                                     return_inverse=True,
                                                                     return_counts=True)

        # Define the sentences and pos-tags arrays as integers instead of strings.
        self.sentences_i = np.arange(len(self.index2word))[self.words_indices].reshape(self.sentences.shape)
        self.pos_tags_i = np.arange(len(self.index2pos))[self.pos_indices].reshape(self.pos_tags.shape)

        # Minus 1 because the empty-string is not really a word, it just indicates that there is no value there.
        self.words_size = len(self.index2word) - 1
        self.pos_size = len(self.index2pos) - 1

        self.word2i = {word: i for (i, word) in enumerate(self.index2word)}
        self.pos2i = {pos: i for (i, pos) in enumerate(self.index2pos)}

        self.transition_prob, self.emission_prob = self.maximum_likelihood_estimation()

    def maximum_likelihood_estimation(self) -> (np.ndarray, np.ndarray):
        """
        Calculate the Maximum Likelihood estimation of the
        transition and emission probabilities for the standard multinomial HMM.
        :return: two numpy arrays - one is the transition probabilities,
                 and the other is the emission probabilities.
        """
        # The PoS probabilities are the amount of time each PoS-tag occurred, divided by the total amount of PoS tags.
        transition_prob = np.zeros(shape=(self.pos_size, self.pos_size), dtype=np.float32)
        emission_prob = np.zeros(shape=(self.pos_size, self.words_size), dtype=np.float32)

        # Calculate the transition probabilities.
        for i in range(1, self.pos_size + 1):
            # If the pos-tag is the end-state, probabilities should be zeros.
            # Handle this case individually because otherwise we'll try to access the pos-tag that comes after
            # the END_STATE (and this is the padding empty-string).
            if self.index2pos[i] == END_STATE:
                continue

            # These are the indices where the pos-tag is i.
            # We look at the succeeding pos-tag in these sentences.
            row_indices, col_indices = np.where(self.pos_tags_i == i)

            n_pos = len(row_indices)
            assert n_pos > 0  # If the pos-tag did not appear in the sentences, something bad happened.

            # For each one of the succeeding pos-tags, calculate how many times it appeared (after the i-th pos-tag).
            succeeding_tags, succeeding_tags_counts = np.unique(self.pos_tags_i[row_indices, col_indices + 1],
                                                                return_counts=True)

            # Define the probabilities - the amount of times a particular succeeding-tag appeared, divided by the
            # total amount of times the i-th pos-tag appeared.
            # Subtract 1 because because the indices in the transition_prob array start from 0,
            # and the indices of the tags themselves start from 1
            # (as the 0-th pos-tag is the empty-string used for padding).
            transition_prob[i - 1, succeeding_tags - 1] = succeeding_tags_counts / n_pos

        # Calculate the emission probabilities.
        for i in range(1, self.pos_size + 1):
            # Create a mask of the same shape as the training-set, where the ij-th entry
            # indicates whether the j-th word (in the i-th sentence) is tagged with that PoS tag.
            pos_mask = (self.pos_tags_i == i)

            # Mask out the words in the training-set where the pos-tag is not i.
            words_at_pos = self.sentences_i * pos_mask

            # Get the set of words the appeared when the pos-tag i appeared,
            # with a count of how many time it happened.
            words_at_pos_unique, words_at_pos_counts = np.unique(words_at_pos, return_counts=True)

            # If the words that appeared at this pos-tag contains the empty-word,
            # it's best to remove it since it's not actually a word.
            if 0 in words_at_pos_unique:
                words_at_pos_unique = words_at_pos_unique[1:]
                words_at_pos_counts = words_at_pos_counts[1:]

            # The emission probability of the j-th word, given that the pos-tag is i,
            # is the amount of times the word j appeared with the pos-tag i,
            # divided by the total number of times the pos-tag i occurred.
            # Subtract 1 because the word2i and pos2i start at 1 (to enable 0 being like NaN).
            emission_prob[i - 1, words_at_pos_unique - 1] = words_at_pos_counts / words_at_pos_counts.sum()

        return transition_prob, emission_prob

    def sample(self, n: int) -> (list, list):
        """
        Sample n sequences of words from the HMM.
        :return: Two lists containing the sentences and the pos-tags.
        """
        sentences = [[START_WORD] for _ in range(n)]
        pos_tags = [[START_STATE] for _ in range(n)]

        for i in range(n):
            sentence = sentences[i]
            tags = pos_tags[i]

            prev_pos_tag = tags[0]
            while prev_pos_tag != END_STATE:
                transition_probabilities = self.transition_prob[self.pos2i[prev_pos_tag] - 1]
                curr_tag = np.random.choice(self.index2pos[1:], p=transition_probabilities)
                emission_probabilities = self.emission_prob[self.pos2i[curr_tag] - 1]
                curr_word = np.random.choice(self.index2word[1:], p=emission_probabilities)

                sentence.append(curr_word)
                tags.append(curr_tag)

                prev_pos_tag = curr_tag

        return sentences, pos_tags

    def viterbi(self, sentences: np.ndarray) -> np.ndarray:
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: numpy array containing word sequences (sentences).
        :return: numpy array containing the pos-tags.
        """
        n_sentences, max_sentence_length = sentences.shape
        pos_predictions = np.zeros(shape=(n_sentences, max_sentence_length), dtype=self.index2pos.dtype)

        # Define the log transition array, while avoiding taking np.log of 0
        # (which results in the desired output -inf, bu cause an annoying warning).
        log_transition = np.full_like(self.transition_prob, fill_value=-np.inf)
        positive_transition = (self.transition_prob > 0)
        log_transition[positive_transition] = np.log(self.transition_prob[positive_transition])

        for i in range(n_sentences):
            sentence_mask = (sentences[i] != '')
            sentence = sentences[i, sentence_mask]
            n = len(sentence)

            # Define the two tables to be filled using the dynamic-programming algorithm.
            # max_log_prob is the maximal log-probability among all possible sequences
            # of pos-tags up to time t, that end in PoS i.
            # back_pointers is an array containing the arg-maxes, to enable extracting the pos-tags
            # that led to the maximal probability.
            max_log_prob = np.full(shape=(n, self.pos_size), fill_value=-np.inf, dtype=np.float32)
            back_pointers = np.zeros(shape=(n, self.pos_size), dtype=np.int)

            # Initialize the first row to be zero, which is like initializing it to 1
            # (if using probabilities and not log-probabilities).
            max_log_prob[0, :] = 0

            # Start from the second row, and fill the tables row-by-row.
            for l in range(1, n):
                word = sentence[l]

                # If the word was not seen in the training-phase, treat it as a rare-word.
                if word not in self.word2i:
                    word = RARE_WORD

                word_index = self.word2i[word] - 1

                # Define the log emission array, while avoiding taking np.log of 0
                # (which results in the desired output -inf, bu cause an annoying warning).
                log_emission = np.full_like(self.emission_prob[:, word_index], fill_value=-np.inf)
                positive_emission = (self.emission_prob[:, word_index] > 0)
                log_emission[positive_emission] = np.log(self.emission_prob[positive_emission, word_index])

                # Define the 2D array to take the maximal value and the arg-max from.
                # The ij-th entry in the log_transition matrix is the log-probability of the
                # transition from the PoS i to the PoS j.
                # Adding the log_emission as a row-vector, implies that rows that correspond
                # to PoS with 0 probability to emit the word will be all -inf.
                # In general, We add to each row i the probability of the PoS i to emit the word.
                # Adding the max_log_prob previous row as a column-vector, means adding to each entry the
                # maximal log-probability of previous sequences of PoS tags that end in i.
                arr = max_log_prob[l - 1, :].reshape(-1, 1) + log_transition + log_emission.reshape(1, -1)

                # Taking maximum along each column means finding the maximal sequence of previous PoS tags
                # ending in PoS i, plus the log transition from PoS i to PoS j, plus the log-emission of the current
                # word given the j-th PoS.
                back_pointers[l, :] = np.argmax(arr, axis=0)
                max_log_prob[l, :] = np.max(arr, axis=0)

                # # The code below does the same thing, but not vectorized.
                # # Maybe it can explain some more.
                # temp_back_pointers = np.copy(back_pointers[l, :])
                # temp_max_log_prob = np.copy(max_log_prob[l, :])
                #
                # for j in range(self.pos_size):
                #     log_transition = np.full_like(self.transition_prob[:, j], fill_value=-np.inf)
                #     positive_transition = (self.transition_prob[:, j] > 0)
                #     log_transition[positive_transition] = np.log(self.transition_prob[positive_transition, j])
                #
                #     emission = self.emission_prob[j, word_index]
                #     log_emission = np.log(emission) if emission > 0 else -np.inf
                #
                #     arr = max_log_prob[l - 1, :] + log_transition + log_emission
                #     back_pointers[l, j] = np.argmax(arr)
                #     max_log_prob[l, j] = arr[back_pointers[l, j]]
                #
                # assert np.array_equal(back_pointers[l, :], temp_back_pointers)
                # assert np.array_equal(max_log_prob[l, :], temp_max_log_prob)
            
            # Follow the back-pointers (from the end to the beginning) to get the PoS tags of the sentences.
            pos_prediction = np.empty_like(sentence, dtype=self.index2pos.dtype)

            pos_prediction[n-1] = END_STATE
            for l in range(n-2, 0, -1):
                pos_prediction[l] = self.index2pos[1 + back_pointers[l + 1, self.pos2i[pos_prediction[l+1]] - 1]]
            pos_prediction[0] = START_STATE

            pos_predictions[i, sentence_mask] = pos_prediction

        return pos_predictions


class MEMM(object):
    """
    The base Maximum Entropy Markov Model with log-linear transition functions.
    """

    def __init__(self, sentences: np.ndarray, pos_tags: np.ndarray, phi, mapping_dimension: int):
        """
        The init function of the MEMM.
        See the format of the given sentences and pos_tags
        in the documentation of the function convert_to_numpy.
        :param: sentences: The sentences.
        :param: pos_tags: The pos-tags.
        :param: phi: the feature mapping function, which accepts a numpy array containing 
                    multiple triplets of 2 PoS tags and a word, 
                    and returns a numpy array containing indices that have "1" in the binary feature vector.
        :param: mapping_dimension the dimension of the features space 
                (i.e. the dimension of each vector the mapping function outputs).
            
        """
        self.sentences = sentences
        self.pos_tags = pos_tags

        # Set the sparse representation of the dataset, it may be used later.
        self.index2word, self.words_indices, self.words_count = np.unique(sentences,
                                                                          return_inverse=True,
                                                                          return_counts=True)
        self.index2pos, self.pos_indices, self.pos_count = np.unique(pos_tags,
                                                                     return_inverse=True,
                                                                     return_counts=True)

        # Define the sentences and pos-tags arrays as integers instead of strings.
        self.sentences_i = np.arange(len(self.index2word))[self.words_indices].reshape(self.sentences.shape)
        self.pos_tags_i = np.arange(len(self.index2pos))[self.pos_indices].reshape(self.pos_tags.shape)

        # Minus 1 because the empty-string is not really a word, it just indicated that there is no value there.
        self.words_size = len(self.index2word) - 1
        self.pos_size = len(self.index2pos) - 1

        self.word2i = {word: i for (i, word) in enumerate(self.index2word)}
        self.pos2i = {pos: i for (i, pos) in enumerate(self.index2pos)}

        self.phi = phi
        self.mapping_dimension = mapping_dimension
        self.w = np.zeros(shape=self.mapping_dimension, dtype=np.float32)

        # TODO maybe initialize as normal?
        # self.w = np.random.normal(loc=0, scale=0.1, size=self.mapping_dimension).astype(np.float32)

        self.perceptron()

    def viterbi(self, sentences: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: Sentences to predict.
        :param w: a vector of weights, if not given this is defined as the model self.w
        :return: iterable sequence of POS tag sequences.
        """
        if w is None:
            w = self.w

        n_sentences, max_sentence_length = sentences.shape
        pos_predictions = np.zeros(shape=(n_sentences, max_sentence_length), dtype=self.index2pos.dtype)

        for i in range(n_sentences):
            sentence_mask = (sentences[i] != '')
            sentence = sentences[i, sentence_mask]
            n = len(sentence)

            # Define the two tables to be filled using the dynamic-programming algorithm.
            # max_log_prob is the maximal log-probability among all possible sequences
            # of pos-tags up to time t, that end in PoS i.
            # back_pointers is an array containing the arg-maxes, to enable extracting the pos-tags
            # that led to the maximal probability.
            max_log_prob = np.full(shape=(n, self.pos_size), fill_value=-np.inf, dtype=np.float32)
            back_pointers = np.zeros(shape=(n, self.pos_size), dtype=np.int)

            # Calculate log Z(pos_tag, word) for each pos_tag and word in the sentence.
            # This will save time later, by avoiding repeated calculations.
            z = logsumexp(
                np.sum(w[self.phi(self.index2pos[1:], self.index2pos[1:], sentence)], axis=-1),
                axis=1
            )

            # # This is the non-vectorized code, used for debugging purposes.
            # z__ = np.empty(shape=(self.pos_size, n), dtype=np.float32)
            # for l in range(n):
            #     for j in range(self.pos_size):
            #         prev_pos = self.index2pos[j + 1]
            #         z__[j, l] = logsumexp(np.sum(w[self.phi(prev_pos, self.index2pos[1:], sentence[l])], axis=1))
            #
            # z_ = np.empty(shape=(self.pos_size, n), dtype=np.float32)
            # for l in range(n):
            #     z_[:, l] = logsumexp(
            #         np.sum(w[self.phi(self.index2pos[1:], self.index2pos[1:], sentence[l])], axis=-1),
            #         axis=1
            #     )
            # assert np.allclose(z, z_, atol=1e-07)
            # assert np.allclose(z_, z__, atol=1e-07)

            # Initialize the first row to be zero, which is like initializing it to 1
            # (if using probabilities and not log-probabilities).
            max_log_prob[0, :] = 0

            # Start from the second row, and fill the tables row-by-row.
            for l in range(1, n):
                word = sentence[l]

                # If the word was not seen in the training-phase, treat it as a rare-word.
                if word not in self.word2i:
                    word = RARE_WORD
                # arr = max_log_prob[l - 1, :].reshape(-1, 1) + log_transition + log_emission.reshape(1, -1)

                arr = (max_log_prob[l - 1, :].reshape(-1, 1) +
                       np.sum(w[self.phi(self.index2pos[1:], self.index2pos[1:], word)], axis=-1) -
                       z[:, l].reshape(-1, 1))

                max_log_prob[l, :] = np.max(arr, axis=0)
                back_pointers[l, :] = np.argmax(arr, axis=0)

                # # This is the non-vectorized code, used for debugging purposes.
                # max_log_prob_ = np.empty_like(max_log_prob[l, :])
                # back_pointers_ = np.empty_like(back_pointers[l, :])
                # for j in range(self.pos_size):
                #     curr_pos = self.index2pos[j + 1]
                #     arr_ = (max_log_prob[l - 1, :] +
                #             np.sum(w[self.phi(self.index2pos[1:], curr_pos, word)], axis=-1) -
                #             z[:, l])
                #
                #     # # This is the non-vectorized code that fills arr element-by-element.
                #     # arr2 = np.empty(shape=self.pos_size, dtype=np.float32)
                #     # for k in range(self.pos_size):
                #     #     prev_pos = self.index2pos[k + 1]
                #     #     arr2[k] = (max_log_prob[l - 1, k] +
                #     #                np.sum(w[self.phi(prev_pos, curr_pos, word)]) -
                #     #                z[k, l])
                #     # assert np.array_equal(arr, arr2)
                #     # max_log_prob[l, j] = np.max(arr)
                #     # back_pointers[l, j] = np.argmax(arr)
                #
                #     max_log_prob_[j] = np.max(arr_)
                #     back_pointers_[j] = np.argmax(arr_)
                #
                # assert np.allclose(max_log_prob[l, :], max_log_prob_, atol=1e-07)
                # assert np.allclose(back_pointers[l, :], back_pointers_, atol=1e-07)

            # Follow the back-pointers (from the end to the beginning) to get the PoS tags of the sentences.
            pos_prediction = np.zeros_like(sentence, dtype=self.index2pos.dtype)

            pos_prediction[n-1] = END_STATE
            for l in range(n-2, 0, -1):
                pos_prediction[l] = self.index2pos[1 + back_pointers[l + 1, self.pos2i[pos_prediction[l+1]] - 1]]
            pos_prediction[0] = START_STATE

            pos_predictions[i, sentence_mask] = pos_prediction

        return pos_predictions

    def perceptron(self, eta: float = 0.1, epochs: int = 1, accuracy_batch_size: int = 64):
        """
        learn the weight vector of a log-linear model according to the training set.
        :param eta: the learning rate for the perceptron algorithm.
        :param epochs: the amount of times to go over the entire training data (default is 1).
        :param accuracy_batch_size: amount of samples to draw from the
                                    train-set to calculate accuracy
        """
        print_iterations = 100
        n_samples = len(self.sentences)

        w = np.copy(self.w)

        # In each epoch, create a random permutation of the sentences and go over them sequentially..
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            for i in permutation:
                sentence_mask = (self.sentences[i] != '')
                sentence = self.sentences[i, sentence_mask]
                pos_tag = self.pos_tags[i, sentence_mask]
                n = len(sentence)

                iteration_number = np.where(permutation == i)[0][0] + 1

                if iteration_number % print_iterations == 0 or iteration_number == 1:
                    train_indices = np.random.choice(len(self.sentences), size=accuracy_batch_size)
                    train_pos_predictions = self.viterbi(self.sentences[train_indices], w)
                    train_accuracy = evaluate_model(self.pos_tags[train_indices], train_pos_predictions)

                    print(f'Perceptron iteration #{iteration_number:6d},\t'
                          f'train-accuracy = {100 * train_accuracy:.2f}%')
                    # if test_sentences is not None and test_pos_tags is not None:
                    #     test_indices = np.random.choice(len(test_sentences), size=accuracy_batch_size)
                    #     test_mask = test_sentences[test_indices] != ''
                    #     test_pos_predictions = self.viterbi(test_sentences[test_indices], w)
                    #     test_accuracy = np.mean(test_pos_predictions[test_mask] ==
                    #                             test_pos_tags[test_indices][test_mask])
                    #
                    #     s += f'test-accuracy = #{test_accuracy:.2f},\t'

                # Calculate the most likely sequence of PoS tags,
                # given the current parameters of the model and the current sentence,
                pos_predict = self.viterbi(sentence.reshape(1, -1), w).flatten()

                # Update the weight-vector in the corresponding index.
                positive_indices = np.array(list(), dtype=int)
                negative_indices = np.array(list(), dtype=int)

                for j in range(1, n):
                    positive_indices = np.concatenate((positive_indices,
                                                       self.phi(pos_tag[j - 1],
                                                                pos_tag[j],
                                                                sentence[j])))
                    negative_indices = np.concatenate((negative_indices,
                                                       self.phi(pos_predict[j - 1],
                                                                pos_predict[j],
                                                                sentence[j])))

                positive_indices_unique, positive_indices_counts = np.unique(positive_indices, return_counts=True)
                negative_indices_unique, negative_indices_counts = np.unique(negative_indices, return_counts=True)

                w[positive_indices_unique] += eta * positive_indices_counts
                w[negative_indices_unique] -= eta * negative_indices_counts

        self.w = w


def get_mapping(index2pos: np.ndarray, index2word: np.ndarray):
    """
    Get the mapping function.
    It returns the indices of 1 in the binary vector for each one of the given triplets.
    :param index2pos: index to PoS tag
    :param index2word: index to word
    :return: The mapping function, returning indices of one in the binary
             vector for each one of the given triplets.
    """
    n_pos = len(index2pos)
    n_words = len(index2word)
    mapping_dimension = n_pos ** 2 + n_pos * n_words

    def mapping_function(prev_pos_tags: np.ndarray,
                         curr_pos_tags: np.ndarray,
                         words: np.ndarray) -> np.ndarray:
        """
        Given previous pos-tag, current pos-tag, and the current word, returns the indices
        corresponding to 1 in the binary representation vector.
        The arguments may be multiple values, in this case all possible combinations of
        previous pos-tag, current pos-tag and current word will be calculated.
        :param prev_pos_tags: (possibly many) previous pos-tags.
        :param curr_pos_tags: (possibly many) current pos-tags.
        :param words: (possibly many) current words.
        :return: a numpy array of shape (#prev_pos_tags, #curr_pos_tags, #words, 2),
                 where each pair in the index i,j,k is the indices of 1 in the binary
                 representation vector of the corresponding prev&curr pos-tag, and curr word.
        """
        # Get the indices of the given pos_tags (each on is possibly many pos-tags),
        # and the indices of the words (which also can be many words).
        prev_pos_tags_indices = np.searchsorted(index2pos, prev_pos_tags)
        curr_pos_tags_indices = np.searchsorted(index2pos, curr_pos_tags)
        words_indices = np.searchsorted(index2word, words)

        # The output shape of the combinations array is (#prev_pos_tags, #curr_pos_tags, #words, 3)
        output_shape = prev_pos_tags_indices.shape + curr_pos_tags_indices.shape + words_indices.shape + (3,)

        # Create all possible combinations - each triplet in the index i,j,k is the indices
        # of the corresponding prev&curr pos-tag, and curr word.
        combinations = np.array(list(product(np.atleast_1d(prev_pos_tags_indices),
                                             np.atleast_1d(curr_pos_tags_indices),
                                             np.atleast_1d(words_indices)))).reshape(output_shape)

        # Create the transition indices - the index of the position corresponding to the
        # transition from the previous pos-tag and the current pos-tag.
        transition_indices = n_pos * combinations[..., 0] + combinations[..., 1]

        # Create the emission indices - the index of the position corresponding to the
        # emission of the current word at the current pos-tag.
        emission_indices = n_pos ** 2 + combinations[..., 1] * n_words + combinations[..., 2]

        # Stack both of them to shape (#prev_pos_tags, #curr_pos_tags, #words, 2)
        return np.stack((transition_indices, emission_indices), axis=-1)

    return mapping_function, mapping_dimension


def evaluate_model(pos_tags: np.ndarray, pos_predictions: np.ndarray) -> float:
    """
    Calculate the accuracy, given two numpy arrays containing PoS tags and PoS predictions.
    The calculation ignores empty-string cells (which are indicators for no-item, they are used for padding).
    :param pos_tags: The PoS tags
    :param pos_predictions: The PoS predictions
    :return: The accuracy.
    """
    n_correct = np.sum((~np.isin(pos_tags, ['', START_STATE, END_STATE])) & (pos_predictions == pos_tags))
    n_words = np.sum(~np.isin(pos_tags, ['', START_STATE, END_STATE]))
    accuracy = n_correct / n_words

    return accuracy


def sample_and_print(model: HMM, amount_to_sample: int = 16):
    """
    Draw samples and print them.
    :param model: a HMM model to sample from
    :param amount_to_sample: how many samples to draw.
    """
    sampled_sentences, sampled_pos_tags = model.sample(amount_to_sample)
    print('The sampled sentences are:')
    for i in range(len(sampled_sentences)):
        sentence_str = ''
        tags_str = ''
        sentence = sampled_sentences[i]
        tags = sampled_pos_tags[i]
        for j in range(len(sentence)):
            word = sentence[j]
            tag = tags[j]
            max_len = max(len(tag), len(word))

            sentence_str += f'{word:^{max_len + 2}}'
            tags_str += f'{tag:^{max_len + 2}}'

        print('\t' + sentence_str)
        print('\t' + tags_str)


def main(model_name: str, training_data_portion_to_use: float = 1):
    """
    The main function.
    Build the model, train it and evaluate on the test set.
    :param model_name: the name of the model to work with.
                       Must be one of 'baseline', 'hmm' or 'memm
    :param training_data_portion_to_use: How much of the training-data to use.
                                         By default, uses all training-data, but it can be used to accelerate
                                         the running-time and use for example only 10% of the training-data.
    """
    dataset, words, total_pos_tags = get_data()
    sentences, pos_tags = convert_to_numpy(dataset, rare_threshold=5)

    train_sentences, train_pos_tags, test_sentences, test_pos_tags = split_train_test(sentences,
                                                                                      pos_tags,
                                                                                      train_ratio=0.9)

    # Use only a portion of the training-data, as defined
    # by the given argument training_data_portion_to_use.
    train_permutation = np.random.permutation(len(train_sentences))
    n_samples_train = int(training_data_portion_to_use * len(train_sentences))
    train_sentences = train_sentences[train_permutation][:n_samples_train]
    train_pos_tags = train_pos_tags[train_permutation][:n_samples_train]

    if model_name == 'baseline':
        model = Baseline(train_sentences, train_pos_tags)
        # train_pos_predictions = model.MAP(train_sentences)
        test_pos_predictions = model.MAP(test_sentences)
    elif model_name == 'hmm':
        model = HMM(train_sentences, train_pos_tags)
        sample_and_print(model)
        # train_pos_predictions = model.viterbi(train_sentences)
        test_pos_predictions = model.viterbi(test_sentences)
    elif model_name == 'memm':
        phi, mapping_dimension = get_mapping(index2pos=np.unique(pos_tags)[1:],
                                             index2word=np.unique(sentences)[1:])
        model = MEMM(train_sentences, train_pos_tags, phi, mapping_dimension)
        # train_pos_predictions = model.viterbi(train_sentences)
        test_pos_predictions = model.viterbi(test_sentences)
    else:
        raise ValueError("ERROR: Unrecognized model-name!")

    # train_accuracy = evaluate_model(train_pos_tags, train_pos_predictions)
    test_accuracy = evaluate_model(test_pos_tags, test_pos_predictions)

    # print(f'The accuracy of the {model_name} model on the train-data is {100 * train_accuracy:.2f}%.')
    print(f'The accuracy of the {model_name} model on the test-data  is {100 * test_accuracy:.2f}%.')


if __name__ == '__main__':
    main(model_name='hmm', training_data_portion_to_use=0.1)
