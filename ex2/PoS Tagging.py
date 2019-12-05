import pickle
import os
import random
import numpy as np
from scipy.special import logsumexp

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'


def get_data(data_path='PoS_data.pickle',
             words_path='all_words.pickle',
             pos_path='all_PoS.pickle'):
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


def convert_to_numpy(dataset: list, rare_threshold: int = 0) -> tuple:
    """
    Create a sparse representation of the data, using numpy's unique fucntion.
    See a simple example on how to reconstruct the dataset using the return values
    of this function in the comments explaining the code.

    This function also handles the rare words by replacing them with the RARE_WORD symbol.
    Furthermore, it adds START and END symbols for the sentences and the pos-tags.

    :param dataset: A list of word sequences.
    :param rare_threshold: An integer representing the threshold for a word to be regarded as a "rare word".
                           By default it is 0, meaning that nothing will be done with the rare-words.
    :return: index2word, words_indices, words_count,
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
             And the same holds for the pos-taggings.
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

    # Create the sparse representation of the data, using numpy's unique fucntion.
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


def split_train_test(sentences, sentences_tags, train_ratio=0.9):
    """
    Split the given dataset to train/test, according to the given train_ratio.
    The split will be random, meaning that the train-data will be sampled randomly from the given dataset.
    :param dataset: A list containing tuples, where each tuple is a single sample.
                    - The first element is the PoS tagging of the sentence.
                    - The second element sentence itself.
    :param train_ratio: A number between 0 and 1, portion of the dataset to be train-data.
    :return: A tuple containing the train-data and the test-data (in the same format as the given dataset).
    """
    n_samples = sentences.shape[0]
    n_train = int(train_ratio * n_samples)

    # Shuffle the data-set, to split to train/test randomly.
    permutation = np.random.permutation(n_samples)
    sentences = sentences[permutation]
    sentences_tags = sentences_tags[permutation]

    train_sentences = sentences[:n_train]
    train_sentences_tags = sentences_tags[:n_train]
    test_sentences = sentences[n_train:]
    test_sentences_tags = sentences_tags[n_train:]

    return train_sentences, train_sentences_tags, test_sentences, test_sentences_tags


class Baseline(object):
    """
    The baseline model.
    """

    def __init__(self, sentences, pos_tags):
        """
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
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

        self.pos_prob, self.emission_prob = self.maximum_likelihood_estimation()

    def maximum_likelihood_estimation(self):
        """
        Calculate the Maximum Likelihood estimation of the
        multinomial and emission probabilities for the baseline model.
        """
        pos_tags_counts = self.pos_count[1:]  # Remove the first element - it corresponds to the zero pos-tag.

        # The PoS probabilities are the amount of time each PoS-tag occurred, divided by the total amount of PoS tags.
        pos_prob = pos_tags_counts / pos_tags_counts.sum()
        emission_prob = np.zeros(shape=(self.pos_size, self.words_size), dtype=np.float32)

        # Go over all pos-tags and for each one create a mask of same shape as the training-set,
        # where the ij-th entry indicates whether the j-th word (in the i-th sentence) is
        for i in range(1, self.pos_size + 1):
            pos_mask = (self.pos_tags_i == i)
            # Mask out the words in the training-set where the pos-tag is not i.
            words_at_pos = self.sentences_i * pos_mask

            # Get the set of words the appeared when the pos-tag i appeared,
            # with a count of how many time it happened.
            words_at_pos_unique, words_at_pos_counts = np.unique(words_at_pos, return_counts=True)

            # The first element in each array corresponds to the 0 word,
            # which is not really a word and more like a NaN.
            words_at_pos_unique = words_at_pos_unique[1:]
            words_at_pos_counts = words_at_pos_counts[1:]

            # The emission probability of the j-th word, given that the pos-tag is i,
            # is the amount of times the word j appeared with the pos-tag i,
            # divided by the total number of times the pos-tag i occurred.
            # Subtract 1 because the word2i and pos2i start at 1 (to enable 0 being like NaN).
            emission_prob[i - 1, words_at_pos_unique - 1] = words_at_pos_counts / words_at_pos_counts.sum()

        return pos_prob, emission_prob

    def MAP(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
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

    def __init__(self, sentences, pos_tags):
        """
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
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

        self.transition_prob, self.emission_prob = self.maximum_likelihood_estimation()

    def maximum_likelihood_estimation(self):
        """
        Calculate the Maximum Likelihood estimation of the
        transition and emission probabilities for the standard multinomial HMM.
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
            # Create a mask of same shape as the training-set,
            # where the ij-th entry indicates whether the j-th word (in the i-th sentence) is
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

    def sample(self, n):
        """
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
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

    def viterbi(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
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
                # Adding the max_log_prob previous row as a column-vector, means adding toe each entry the
                # maximal log-probability of previous sequences of PoS tags that end in i.
                arr = max_log_prob[l - 1, :].reshape(-1, 1) + log_transition + log_emission.reshape(1, -1)

                # Taking maximum among each column means finding the maximal sequence of previous PoS tags
                # ending in PoS i, plus the log transition from PoS i to PoS j, plus the log-emission of the current
                # word given the j-th PoS.
                back_pointers[l, :] = np.argmax(arr, axis=0)
                max_log_prob[l, :] = np.max(arr, axis=0)

                # The code below does the same thing, but not vectorized.
                # Maybe it can explain some more.

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

    def __init__(self, sentences, pos_tags, phi, mapping_dimension):
        """
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
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
        self.phi_vec = get_mapping_vec_func(phi, mapping_dimension)
        self.w = np.random.normal(loc=0, scale=0.1, size=self.mapping_dimension).astype(np.float32)

        self.perceptron()

    def viterbi(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        """
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

            # Initialize the first row to be zero, which is like initializing it to 1
            # (if using probabilities and not log-probabilities).
            max_log_prob[0, :] = 0

            # Calculate log Z(pos_tag, word) for each pos_tag and word in the sentence.
            # This will save time later, by avoiding repeated calculations.
            z = np.empty(shape=(self.pos_size, n), dtype=np.float32)
            for l in range(1, n):
                for j in range(self.pos_size):
                    prev_pos = self.index2pos[j + 1]
                    z[j, l] = logsumexp(np.sum(self.w[self.phi(prev_pos, self.index2pos[1:], sentence[l])], axis=1))

            # Start from the second row, and fill the tables row-by-row.
            for l in range(1, n):
                word = sentence[l]

                # If the word was not seen in the training-phase, treat it as a rare-word.
                if word not in self.word2i:
                    word = RARE_WORD

                # TODO vectorize, by modifying phi
                for j in range(self.pos_size):
                    curr_pos = self.index2pos[j + 1]
                    arr = (max_log_prob[l - 1, :] +
                           np.sum(self.w[self.phi(self.index2pos[1:], curr_pos, word)], axis=1) -
                           z[:, l])

                    # # This is the non-vectorized code that fills arr element-by-element.
                    # arr2 = np.empty(shape=self.pos_size, dtype=np.float32)
                    # for k in range(self.pos_size):
                    #     prev_pos = self.index2pos[k + 1]
                    #     arr2[k] = (max_log_prob[l - 1, k] +
                    #                np.sum(self.w[self.phi(prev_pos, curr_pos, word)]) -
                    #                z[k, l])
                    # assert np.array_equal(arr, arr2)
                    max_log_prob[l, j] = np.max(arr)
                    back_pointers[l, j] = np.argmax(arr)

            pos_prediction = np.empty_like(sentence, dtype=self.index2pos.dtype)

            pos_prediction[n-1] = END_STATE
            for l in range(n-2, 0, -1):
                pos_prediction[l] = self.index2pos[1 + back_pointers[l + 1, self.pos2i[pos_prediction[l+1]] - 1]]
            pos_prediction[0] = START_STATE

            pos_predictions[i, sentence_mask] = pos_prediction

        return pos_predictions

    def perceptron(self, eta=0.1, epochs=1):
        """
        learn the weight vector of a log-linear model according to the training set.
        :param training_set: iterable sequence of sentences and their parts-of-speech.
        :param initial_model: an initial MEMM object, containing among other things
                the phi feature mapping function.
        :param w0: an initial weights vector.
        :param eta: the learning rate for the perceptron algorithm.
        :param epochs: the amount of times to go over the entire training data (default is 1).
        :return: w, the learned weights vector for the MEMM.
        """
        n_samples = len(self.sentences)

        # Define an array of weight-vectors to be filled during the perceptron algorithm.
        ws = np.empty(shape=(epochs * n_samples, self.mapping_dimension), dtype=np.float32)

        # The first weight-vector is the current one (initialized in the constructor with zeros / normal distribution).
        ws[0] = self.w

        # In each epoch, create a random permutation of the sentences and go over them sequentially..
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            for i in permutation:
                sentence_mask = (self.sentences[i] != '')
                sentence = self.sentences[i, sentence_mask]
                pos_tag = self.pos_tags[i, sentence_mask]
                n = len(sentence)

                # Calculate the most likely sequence of PoS tags,
                # given the current parameters of the model and the current sentence,
                pos_predictions = self.viterbi(sentence.reshape(1, -1)).flatten()

                s = 0
                for j in range(1, n):
                    s += (self.phi_vec(pos_tag[j], pos_tag[j - 1], sentence[j]) -
                          self.phi_vec(pos_predictions[j], pos_predictions[j - 1], sentence[j]))

                # Update the weight-vector in the corresponding index.
                w_index = epoch * n_samples + i
                ws[w_index] = ws[w_index - 1] + eta * s

        self.w += ws.mean()


def get_mapping(index2pos, index2word):
    n_pos = len(index2pos)
    n_words = len(index2word)
    mapping_dimension = 2 * n_pos + n_words

    def mapping_function(pos_tags1, pos_tags2, words):
        # Get the indices of the given two pos_tags (each on is possibly many pos-tags),
        # and the indices of the words (which also can be many words).
        pos_tags1_indices = np.searchsorted(index2pos, pos_tags1)
        pos_tags2_indices = np.searchsorted(index2pos, pos_tags2)
        words_indices = np.searchsorted(index2word, words)

        # Define the maximal-shape, which is the shape which has the maximal number of dimensions.
        max_shape = max(pos_tags1_indices.shape, pos_tags2_indices.shape, words_indices.shape, key=len)

        # If the maximal-shape is 0-dimensional, it means that all given arguments were scalars.
        # In this case, define the maximal-shape to be a tuple holding a single element 1,
        # which means that each scalar is treated as it is an array containing a single element.
        if len(max_shape) == 0:
            max_shape = (1,)

        # No support for multi-dimensional arrays
        assert len(max_shape) == 1

        # In case one of the given arguments contains multiple values (let's say N),
        # the returned array will be of shape (N, 3), so each row will have the corresponding indices in the mapping.
        # If all arguments were scalars, squeeze to return an array of shape (3,).
        return np.stack((np.broadcast_to(pos_tags1_indices, max_shape),
                         np.broadcast_to(n_pos + pos_tags2_indices, max_shape),
                         np.broadcast_to(2 * n_pos + words_indices, max_shape)), axis=1).squeeze()

    return mapping_function, mapping_dimension


def get_mapping_vec_func(phi, mapping_dimension):
    def phi_vec(pos_tags1, pos_tags2, words):
        # Define the maximal-shape, which is the shape which has the maximal number of dimensions.
        max_shape = max(pos_tags1.shape, pos_tags2.shape, words.shape, key=len)

        arr = np.zeros(shape=max_shape + (mapping_dimension,), dtype=np.float32)
        indices = phi(pos_tags1, pos_tags2, words)
        arr[indices] = 1
        return arr

    return phi_vec


def evaluate_model(pos_tags, pos_predictions):
    n_correct = np.sum((pos_tags != '') & (pos_predictions == pos_tags))
    n_words = np.count_nonzero(pos_tags)
    accuracy = n_correct / n_words

    return accuracy


def sample_and_print(model, amount_to_sample=4):
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


def main(model_name):
    dataset, words, total_pos_tags = get_data()
    dataset = random.sample(dataset, k=1000)  # TODO remove
    sentences, pos_tags = convert_to_numpy(dataset, rare_threshold=5)

    train_sentences, train_pos_tags, test_sentences, test_pos_tags = split_train_test(sentences,
                                                                                      pos_tags,
                                                                                      train_ratio=0.9)
    # Set the sparse representation of the dataset, it may be used later.
    index2word, words_indices, words_count = np.unique(sentences, return_inverse=True, return_counts=True)
    index2pos, pos_indices, pos_count = np.unique(pos_tags, return_inverse=True, return_counts=True)

    # word2i = {word: i for (i, word) in enumerate(index2word)}
    # pos2i = {pos: i for (i, pos) in enumerate(index2pos)}

    if model_name == 'baseline':
        model = Baseline(train_sentences, train_pos_tags)
        train_pos_predictions = model.MAP(train_sentences)
        test_pos_predictions = model.MAP(test_sentences)
    elif model_name == 'hmm':
        model = HMM(train_sentences, train_pos_tags)
        sample_and_print(model)
        # train_pos_predictions = model.viterbi(train_sentences)
        test_pos_predictions = model.viterbi(test_sentences)
    elif model_name == 'memm':
        phi, mapping_dimension = get_mapping(index2pos[1:], index2word[1:])
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
    main('memm')
