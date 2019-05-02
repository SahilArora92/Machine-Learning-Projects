import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""

	model = None

	# pi= number of sequences starting with s/ number of sequences
	S = len(tags)
	total_lines = len(train_data)
	state_dict = {tags[i]: i for i in range(len(tags))}
	pi = np.zeros([S])
	# states_count = np.zeros([S])
	A = np.zeros([S, S])

	outcome_tag_dict = {}
	outcome_dict = {}
	unique_word_pos = -1
	for line in train_data:

		pi[state_dict[line.tags[0]]] += 1

		for curr_ind in range(line.length):
			# states_count[state_dict[line.tags[curr_ind]]] += 1

			if not (line.tags[curr_ind], line.words[curr_ind]) in outcome_tag_dict:
				outcome_tag_dict[(line.tags[curr_ind], line.words[curr_ind])] = 1
			else:
				outcome_tag_dict[(line.tags[curr_ind], line.words[curr_ind])] += 1

			if not line.words[curr_ind] in outcome_dict:
				unique_word_pos += 1
				outcome_dict[line.words[curr_ind]] = unique_word_pos

			if curr_ind == line.length - 1:
				continue
			# states_count[state_dict[line.tags[curr_ind + 1]]] += 1
			A[state_dict[line.tags[curr_ind]]][state_dict[line.tags[curr_ind + 1]]] += 1
	B = np.zeros((S, len(outcome_dict)))
	for key, item in outcome_tag_dict.items():
		B[state_dict[key[0]]][outcome_dict[key[1]]] = item

	# remove unnecessary states
	indxs = find_removal_index(A)
	if indxs:
		pi = np.delete(pi, indxs)
		state_dict = switch_key_value_dict(state_dict)
		for ind in indxs:
			del state_dict[ind]
		state_dict = switch_key_value_dict(state_dict)
	# remove rows containing zero
	B = B[~np.all(B == 0, axis=1)]
	A = A[~np.all(A == 0, axis=1)]
	A = A.T[~np.all(A == 0, axis=0)]
	A = A.T

	# compute probabilities and normalize
	B = np.nan_to_num(B / np.sum(B, axis=0))
	pi = pi / total_lines

	A = np.nan_to_num(A / np.sum(A, axis=1))

	model = HMM(pi, A, B, outcome_dict, state_dict)


	return model


def switch_key_value_dict(state_dict):
	return {y: x for x, y in state_dict.items()}


def find_removal_index(arr):
	arg = np.argwhere(np.sum(arr, axis=0) == 0)
	if arg:
		return np.reshape(arg, (1,))
	return None



# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	for line in test_data:
		tg = model.viterbi(line.words)
	###################################################
	return tagging
