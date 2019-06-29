from __future__ import division
import random
import math
import numpy as np
import pandas as pd
import copy
import sys

#################################

def make_start_weights_dict(data):
	'''Takes the tabs.csv file, read in as a pandas DataFrame, and returns a dictionary of the initial
	constraint weights.'''
	start_weights = {}
	for x in list(data.columns.values):
		if x not in ['Input', 'Output', 'Prob']:
			start_weights[x] = data.loc[0,x]
	return start_weights

def make_start_probs_dict(data):
	'''Takes the tabs.csv file, read in as a pandas DataFrame, and returns a dictionary of the initial
	probability for each input-output candidate.'''
	start_probs = {}
	for x in range(len(data.index)):
		if data.iloc[x,0] != 'XweightX':
			if data.iloc[x,0] not in start_probs.keys():
				start_probs[data.iloc[x,0]] = {}
			start_probs[data.iloc[x,0]][data.iloc[x,1]] = data.iloc[x,2]
	return start_probs

# takes a Xtabs.csv and makes the violations dictionary
# violations contains the violations of each constraint
# for each input-output pair
def make_violations_dict(data):
	'''Takes the tabs.csv file, read in as a pandas DataFrame, and returns a dictionary of the constraint
	violations of each input-output candidate.'''
	violations = {}
	for x in range(len(data.index)):
		if data.iloc[x,0] != 'XweightX':
			if data.iloc[x,0] not in violations.keys():
				violations[data.iloc[x,0]] = {}
			violations[data.iloc[x,0]][data.iloc[x,1]] = {}
			for c in list(data.columns.values):
				if c not in ['Input', 'Output', 'Prob']:
					violations[data.iloc[x,0]][data.iloc[x,1]][c] = data.loc[x,c]
	return violations

def make_languages_dict(data):
	'''Takes the langs.csv file, read in as a pandas DataFrame, and returns a dictionary of each 
	possible pattern in the typology, and the input-output pairs that define it.'''
	languages = {}
	for x in range(len(data.index)):
		if data.iloc[x,0] not in languages.keys():
			languages[data.iloc[x,0]] = {}
		for c in list(data.columns.values):
			if c != 'Language':
				if c not in languages[data.iloc[x,0]].keys():
					languages[data.iloc[x,0]][c] = data.loc[x,c]
	return languages

def get_constraints(start_weights):
	'''Takes the start_weights dictionary and returns a list of the constraint labels.'''
	constraints = []
	for c in start_weights.keys():
		constraints.append(c)
	return constraints

def get_outputs(start_probs):
	'''Takes the start_probs dictionary and returns a list of all output forms.'''
	outputs = []
	for i in start_probs.keys():
		outs = start_probs[i]
		for x in outs.keys():
			outputs.append(x)
	return outputs

def sample_random_weights(start_weights, samp_range):
	'''Takes the start_weights dictionary and an upper bound on initial constraint weights to define
	the sampling range. Samples random initial weights for each constraint, and returns a new dictionary.'''
	weights = {}
	for k in start_weights.keys():
		w = random.uniform(0, samp_range)
		weights[k] = w
	return weights

def assign_zero_weights(start_weights):
	'''Takes the start_weights dictionary and returns a new dictionary where each constraint weight
	is set to zero.'''
	weights = {}
	for k in start_weights.keys():
		w = 0
		weights[k] = w
	return weights

def sample_output(prob_dict):
	'''Takes a dictionary containing the probabilities on each output candidate for a given input. Returns
	an output form sampled according to the probability distribution.'''
	items = prob_dict.keys()
	values = []
	for i in items:
		val = prob_dict[i]
		values.append(val)
	samp_out = np.random.choice(items, p=values)
	return samp_out

def update_weights(old_weights, vios_T, vios_L, learning_rate):
	'''Takes a dictionary containing the learner agents' current constraint weights, the violations
	of the teacher agent's input-output form, and the learner agent's input-output form, and the
	learning rate. Returns a dictionary of the learner agent's updated probability distribution.'''
	new_weights = {}
	for con in old_weights:
		curr_old = old_weights[con]
		T = vios_T[con]
		L = vios_L[con]
		new_w = curr_old + ((T-L)*learning_rate)
		if new_w < 0:
			new_w = 0
		new_weights[con] = new_w
	return new_weights

def update_probs(new_weights, violations):
	'''Takes the dictionary containing the learner agent's updated constraint weights, and the
	dictionary containing the violation profiles of the input-output candidates. Returns a
	dictionary containing the learner agent's updated probability distribution.'''
	new_probs = {}
	for i in violations.keys():
		new_probs[i] = {}
		curr_tab = violations[i]
		tab_sum = 0
		for o in curr_tab.keys():
			curr_out = curr_tab[o]
			h = 0
			for con in curr_out.keys():
				h += (new_weights[con] * curr_out[con])
			tab_sum += math.exp(h)
		for o in curr_tab.keys():
			curr_out = curr_tab[o]
			h = 0
			for con in curr_out.keys():
				h += (new_weights[con] * curr_out[con])
			new_probs[i][o] = math.exp(h) / tab_sum
	return new_probs

## identify pattern corresponding to agent's grammar
def categorize_lang(probs_dict, languages):
	'''Takes a dictionary containing the current probabilities on input-output candidates,
	and the dictionary containing the definitions of each possible pattern in the typology,
	and returns the pattern label.'''
	pairs = {}
	for i in probs_dict.keys():
		curr_tab = probs_dict[i]
		for k, v in curr_tab.items():
			if v == max(curr_tab.values()):
				if i in pairs.keys():
					pairs[i] = 'EQUAL'
				else:
					pairs[i] = k
	for k, v in languages.items():
		if pairs == v:
			lang = k
	if pairs not in languages.values():
		lang = 'OTHER'
	return lang

## get probabilities on each output candidate
def get_outprobs(probs_dict):
	'''Takes a dictionary containing the current probabilities on input-output candidates,
	and returns a dictionary with the output forms as keys and their probabilities as values.'''
	outprobs = {}
	for i in probs_dict.keys():
		curr_tab = probs_dict[i]
		for k, v in curr_tab.items():
			outprobs[k] = v
	return outprobs