from __future__ import division
import random
import math
import numpy as np
import pandas as pd
import copy
import sys
import os
import imp

from general_learner_functions import *

#### READ IN COMMAND LINE ARGUMENTS

# get learning model type
model = sys.argv[1]
if model not in ['iterated', 'interactive']:
	print "ERROR: INVALID MODEL TYPE"
	sys.exit()

# check for correct number of parameters
if model == 'iterated':
	if len(sys.argv) != 9:
		print "ERROR: INCORRECT NUMBER OF ARGUMENTS"
		sys.exit()
else:
	if len(sys.argv) != 8:
		print "ERROR: INCORRECT NUMBER OF ARGUMENTS"
		sys.exit()

# import grammar data
# module = imp.load_source('setup', sys.argv[2])
# from setup import *
# # get data file name to use in naming output file
# input_file = sys.argv[2].split('.')[0]
input_file_tag = sys.argv[2]

tabs_data = pd.read_csv(input_file_tag+'tabs.csv', sep=',', header=0)
start_probs = make_start_probs_dict(tabs_data)
start_weights = make_start_weights_dict(tabs_data)
violations = make_violations_dict(tabs_data)

langs_data = pd.read_csv(input_file_tag+'langs.csv', sep=',', header=0)
languages = make_languages_dict(langs_data)

# assign numerical parameters
runs = int(sys.argv[3])
gens = int(sys.argv[4])
steps = int(sys.argv[5])
learning_rate = float(sys.argv[6])

# get assignment conditions for initial and/or new agents
init_agents = sys.argv[7]
if model == 'iterated':
	new_agents = sys.argv[8]
else:
	new_agents = sys.argv[7]
if init_agents not in ['random', 'zero', 'preset']:
	print "ERROR: INVALID ARGUMENT FOR INITIAL AGENT WEIGHTS"
	sys.exit()
if new_agents not in ['random', 'zero', 'preset']:
	print "ERROR: INVALID ARGUMENT FOR NEW AGENT WEIGHTS"
	sys.exit()	

#### ADDITIONAL PARAMETERS

# set sampling range for random constraint weights
samp_range = 10

# set how often output writes out
# output will be written every steps/div learning steps
if steps < 20:
	div = 1
elif model == 'iterated':
	if gens > 1:
		div = steps
	else:
		div = steps/20
else:
	div = steps/20

#### SET NAME OF OUTPUT FILE

if model == 'iterated':
	output_file = model+'_'+input_file_tag+\
					'_'+'R'+str(runs)+'_'+'G'+str(gens)+'_'+'LS'+str(steps)+\
					'_'+init_agents+new_agents+'.txt'
else:
	output_file = model+'_'+input_file_tag+\
					'_'+'R'+str(runs)+'_'+'G'+str(gens)+'_'+'LS'+str(steps)+\
					'_'+new_agents+'.txt'

#### GET LIST OF CONSTRAINTS AND OUTPUT FORMS #################

constraints = get_constraints(start_weights)
outputs = get_outputs(start_probs)

Cons = ','.join(c for c in constraints)
Outs = ','.join(o for o in outputs)

#### IF OUTPUT FILE WITH SAME NAME ALREADY EXISTS, GIVE A WARNING AND QUIT
#### OTHERWISE INITIALIZE OUTPUT FILE

exists = os.path.isfile(output_file)
if exists:
	print 'ERROR: FILE NAMED', output_file, 'ALREADY EXISTS'
	sys.exit()
else:
	with open(output_file, 'a') as outfile:
		outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % 
						('Run', 'Gen', 'Step', 'Agent', 'Lang', 'Var', 
							Cons, Outs, 'Avg_Win', 'Updates'))
		outfile.write('\n')

#### START LEARNING LOOP

for r in np.arange(runs):
	print "RUN", r+1, '/', runs
	# set initial weights for both agents
	if init_agents == 'preset':
		A1_weights = copy.deepcopy(start_weights)
	elif init_agents == 'zero':
		A1_weights = assign_zero_weights(start_weights)
	elif init_agents == 'random':
		A1_weights = sample_random_weights(start_weights, samp_range)
	if model == 'iterated':
		if new_agents == 'random':
			A2_weights = sample_random_weights(start_weights, samp_range)
		elif new_agents == 'preset':
			A2_weights = copy.deepcopy(start_weights)
		elif new_agents == 'zero':
			A2_weights = assign_zero_weights(start_weights)
	else: 
		A2_weights = copy.deepcopy(A1_weights)
	# set initial probability distributions for both agents
	A1_probs = update_probs(A1_weights, violations)
	A2_probs = update_probs(A2_weights, violations)
	# calculate average probability over highest probability candidates
	A1_avg_win = calc_avg_win(A1_probs)
	A2_avg_win = calc_avg_win(A2_probs)
	# categorize extent of variation in grammars of each agent
	A1_var = categorize_var(A1_avg_win)
	A2_var = categorize_var(A2_avg_win)
	# categorize pattern corresponding to each agent's grammar
	A1_lang = categorize_lang(A1_probs, languages)
	A2_lang = categorize_lang(A2_probs, languages)

	for g in np.arange(gens):
		A1_updates = 0
		A2_updates = 0
		for s in np.arange(steps):
			## write initial state data
			if s == 0:
				# categorize pattern corresponding to each agent's grammar
				A1_lang = categorize_lang(A1_probs, languages)
				A2_lang = categorize_lang(A2_probs, languages)
				# get probabilities on each output candidate for each agent
				A1_outprobs = get_outprobs(A1_probs)
				A2_outprobs = get_outprobs(A2_probs)
				# create and write output lines
				A1_write = ','.join(str(A1_weights[c]) for c in constraints)
				A2_write = ','.join(str(A2_weights[c]) for c in constraints)
				A1_probwrite = ','.join(str(A1_outprobs[o]) for o in outputs)
				A2_probwrite = ','.join(str(A2_outprobs[o]) for o in outputs)
				with open(output_file, 'a') as outfile:
					outfile.write('%s,%s,%s,%s,%s,%s,%s,%s' % 
									(r+1, g+1, s, 'A1', A1_lang, A1_write, 
										A1_probwrite, A1_updates))
					outfile.write('\n')
					outfile.write('%s,%s,%s,%s,%s,%s,%s,%s' % 
									(r+1, g+1, s, 'A2', A2_lang, A2_write, 
										A2_probwrite, A2_updates))
					outfile.write('\n')
			## A1 is teacher agent, A2 is learner agent
			# sample an input form
			samp_input = np.random.choice(start_probs.keys())
			# sample outputs for each agent
			Out_A1 = sample_output(A1_probs[samp_input])
			Out_A2 = sample_output(A2_probs[samp_input])
			if Out_A1 != Out_A2:
				A2_updates += 1
				# get violation profiles
				A1_vios = violations[samp_input][Out_A1]
				A2_vios = violations[samp_input][Out_A2]
				# update learner agent weights
				A2_weights = update_weights(A2_weights, A1_vios, A2_vios, learning_rate)
				# update learner agent probabilities
				A2_probs = update_probs(A2_weights, violations)
			if model == 'interactive':
				## switch roles; A2 is teacher agent, A1 is learner agent
				# sample an input form
				samp_input = np.random.choice(start_probs.keys())
				# sample outputs for each agent
				Out_A1 = sample_output(A1_probs[samp_input])
				Out_A2 = sample_output(A2_probs[samp_input])
				if Out_A1 != Out_A2:
					A1_updates += 1
					# get violation profiles
					A1_vios = violations[samp_input][Out_A1]
					A2_vios = violations[samp_input][Out_A2]
					# update learner agent weights
					A1_weights = update_weights(A1_weights, A2_vios, A1_vios, learning_rate)
					# update learner agent probabilities
					A1_probs = update_probs(A1_weights, violations)
			## write current agent data at appropriate learning steps
			if (s+1)%div == 0:
				# categorize pattern corresponding to each agent's grammar
				A1_lang = categorize_lang(A1_probs, languages)
				A2_lang = categorize_lang(A2_probs, languages)
				# get probabilities on each output candidate for each agent
				A1_outprobs = get_outprobs(A1_probs)
				A2_outprobs = get_outprobs(A2_probs)
				# create and write output lines
				A1_write = ','.join(str(A1_weights[c]) for c in constraints)
				A2_write = ','.join(str(A2_weights[c]) for c in constraints)
				A1_probwrite = ','.join(str(A1_outprobs[o]) for o in outputs)
				A2_probwrite = ','.join(str(A2_outprobs[o]) for o in outputs)
				with open(output_file, 'a') as outfile:
					outfile.write('%s,%s,%s,%s,%s,%s,%s,%s' % 
									(r+1, g+1, s+1, 'A1', A1_lang, A1_write, 
										A1_probwrite, A1_updates))
					outfile.write('\n')
					outfile.write('%s,%s,%s,%s,%s,%s,%s,%s' % 
									(r+1, g+1, s+1, 'A2', A2_lang, A2_write, 
										A2_probwrite, A2_updates))
					outfile.write('\n')

		## after final learning step, A2 becomes A1, initialize new A2
		A1_weights = copy.deepcopy(A2_weights)
		if new_agents == 'random':
			A2_weights = sample_random_weights(start_weights, samp_range)
		elif new_agents == 'preset':
			A2_weights = copy.deepcopy(start_weights)
		elif new_agents == 'zero':
			A2_weights = assign_zero_weights(start_weights)
		A1_probs = update_probs(A1_weights, violations)
		A2_probs = update_probs(A2_weights, violations)

print 'DONE', output_file