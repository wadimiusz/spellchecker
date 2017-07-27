import re, ngram_model, editdistance, Levenshtein, hunspell, codecs, time, pickle, sys, time, subprocess
#import pos_model
import morpho_model

import numpy as np

from metrics import alignment
#from pos_model import word2pos
from morpho_model import queries2morpho
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#from word2vecs import suggest
from get_close_matches import suggest
from random import shuffle
from word2phonetics import word2phonetics
from metrics import count

from prep_model0 import corpus2prep as corpus2prep0
from prep_model0 import score as score0
from prep_model1 import corpus2prep as corpus2prep1
from prep_model1 import score as score1

from itertools import product, groupby
from functools import reduce

from weight_levenshtein import distance, delete_cost, insert_cost, substitute_cost
from Levenshtein import distance as simple_distance

print(time.strftime('%X %x'))

h = hunspell.HunSpell('ru_RU1_utf.dic', 'ru_RU_utf.aff')
c = joblib.load('c.pkl')

#pos_score = pos_model.ngram_score
ngram_score = ngram_model.ngram_score
morpho_score = morpho_model.ngram_score

#def suggest(s):
#	s = ' '.join(s)
#	commandline = '~/"Рабочий стол"/Artifacts/unisearch parse ~/"Рабочий стол"/Artifacts/unisearch.idx -q "%s" > ~/"Рабочий стол"/Artifacts/parse_result.txt' % s
#	returncode = subprocess.call(commandline, shell = True)
#	assert returncode == 0
#	output = open("/home/v.fomin/Рабочий стол/Artifacts/parse_result.txt").read().split("\n")
#	start = next(counter for counter, el in enumerate(output) if el == "Query Parts:")
#	output = output[start+1:-3]
#	output = [list(group) for k, group in groupby(output, lambda x: x == "") if not k]
#	output = [['_NUMBER_'] if group[0].split(' ')[0] == 'number' else [group[0].split(' ')[0]] + [s.split(' | ')[-1] for s in group[2:]] for group in output]
#	max_options = int(1000 ** (1.0 / len(output)))
#	options = [list(set(option[:max_options])) for option in output]
#	suggestions = [list(x) for x in (product(*options))]
#	return suggestions


has_dictionary_partitions = lambda word: max([min([h.spell(word) for word in suggestion.split(' ')] + [True]) for suggestion in h.suggest(word)] + [False])

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(re.findall('[А-Яа-яЁё0-9A-Za-z_]+', x))

all_spell = lambda s: min([h.spell(word) for word in tokenize(s)] + [True])

def space_insertions(sentence, suggestion):
	return len([1 for (type, spos, dpos)  in Levenshtein.editops(sentence, suggestion) if type == 'insert' and suggestion[dpos] == ' '])

def space_deletions(sentence, suggestion):
	return len([1 for (type, spos, dpos)  in Levenshtein.editops(sentence, suggestion) if type == 'delete' and sentence[spos] == ' '])

def corpus2features(sentences, suggestions):
	sent_sug = list(zip(sentences[:], suggestions[:]))
	aligned = [[(' '.join(sent), ' '.join(sug1)) for (sent, sug1, sug2) in alignment(sentence[:], suggestion[:], suggestion[:])] for sentence, suggestion in sent_sug]
	result = np.array([len(suggestion) for sentence, suggestion in sent_sug]) #1
	result = np.vstack([result, [editdistance.eval(' '.join(sentence), ' '.join(suggestion)) for sentence, suggestion in sent_sug]]) #2; это error score, поставил edit distance т. к. пока не понимаю откуда брать error score
	a = [ngram_score(suggestion) for suggestion in suggestions]
	result = np.vstack([result, a]) #3
	
	dict_words = [[(sent, sug) for (sent, sug) in align if all_spell(sent)] for align in aligned]
	OOVs = [[(sent, sug) for (sent, sug) in align if not all_spell(sent)] for align in aligned] #out of vocabulary words
	
	#result = np.vstack([result, [len(OOV) for OOV in OOVs]])
	#лучше бы так:
	result = np.vstack([result, [len([(sent, sug) for (sent, sug) in align if not all_spell(sug)]) for align in aligned]]) #4
	result = np.vstack([result, [len([(sent, sug) for (sent, sug) in align if all_spell(sent) and not all_spell(sug)]) for align in aligned]]) #5
	result = np.vstack([result, [len([(sent, sug) for (sent, sug) in align if not all_spell(sent) and all_spell(sug)]) for align in aligned]]) #6
	
	more_frequent_group1 = lambda group1, group2: min([c[word] for word in group1], default = 0) - min([c[word] for word in group2], default = 0)
	more_frequent_group2 = lambda group1, group2: min([c[word] for word in group2], default = 0) - min([c[word] for word in group1], default = 0)
	more_frequent_sent1 = lambda pairs: len([1 for pair in pairs if more_frequent_group1(*pair)])
	more_frequent_sent2 = lambda pairs: len([1 for pair in pairs if more_frequent_group2(*pair)])

	more_freq1 = [more_frequent_sent1(pair) for pair in aligned]
	more_freq2 = [more_frequent_sent2(pair) for pair in aligned]

	#result = np.vstack([result, more_freq1]) #7
	result = np.vstack([result, more_freq2]) #7
	#много ли слов сменилось с более частотных на менее и наоборот	
	
	OOV_corrections = [sum([editdistance.eval(sent, sug) for (sent, sug) in OOV]) for OOV in OOVs]
	dict_word_corrections = [sum([editdistance.eval(sent, sug) for (sent, sug) in dict_word]) for dict_word in dict_words]
	
	result = np.vstack([result, OOV_corrections]) #8
	result = np.vstack([result, dict_word_corrections]) #9
	
	#фича №8 -- number of corrections in capitalized words, нам неактуально
	
	distance_one = [sum([1 for (sent, sug) in align if editdistance.eval(sent, sug) == 1]) for align in aligned]
	result = np.vstack([result, distance_one]) #10
	
	#number of corrections by phonetic similarity пока не разработана
	#number of corrections by word lists пока не разработана
	
	result = np.vstack([result, [space_insertions(' '.join(sentence), ' '.join(suggestion)) for sentence, suggestion in sent_sug]]) #11
	result = np.vstack([result, [space_deletions(' '.join(sentence), ' '.join(suggestion)) for sentence, suggestion in sent_sug]]) #12
	OOV_dictionary_partitions = [sum([1 for (sent, sug) in align if has_dictionary_partitions(sent)]) for align in aligned]
	result = np.vstack([result, OOV_dictionary_partitions]) #13; низачем похоже не нужно
	
	#result.append(pos_score(word2pos(suggestion)))
	result = np.vstack([result, [morpho_score(suggestion) for suggestion in queries2morpho(suggestions)]]) #14
	
	distance1 = lambda word1, word2: distance(word1, word2, delete_cost = lambda x: 10, insert_cost = lambda x: 1)
	distance2 = lambda word1, word2: distance(word1, word2, delete_cost = lambda x: 1, insert_cost = lambda x: 10)
	
	result = np.vstack([result, [distance(' '.join(sentence), ' '.join(suggestion)) for sentence, suggestion in sent_sug]]) #15
	result = np.vstack([result, [distance1(' '.join(sentence), ' '.join(suggestion)) for sentence, suggestion in sent_sug]]) #16
	result = np.vstack([result, [distance2(' '.join(sentence), ' '.join(suggestion)) for sentence, suggestion in sent_sug]]) #17
	
	result = np.vstack([result, [editdistance.eval(word2phonetics(' '.join(sentence)), word2phonetics(' '.join(suggestion))) for sentence, suggestion in sent_sug]]) #18
	
	my_w2v1 = Word2Vec.load('word2vecs/my_w2v1')
	result = np.vstack([result, my_w2v1.score(suggestions)]) #19
	

	#последняя фича -- предложная модель, её мы ещё не разработали
	sugs0 = corpus2prep0([' '.join(suggestion) for suggestion in suggestions])
	sugs1 = corpus2prep1([' '.join(suggestion) for suggestion in suggestions])
	
	result = np.vstack([result, [score0(sug) for sug in sugs0]]) #20
	result = np.vstack([result, [sum([score1(x) for x in sug]) for sug in sugs1]]) #21
	
	return np.array(result).transpose()

def best_suggestion(query, score):
	suggestions = suggest(query)
	suggestion_scores = list(zip(suggestions, score(query, suggestions)))
	suggestions = sorted(suggestion_scores, key = lambda x: x[1])
	return suggestions[0][0]

iterations = {}

def iter_best_suggestion(query, score):
	result = query
	results = [query]
	best = best_suggestion(result, score)
	for counter in range(3):
		print("Iteration", counter, "intermediate result", result)
		counter += 1
		if best in results:
			print("Final result is", result)
			return result
		else:		
			result = best[:]
			results.append(result)
			best = best_suggestion(result, score)
 
	print("Counter: ", counter)
	if counter in iterations:
		iterations[counter] += 1
	else:
		iterations[counter] = 1	
	return result

def main():
	f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
	corpus_lines = f.read().split('\r\n')
	border = int(corpus_lines.pop(0))
	marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border]]
	misspelled = [(tokenize(query), tokenize(suggestion)) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>' and suggestion != '']

	f = codecs.open("queries/queries1.csv", mode = "rU", encoding = "utf-8")
	lines = f.read().split('\r\n')
	f.close()
	f = codecs.open("queries/queries2.csv", mode = "rU", encoding = "utf-8")
	lines = lines + f.read().split('\r\n')
	f.close()

	lines = [line for line in lines if line != '']
	lines = [line.rsplit(',', 1) for line in lines]
	queries = [tokenize(processed_query) for (query, processed_query) in lines[border:]]
	
	#subprocess.call('python3 morpho_model.py train', shell = True)

	kf = KFold(n_splits = 5, shuffle = True)
	(TPs, FPs, FNs, F1s, precisions, recalls) = ([], [], [], [], [], [])
	iteration = 0
	for iteration, (train_index, test_index) in enumerate(kf.split(misspelled)):
		print("CV iteration #%d" % iteration)
	
		misspelled_train = [misspelled[i] for i in train_index]
		misspelled_test = [misspelled[i] for i in test_index]
	
		if 'save' in sys.argv:
			print("Creating the training set")
			for counter, (query, suggestion) in enumerate(misspelled_train):
				print("Trying to learn from query %s at number %d" % (query, counter))
				sugs = sorted(suggest(query), key = lambda x: simple_distance(' '.join(x), ' '.join(suggestion)))
				(new_winner, old_winner) = (sugs[0], query)
				for i in range(10):
					print("New winner: ", new_winner) #debug
					losers = sugs[1:]
					winner_vect = corpus2features([old_winner], [new_winner])
					losers_vect = corpus2features([old_winner] * len(losers), losers)
					X_train_append = winner_vect - losers_vect
					try:					
						X_train = np.vstack([X_train, X_train_append])
						y_train = np.hstack([y_train, [1] * (len(sugs) - 1)])
					except NameError:
						X_train = X_train_append
						y_train = np.array([1] * (len(sugs) - 1))
		
					X_train = np.vstack([X_train, -X_train_append])
					y_train = np.hstack([y_train, [0] * (len(sugs) - 1)])
		
					sugs = sorted(suggest(new_winner), key = lambda x: simple_distance(' '.join(x), ' '.join(suggestion)))
					(new_winner, old_winner) = (sugs[0], new_winner)
		
					if len(sugs) == 0 or new_winner == old_winner:
						break
	
				assert X_train.shape[0] == y_train.shape[0]	
				joblib.dump(X_train, 'X_train.pkl')
				joblib.dump(y_train, 'y_train.pkl')
		
		else:
			print("Loading the training set")
			X_train = joblib.load('X_train.pkl')
			y_train = joblib.load('y_train.pkl')

		Scaler = StandardScaler()
		X_train = Scaler.fit_transform(X_train)

		joblib.dump(Scaler, 'Scaler.pkl')

		print("Training set is ready")

		if 'train' in sys.argv:
			logreg = LogisticRegression(verbose = True, C = 1, fit_intercept = False).fit(X_train, y_train)
			joblib.dump(logreg, 'logreg.pkl')
			MLP = MLPClassifier(verbose = True, hidden_layer_sizes = (500,)).fit(X_train, y_train)
			joblib.dump(MLP, 'MLP.pkl')

			#SVM = SVC(verbose = True).fit(X_train, y_train)
			#joblib.dump(SVM, 'SVM.pkl')
		else:
			logreg = joblib.load('logreg.pkl') 
			MLP = joblib.load('MLP.pkl')
			SVM = joblib.load('SVM.pkl')


		print("Models are ready")
		
		logreg_function = lambda query, s: -logreg.decision_function(Scaler.transform(corpus2features([query] * len(s), s)))
		MLP_function = lambda query, s: -MLP.predict_proba(Scaler.transform(corpus2features([query] * len(s), s)))[:, 1]
		SVM_function = lambda query, words: -SVM.decision_function([sent2features(words, query)])
		
		if 'logreg' in sys.argv:
			result = []
			(TP, FN, FP) = (0, 0, 0)
			for (counter, (query, correct_answer)) in enumerate(misspelled_test):
				print("Trying to process " + ' '.join(query) + " with logreg")
				bestie = iter_best_suggestion(query, score = logreg_function)
				print(query, correct_answer, bestie, counter, sep = '\n')
				(new_TP, new_FP, new_FN) = count(query, correct_answer, bestie)
				TP += new_TP
				FP += new_FP
				FN += new_FN
				result.append("%s,%s,%s,%s" % (' '.join(query), ' '.join(correct_answer), ' '.join(bestie), ' '.join(bestie)))
				print(TP, FP, FN)
				if counter % 10 == 0:
					f = open('result1.txt', 'w')
					f.write('\n'.join(result))
					f.close()
			TPs.append(TP)
			FPs.append(FP)
			FNs.append(FN)
			precision = TP / (TP + FP)
			recall = TP / (TP + FN)
			F1 = 2 * precision * recall / (precision + recall)
			print("Iteration number %d has performed as follows: " % iteration)
			print("Precision: ", precision)
			print("Recall: ", recall)
			print("F1: ", F1)
			F1s.append(F1)
			precisions.append(precision)
			recalls.append(recall)
			print("F1s:")
			print(*F1s)
			print("Precisions: ", *precisions)
			print("Recalls: ", *recalls)
			f = open('F1s.txt', 'w')
			f.write(str(F1s))
			f.close()
			f = open('precisions.txt', 'w')
			f.write(str(precisions))
			f.close()
			f = open('recalls.txt', 'w')
			f.write(str(recalls))
			f.close()
	
		if 'MLP' in sys.argv:
			result = []
			for (counter, ((query, correct_answer), sugs)) in enumerate(zip(misspelled_test, suggestions)):
				print("Trying to process " + ' '.join(query) + " with MLP")
				bestie = iter_best_suggestion(query, score = MLP_function)
				print(query, correct_answer, bestie, counter, sep = '\n')
				result.append("%s,%s,%s,%s" % (' '.join(query), ' '.join(correct_answer), ' '.join(bestie), ' '.join(bestie)))
				if counter % 10 == 0:
					f = open('result2.txt', 'w')
					f.write('\n'.join(result))
					f.close()

		if 'SVM' in sys.argv:
			result = []
			counter = 0
			for (query, suggestion) in misspelled_test:
				print("Trying to process" + ' '.join(query) + 'with SVM')
				counter += 1
				bestie = iter_best_suggestion(query, score = SVM_function)
				print(query, suggestion, bestie, counter, sep = '\n')
				result.append("%s,%s,%s,%s" % (' '.join(query), ' '.join(suggestion), ' '.join(bestie), ' '.join(bestie)))
				if counter % 10 == 0:
					f = open('result3.txt', 'w')
					f.write('\n'.join(result))
					f.close()

	print(time.strftime('%X %x'))

if __name__ == '__main__':
	main()
