import codecs, re, nltk, collections, hunspell, pymorphy2, sys

from math import log
from random import choice
from itertools import product

morph = pymorphy2.MorphAnalyzer() 

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(re.findall('[А-Яа-яЁё0-9A-Za-z_]+', x))

def word2pos(words):
	result = []
	for word in words:
		if word in ('д', 'ул', 'пл', 'стр', 'вл', 'кв', 'д', 'г', 'пр'):
			result.append('NOUN')
		elif word == '_NUMBER_':
			result.append('_NUMBER_')
		else:
			result.append(morph.parse(word)[0].tag.POS)
	
	return result

h = hunspell.HunSpell('ru_RU1_utf.dic', 'ru_RU_utf.aff')

f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
corpus_lines = f.read().split('\r\n')
border = int(corpus_lines.pop(0))
marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border]]
misspelled_queries = [tokenize(query) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']
suggestions_for_misspelled_queries = [tokenize(suggestion) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']

f.close()

f = codecs.open("queries/queries1.csv", mode = "rU", encoding = "utf-8")
lines = f.read().split('\r\n')
f.close()
f = codecs.open("queries/queries2.csv", mode = "rU", encoding = "utf-8")
lines = lines + f.read().split('\r\n')
f.close()

lines = [line for line in lines if line != '']
lines = [line.rsplit(',', 1) for line in lines]
queries = [tokenize(processed_query) for (query, processed_query) in lines[border:]] 
word2pos_queries = [word2pos(query) for query in queries]

ngram_dict = collections.Counter()
lens = {}

for query in word2pos_queries:
	for i in range(1, 7):
		if i in lens.keys():
			lens[i] += len(list(nltk.ngrams(query, i)))
		else:
			lens[i] = len(list(nltk.ngrams(query, i)))
		for ngram in nltk.ngrams(query, i):
			ngram_dict[ngram] += 1


def ngram_freq(ngrm):
	ngrm = tuple(ngrm)
	if ngrm in ngram_dict:
		return ngram_dict[ngrm]  / lens[len(ngrm)]
	elif len (ngrm) == 0:
		return 1
	else:
		return 0

def conditional_probability(word, previous_words):
	try:
		result = log(ngram_freq(previous_words + [word]) / ngram_freq(previous_words))
	except Exception:
		result = -1000000
	finally:
		return result

def ngram_score(words, window = 1):
	result = 0
	for counter, word in enumerate(words):
		if counter-window > 0:
			result += conditional_probability(word, words[counter-window:counter])
		else:
			result += conditional_probability(word, words[:counter])	
	
	return result


def suggest_word(word):
	if h.spell(word):
		return [word]
	else:
		return list(set([word] + h.suggest(word)))

def suggest(sentence):
	result = product(*[suggest_word(word) for word in sentence])
	result = [tokenize(' '.join(x)) for x in result]
	return result

def best_suggestion(sentence):
	suggestions = suggest(sentence)
	suggestions_scores = zip(suggestions, [ngram_score(word2pos(suggestion)) for suggestion in suggestions])
	suggestions_scores = sorted(suggestions_scores, key = lambda x: x[1])
	return suggestions_scores[-1][0]

if __name__ == '__main__':	
	for i in range(10):
		specific_query = choice(misspelled_queries)
		print(specific_query)
		print(best_suggestion(specific_query))
	
	number_misspelled = len(misspelled_queries)
	print(number_misspelled)
	number = 0
	result = ''
	number = 0
	
	for i in range(number_misspelled):
		print('\n')
		number += 1
		current_sentence = misspelled_queries[i]
		right_answer = suggestions_for_misspelled_queries[i]
		current_suggestion1 = current_suggestion2 = best_suggestion(current_sentence)
		print(' '.join(current_sentence),' '.join(right_answer),' '.join(current_suggestion1),' '.join(current_suggestion2), sep = '\n')
		result = result + ('\n%s,%s,%s,%s' % (' '.join(current_sentence),' '.join(right_answer),' '.join(current_suggestion1),' '.join(current_suggestion2)))
		print(number)
	
	
	f = open('result.txt', 'w')
	f.write(result)
	f.close()

