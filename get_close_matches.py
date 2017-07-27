import subprocess, codecs, re, hunspell

from itertools import groupby, product
from Levenshtein import distance, ratio 
from functools import reduce
from sklearn.externals import joblib
from difflib import get_close_matches
from fuzzywuzzy import process, fuzz

mask = lambda x: re.sub('[0-9]+[А-ЯЁа-яё]?', '_NUMBER_', x)

c = joblib.load('c.pkl')
vocab = c.keys()

h = hunspell.HunSpell('ru_RU1_utf.dic', 'ru_RU_utf.aff')

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(re.findall('[А-Яа-яЁё0-9A-Za-z_]+', x))

def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i:i+n]) for i in range(len(lst)-n+1))

def adjacent_vocab_corrections(sentence, suggestion):
	return contains_sublist([sent != sug and h.spell(sent) for sent, sug in zip(sentence, suggestion)], [True, True])

lexicon = ['_NUMBER_']
def suggest(words):
	options = []
	for word in words:
		if word in lexicon:
			appended = [(word,)]
		else:
			close_matches = sorted([(a,) for a, b in process.extract(word, vocab, limit = 10, scorer = ratio)], key = lambda x: -c[x[0]])
			appended = close_matches[:2] + [(word,)]
			appended = appended + [(word[:i], word[i:]) for i in range(1, len(word)) if c[word[:i]] > c[word] and c[word[i:]] > c[word]]
			appended = appended + [(word[:i-1], word[i:]) for i in range(2, len(word)) if c[word[:i-1]] > c[word] and c[word[i:]] > c[word]]
		
		options.append(appended)
	
	options = set(product(*options))
	options = [sum(sug, ()) for sug in options]
	options = [list(opt) for opt in options]
	options = sorted(options, key = lambda sug: distance(' '.join(words), ' '.join(sug)))[:50]
	options = [opt for opt in options if not adjacent_vocab_corrections(words, opt)]
	return options

def best_suggestion(sentence, correct):
	sugs = suggest(sentence)
	sugs = sorted(sugs, key = lambda sug: distance(' '.join(sug), ' '.join(correct)))
	return sugs[0]

iterations = {}

def iter_best_suggestion(sentence, correct):
	result = sentence
	best = best_suggestion(result, correct)
	for counter in range(10):
		result = best[:]
		best = best_suggestion(result, correct)
		if result == best:
			if counter in iterations:
				iterations[counter] += 1
			else:
				iterations[counter] = 1
			print("Counter: ", counter)
			return result
	
	return result

f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
corpus_lines = f.read().split('\r\n')
border = int(corpus_lines.pop(0))
marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border]]
misspelled = [(tokenize(query), tokenize(suggestion)) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>' and suggestion != '']


if __name__ == '__main__':
	print(best_suggestion("жд вркзал".split(), "жд вокзал".split()))
	result = []
	for (counter, (query, correct_answer)) in enumerate(misspelled):
		print("Trying to process " + ' '.join(query) + " with get_close_matches")
		bestie = iter_best_suggestion(query, correct_answer)
		print(query, correct_answer, bestie, bestie, counter, sep = '\n')
		result.append("%s,%s,%s,%s" % (' '.join(query), ' '.join(correct_answer), ' '.join(bestie), ' '.join(bestie)))
		if counter % 10 == 0:
			f = open('result.txt', 'w')
			f.write('\n'.join(result))
			f.close
			print(iterations)
	f = open('result.txt', 'w')
	f.write('\n'.join(result))
	f.close
	print(iterations)
