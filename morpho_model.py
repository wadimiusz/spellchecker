import codecs, re, collections, hunspell, sys, treetagger, pickle, subprocess

from math import log
from random import choice
from itertools import product
from sklearn.externals import joblib
from pynlpl.lm.lm import ARPALanguageModel
from itertools import groupby

LM = ARPALanguageModel("/home/v.fomin/morpho_model/morpho_model")

tt_ru = treetagger.TreeTagger(language = 'russian')

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(re.findall('[А-Яа-яЁё0-9A-Za-z_]+', x))

f = open('/home/v.fomin/queries.csv')
queries = [x for x in f.read().split('\n') if len(x) > 0]
f.close()

c = collections.Counter(tokenize(' '.join(queries)))
pickle.dump(c, open('c.pkl', 'wb'))

most_common = [a for a, b in c.most_common(10)] + ['_NEWLINE_']
del c

def queries2morpho(queries):
	queries = ' _NEWLINE_ '.join([' '.join(query) for query in queries])
	queries = tt_ru.tag(queries)
	queries = [query[0] if query[0] in most_common else query[1] for query in queries]
	queries = [list(group) for k, group in groupby(queries, lambda x: x == "_NEWLINE_") if not k]
	return queries


h = hunspell.HunSpell('ru_RU1_utf.dic', 'ru_RU_utf.aff')

f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
corpus_lines = f.read().split('\r\n')

border = int(corpus_lines.pop(0))
marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border] if len(query.rsplit(',', 3)) == 4]
misspelled_queries = [tokenize(query) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']
right_answers_for_misspelled_queries = [tokenize(suggestion) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']

f.close()


if __name__ == '__main__' and 'train' in sys.argv:
	print("Training")
	word2morpho_queries = queries2morpho([tokenize(query) for query in queries])
	joblib.dump(word2morpho_queries, 'word2morpho_queries.pkl')
	f = open('/home/v.fomin/queries_morpho1.csv', 'w')
	f.write('\n'.join([' '.join(line) for line in word2morpho_queries]))
	f.close()
	subprocess.call('/home/v.fomin/SRILM/bin/i686-m64/ngram-count -text /home/v.fomin/queries_morpho1.csv -lm /home/v.fomin/morpho_model/morpho_model -order 3 -write /home/v.fomin/morpho_model/morpho_ngram -kndiscount3 -unk', shell = True)
else:
	f = codecs.open('/home/v.fomin/queries_morpho.csv', 'r')
	word2morpho_queries = [tokenize(line) for line in f.read().split('\r\n')]
	f.close()


ngram_score = lambda sentence: LM.score(sentence)

def suggest_word(word):
	if h.spell(word):
		return [word]
	else:
		return list(set([word] + h.suggest(word)))

def suggest(sentence):
	result = product(*[suggest_word(word) for word in sentence])
	result = [tokenize(' '.join(x)) for x in result]
	return result

def best_suggestion(suggestions):
	suggestions_scores = zip(suggestions, [ngram_score(suggestion) for suggestion in suggestions])
	suggestions_scores = sorted(suggestions_scores, key = lambda x: x[1])
	return suggestions_scores[-1][0]

if __name__ == '__main__' and 'test' in sys.argv:	
	for i in range(10):
		specific_query = choice(misspelled_queries)
		print(specific_query)
		print(best_suggestion(specific_query))
	
	number_misspelled = len(misspelled_queries)
	print(number_misspelled)
	result = ''
	
	#suggestions = [print(counter) or suggest(sentence) for counter, sentence in enumerate(misspelled_queries)]
	#pickle.dump(suggestions, open('suggestions.pkl', 'wb'))
	suggestions = pickle.load(open('suggestions.pkl', 'rb'))
	
	for counter, current_sentence, suggestions, right_answer in zip(range(number_misspelled), misspelled_queries, suggestions,  right_answers_for_misspelled_queries):
		print('\n')
		current_suggestion1 = current_suggestion2 = best_suggestion(suggestions)
		print(' '.join(current_sentence),' '.join(right_answer),' '.join(current_suggestion1),' '.join(current_suggestion2), sep = '\n')
		result = result + ('\n%s,%s,%s,%s' % (' '.join(current_sentence),' '.join(right_answer),' '.join(current_suggestion1),' '.join(current_suggestion2)))
		print(counter)
		if counter % 10 == 0:
			f = open('result.txt', 'w')
			f.write(result)
			f.close()
	
	
	f = open('result.txt', 'w')
	f.write(result)
	f.close()
