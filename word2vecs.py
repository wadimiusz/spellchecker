import codecs, logging, pymorphy2, hunspell, editdistance, re

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from re import findall
from sys import argv
from random import choice
#web = KeyedVectors.load_word2vec_format('word2vecs/web.bin', binary=True)
from itertools import repeat, product

morph = pymorphy2.MorphAnalyzer()

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(findall('[А-Яа-яЁё0-9A-Za-z_]+', x))
tag = lambda x: morph.parse(x)[0].normal_form + '_' + morph.parse(x)[0].tag.POS
h = hunspell.HunSpell('ru_RU1_utf.dic', 'ru_RU_utf.aff')

f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
corpus_lines = f.read().split('\r\n')
border = int(corpus_lines.pop(0)) 
marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border]]
misspelled_queries = [tokenize(query) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']
suggestions_for_misspelled_queries = [tokenize(suggestion) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']

f.close()
if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if 'save' in argv:
	f = codecs.open("queries/queries1.csv", mode = "rU", encoding = "utf-8")
	lines = f.read().split('\r\n')
	f.close()
	f = codecs.open("queries/queries2.csv", mode = "rU", encoding = "utf-8")
	lines = lines + f.read().split('\r\n')
	f.close()
	lines = [line for line in lines if line != '']
	lines = [line.rsplit(',', 1) for line in lines]
	queries = [processed_query for (query, processed_query) in lines]
	tokenized_queries = [tokenize(query) for query in queries]
	my_w2v1 = Word2Vec(tokenized_queries,window = 1, hs = 1, negative = 0, iter = 10, sg = 0, min_count = 1, workers = 8)
	my_w2v1.save('word2vecs/my_w2v1')
	my_w2v2 = Word2Vec(tokenized_queries, window = 1, hs = 1, negative = 0, iter = 10, sg = 1, min_count = 1, workers = 8)
	my_w2v2.save('word2vecs/my_w2v2')

else:
	my_w2v1 = Word2Vec.load('word2vecs/my_w2v1')
	my_w2v2 = Word2Vec.load('word2vecs/my_w2v2')

def suggest_word(word, my_w2v):
	result = [word]
	if not h.spell(word):
		result = result + [w for w in h.suggest(word) if w in my_w2v.wv.vocab]
		if word in my_w2v.wv.vocab:
			result = result + [a for (a, b) in my_w2v.most_similar(word) if editdistance.eval(a, word) <= 3 or word in a]
	return result

def suggest(sentence, my_w2v = my_w2v1):
	result = product(*[suggest_word(word, my_w2v) for word in sentence])
	result = [tokenize(' '.join(x)) for x in result]
	return result

def best_suggestion(sentence, my_w2v):
	suggestions = suggest(sentence, my_w2v)
	suggestions_scores = zip(suggestions, my_w2v.score(suggestions))
	suggestions_scores = sorted(suggestions_scores, key = lambda x: x[1])
	return suggestions_scores[-1][0]

def suggest_hunspell(sentence):
	result = []
	for word in sentence:
		if h.spell(word):
			result.append(word)
		else:
			suggestions = h.suggest(word)
			if len(suggestions) > 0:
				result.append(suggestions[0])
			else:
				result.append(word)
	
	return result

if __name__ == '__main__':
	for i in range(10):
		sentence = choice(misspelled_queries)
		print(best_suggestion(sentence, my_w2v1))
		print(best_suggestion(sentence, my_w2v2))
		print(sentence)

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
		current_suggestion1 = best_suggestion(current_sentence, my_w2v1)
		current_suggestion2 = best_suggestion(current_sentence, my_w2v2)
		print(' '.join(current_sentence),' '.join(right_answer),' '.join(current_suggestion1),' '.join(current_suggestion2), sep = '\n')
		result = result + ('\n%s,%s,%s,%s' % (' '.join(current_sentence),' '.join(right_answer),' '.join(current_suggestion1),' '.join(current_suggestion2)))
		print(number)

	
	f = open('result.txt', 'w')
	f.write(result)
	f.close()


#suggestions = [' '.join(best_suggestion(sentence)) for sentence in misspelled_queries if print(sentence) or sentence in misspelled_queries]
#errors_and_suggestions = list(zip(errors, suggestions))
#correct_suggestions = [x for x in errors_and_suggestions if x[0] == x[1]]
#print(errors_and_suggestions[:20])
#print(number_misspelled)
#print(correct_suggestions)
