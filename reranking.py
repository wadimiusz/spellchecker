import morpho_score, re, ngram_model, editdistance, Levenshtein, hunspell, codecs

import numpy as np

from metrics import alignment
from morpho_model import word2pos
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from word2vecs import suggest
from random import shuffle

h = hunspell.HunSpell('ru_RU1_utf.dic', 'ru_RU_utf.aff')

morpho_score = morpho_model.ngram_score
ngram_score = ngram_model.ngram_score
has_dictionary_partitions = lambda word: max([min([h.spell(word) for word in suggestion.split(' ')] + [True]) for suggestion in h.suggest(word)] + [False])

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(re.findall('[А-Яа-яЁё0-9A-Za-z_]+', x))

def space_insertions(sentence, suggestion):
	return len([1 for (type, spos, dpos)  in Levenshtein.editops(sentence, suggestion) if type == 'insert' and suggestion[dpos] == ' '])

def space_deletions(sentence, suggestion):
	return len([1 for (type, spos, dpos)  in Levenshtein.editops(sentence, suggestion) if type == 'delete' and sentence[spos] == ' '])

def sent2features(sentence, suggestion):
	aligned = [(' '.join(sent), ' '.join(sug1)) for (sent, sug1, sug2) in alignment(sentence[:], suggestion[:], suggestion[:])]
	
	result = []
	result.append(len(suggestion))
	result.append(editdistance.eval(' '.join(sentence), ' '.join(suggestion))) #это error score, поставил edit distance т. к. пока не понимаю откуда брать error score
	result.append(ngram_score(suggestion))
	
	dict_words = [(sent, sug) for (sent, sug) in aligned if h.spell(sent)]
	OOVs = [(sent, sug) for (sent, sug) in aligned if not h.spell(sent)] #out of vocabulary words
	
	result.append(len(OOVs))
	
	OOV_corrections = sum([editdistance.eval(sent, sug) for (sent, sug) in OOVs])
	dict_word_corrections = sum([editdistance.eval(sent, sug) for (sent, sug) in dict_words])
	
	result.append(OOV_corrections)
	result.append(dict_word_corrections)
	
	#фича №8 -- number of corrections in capitalized words, нам неактуально
	
	distance_one = [1 for (sent, sug) in aligned if editdistance.eval(sent, sug) == 1]
	
	#фича №10 -- number of corrections by phonetic similarity, пока не разработана
	#фича №11 -- number of corrections by word lists, пока не разработана
	
	result.append(space_insertions(' '.join(sentence), ' '.join(suggestion)))
	result.append(space_deletions(' '.join(sentence), ' '.join(suggestion)))
	
	OOV_dictionary_particions = len([1 for (sent, sug) in OOVs if has_dictionary_partitions(sent)])
	result.append(OOV_dictionary_particions)
	
	result.append(morpho_score(word2pos(suggestion)))
	
	result.append(editdistance.eval(' '.join(sentence), ' '.join(suggestion)))
	
	my_w2v1 = Word2Vec.load('word2vecs/my_w2v1')
	result.append(my_w2v1.score([suggestion])[0])
	
	#последняя фича -- предложная модель, её мы ещё не разработали
	
	
	return np.array(result)

corpus2features = lambda sentences, suggestions: np.vstack([sent2features(sentence, suggestion) for sentence, suggestion in zip(sentences, suggestions)])

f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
corpus_lines = f.read().split('\r\n')
border = int(corpus_lines.pop(0))
marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border]]
misspelled = [(tokenize(query), tokenize(suggestion)) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>']

f = codecs.open("queries/queries1.csv", mode = "rU", encoding = "utf-8")
lines = f.read().split('\r\n')
f.close()
f = codecs.open("queries/queries2.csv", mode = "rU", encoding = "utf-8")
lines = lines + f.read().split('\r\n')
f.close()

lines = [line for line in lines if line != '']
lines = [line.rsplit(',', 1) for line in lines]
queries = [tokenize(processed_query) for (query, processed_query) in lines[border:]]

shuffle(misspelled)
border = len(misspelled) // 10 * 8

X_train = []
y_train = []
for (query, suggestion) in misspelled[:border]:
	winner = sent2features(query, suggestion)
	length = len(suggest(query))

	losers = corpus2features([query] * length,[sentence for sentence in suggest(query) if sentence != winner])
	X_train = np.array.vstack(result, winner - losers)
	y_train = np.hstack(y_train, np.repeat(1, losers.shape[1]))
	X_train = np.array.vstack(result, losers - winner)
	y_train = np.hstack(y_train, np.repeat(0, losers.shape[1]))

clf = LogisticRegression().fit(X_train, y_train)

def best_suggestion(sentence):
	candidates = suggest(sentence)
	candidates = sorted(candidates, key = lambda words: -clf.decision_function(sent2features(words)))
	return candidates[0]

result = []
for (query, suggestion) in misspelled[border:]
	result.append(best_suggestion(query))

f = open('result.txt', 'w')
f.write('\n'.join(result))
f.close()
print(corpus2features(['ленмна _NUMBER_'.split(), 'авроры _NUMBER_'.split()], ['ленмна _NUMBER_'.split(), 'авроры _NUMBER_'.split()]))
