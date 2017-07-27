import numpy as np

from weighted_levenshtein import lev

def coordinates(letter):
	if letter not in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
		raise Exception("%s not in acceptable alphabet" % letter)
		
	if letter == 'ё':
		letter = 'е'
	
	if letter in 'ячсмитьбю':
		x = 0
	elif letter in 'фывапролджэ':
		x = 1
	elif letter in 'йцукенгшщзхъ':
		x = 2
	
	if x == 0:
		y = 'ячсмитьбю'.find(letter) + 1
	elif x == 1:
		y = 'фывапролджэ'.find(letter) # + 0.5 (это актуально для компьютерной клавиатуры, а не для мобильника)
	elif x == 2:
		y = 'йцукенгшщзхъ'.find(letter)
	
	return (x, y)

def substitute_cost(letter1, letter2):
	try:
		(x1, y1) = coordinates(letter1)
		(x2, y2) = coordinates(letter2)
	except Exception:
		if letter1 == letter2:
			return 0
		else:
			return 100 #костыль
	dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
	return dist

insert_costs = np.ones((ord('ё') + 1))
insert_cost = lambda x: 5

substitute_costs = np.ones((ord('ё') + 1, ord('ё') + 1)) #ё -- последняя маленькая русская буква в кодировке
for letter1 in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
	for letter2 in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
		substitute_costs[ord(letter1), ord(letter2)] = substitute_cost(letter1, letter2)

delete_costs = np.ones((ord('ё') + 1))
delete_cost = lambda x: 5

def distance(word1, word2, insert_cost = insert_cost, substitute_cost = substitute_cost, delete_cost = delete_cost):
	if not word1:
		return len(word2)
	if not word2:
		return len(word1)	
	result = np.zeros((len(word1), len(word2)))
	for i in range(len(word1)):
		for j in range(len(word2)):
			if i == 0 or j == 0:
				result[i,j] = max([i, j])
			else:
				result[i,j] = min([
		result[i-1,j] + insert_cost(word1[i]),
		result[i,j-1] + delete_cost(word2[j]),
		result[i-1, j-1] + substitute_cost(word1[i], word2[j])
	])
	return result[-1, -1]
