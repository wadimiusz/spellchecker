import codecs, collections, re
from Levenshtein import distance, median
f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
lines = f.read().split('\r\n')
f.close()
number_of_line = int(lines.pop(0))
f = codecs.open("all words.txt", mode = "r", encoding = "utf-8")
words = f.read().split('\r\n')
f.close()
f = codecs.open("good words.txt", mode = "r", encoding = "utf-8")
good_words = set(f.read().split('\r\n'))
f.close()
f = codecs.open("good words 3.txt", mode = "r", encoding = "utf-8")
good_words.update(f.read().split('\r\n'))
f.close()
good_words = list(good_words)
print(len(good_words))

c = collections.Counter()

for word in words:
	c[word] += 1

pattern = re.compile('[А-Яа-я]+')

#собираем уже известные ответы
right_answers = {}
for line in lines[:number_of_line-1]:
	line_split = line.split(',')
	question = line_split[-3]
	answer = line_split[-2]
	right_answers[question] = answer

def dictionary():
	dict_words = set(re.findall('[А-Яа-яЁё]+', ' '.join([x.split(',')[-2] for x in lines[0:number_of_line]])))
	dict_words = list(dict_words)
	print(len(dict_words))
	f = codecs.open("good words.txt", mode = "w", encoding = "utf-8")
	for word in dict_words:
		f.write("%s\n" % (word))
	f.close()

def dictionary_update(l):
	global good_words
	good_words += l
	good_words = set(good_words)
	good_words = list(good_words)

def correction(s, n_good, n_bad):
	if pattern.match(s):
		if s in good_words:
			return s
		elif len([x for x in good_words if distance(x, s) <= n_good]) > 0:
			return median([x for x in good_words if distance(x, s) <= n_good])
		elif len([x for x in words if distance(x, s) <= n_bad]) > 0:
			corrs = [x for x in c.most_common() if distance(x[0], s) <= n_bad]
			corrs1 = [x[0] for x in corrs]
			corrs2 = [x[1] for x in corrs]
			return median(corrs1, corrs2)
		else:
			return s
	else:
		return s

while True:
	if number_of_line % 10 == 0:
		print(number_of_line)
		f = codecs.open("corpus2.csv", mode = "w", encoding = "utf-8-sig")
		f.write('\r\n'.join([str(number_of_line)] + lines))
		f.close()
	current_line = lines[number_of_line]
	current_part = current_line.split(',')[-1]
	print('\n' + current_part)
	hypothesis = ' '.join([correction(x, 2, 3) for x in current_part.split(' ')])
	def is_different(s1, s2): 
		if s1 == s2:
			return 'ОК'
		else:
			return 'исправлено'
	if current_part in right_answers:
		correct_answer = right_answers[current_part]
		print('Запрос уже встречался, записываем ответ %s' % correct_answer)
	else:
		responce = input("Гипотеза: %s (%s)\n" % (hypothesis, is_different(current_part, hypothesis)))
		if responce in ['y', 'yes', 'д', 'да', 'ok', 'ок', 'jr', 'щл']:
			correct_answer = current_part
		elif responce in ['h', 'г', "hyp", "гип"]:
			correct_answer = hypothesis
		elif 'quit' in responce or 'dict' in responce:
			dictionary()
			break
		else:
			correct_answer = responce
	dictionary_update(re.findall('[А-Яа-яЁё]+', correct_answer))
	additional_part = ',%s,%d' % (correct_answer, int(current_part != correct_answer))
	lines[number_of_line] += additional_part
	right_answers[current_part] = correct_answer
	number_of_line += 1
lines = [str(number_of_line)] + lines
f = codecs.open("corpus2.csv", mode = "w", encoding = "utf-8-sig")
f.write('\r\n'.join(lines))
f.close()
