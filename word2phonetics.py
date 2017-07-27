import re

def word2phonetics(word):
	word = str(re.sub(r'т[ъь]?с', 'ц', word))
	result = ''
	for (counter, letter) in enumerate(word):
		if letter == ' ':
			result += ' '
		elif letter not in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
			result += '0'
		elif letter in 'ьъ':
			pass
		elif letter in 'аоыуяиеёюяэ' and counter != 0 and word[counter - 1] in 'ьъщчй':
			result += '3'
		elif letter in 'аоыуяиеёюяэ' and counter != 0 and word[counter - 1] in 'шжц':
			result += '1'
		elif letter in 'аоыуя':
			result += '1'
		elif letter in 'иеёюяэ':
			result += '3'
		elif letter in 'бп':
			result += '5'
		elif letter in 'вф':
			result += '6'
		elif letter in 'тд':
			result += '7'
		elif letter in 'кгх':
			result += '8'
		elif letter == 'л':
			result += '9'
		elif letter == 'р':
			result += 'A'
		elif letter == 'м':
			result += 'B'
		elif letter == 'н':
			result += 'C'
		elif letter in 'зс':
			result += 'D'
		elif letter == 'й':
			result += 'E'
		elif letter in 'щч':
			result += 'F'
		elif letter in 'жш':
			result += 'G'
		elif letter == 'ц':
			result += 'H'
	
	result = re.sub(r'0{2,}|1{2,}|2{2,}|3{2,}|4{2,}|5{2,}|6{2,}|7{2,}|8{2,}|9{2,}|A{2,}|B{2,}|C{2,}|D{2,}|E{2,}|F{2,}|G{2,}|H{2,}', lambda match: match.group(0)[0] if len(match.group(0)) > 0 else macth.group(0), result)
	return result
