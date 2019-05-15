# -*- coding: utf8 -*-
import csv
import re
from pyvi import ViTokenizer
def remove_noise(st):
    specials = r"[->\[\]\'\\\n\.\.\.\?\"\“\”\-\,\,>>():;!@#$%^&*?~/`<>{}=+-_]\ufeff"
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    st = str(st).lower()
    st = re.sub(emoji_pattern, ' ', st)
    st = re.sub(specials, ' ', st)
    st = re.sub('\s.\s', ' ', st)
    st = re.sub('\s+\s', ' ', st)
    st = st.strip()
    return st
viet_list = []
with open('TXT_Viet.txt', 'r', encoding='utf8') as file:
    for row in file:
        row = remove_noise(row)
        row = row.replace(' ', '_')
        viet_list.append(row)
viet_list = list(set(viet_list))
dauvaodautien=[]
with open('dauvaodautien.csv', 'r', encoding='utf8') as csvFile:
    for row in csvFile:
        dauvaodautien.append(row)
csvFile.close()
dauvaodautien = [i.replace('\n', "") for i in dauvaodautien]
dauvaodautien = [i for i in dauvaodautien if i != ""]
replace_list = {
            #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
            "👹": " rất chán ", "👻": " rất tốt ", "💃": " rất tốt ",'🤙': ' rất tốt ', '👍': ' rất tốt ',
            "💄": " rất tốt ", "💎": " rất tốt ", "💩": " rất tốt ","😕": " rất chán ", "😱": " rất chán ", "😸": " rất tốt ",
            "😾": " rất chán ", "🚫": "rất chán ",  "🤬": " rất chán ","🧚": " rất tốt ", "🧡": " rất tốt ",'🐶':' rất tốt ',
            '👎': ' rất chán ', '😣': ' rất chán ','✨': ' rất tốt ', '❣': ' rất tốt ','☀': ' rất tốt ',
            '♥': ' rất tốt ', '🤩': ' rất tốt ', 'like': ' rất tốt ', '💌': ' rất tốt ',
            '🤣': ' rất tốt ', '🖤': ' rất tốt ', '🤤': ' rất tốt ', ':(': ' rất chán ', '😢': ' rất chán ',
            '❤': ' rất tốt ', '😍': ' rất tốt ', '😘': ' rất tốt ', '😪': ' rất chán ', '😊': ' rất tốt ',
            '?': ' ? ', '😁': ' rất tốt ', '💖': ' rất tốt ', '😟': ' rất chán ', '😭': ' rất chán ',
            '💯': ' rất tốt ', '💗': ' rất tốt ', '♡': ' rất tốt ', '💜': ' rất tốt ', '🤗': ' rất tốt ',
            '^^': ' rất tốt ', '😨': ' rất chán ', '☺': ' rất tốt ', '💋': ' rất tốt ', '👌': ' rất tốt ',
            '😖': ' rất chán ', '😀': ' rất tốt ', ':((': ' rất chán ', '😡': ' rất chán ', '😠': ' rất chán ',
            '😒': ' rất chán ', '🙂': ' rất tốt ', '😏': ' rất chán ', '😝': ' rất tốt ', '😄': ' rất tốt ',
            '😙': ' rất tốt ', '😤': ' rất chán ', '😎': ' rất tốt ', '😆': ' rất tốt ', '💚': ' rất tốt ',
            '✌': ' rất tốt ', '💕': ' rất tốt ', '😞': ' rất chán ', '😓': ' rất chán ', '️🆗️': ' positive ',
            '😉': ' rất tốt ', '😂': ' rất tốt ', ':v': '  rất tốt ', '=))': '  rất tốt ', '😋': ' rất tốt ',
            '💓': ' rất tốt ', '😐': ' rất chán ', ':3': ' rất tốt ', '😫': ' rất chán ', '😥': ' rất chán ',
            '😃': ' rất tốt ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' rất tốt ',
            '😗': ' rất tốt ', '🤔': ' rất chán ', '😑': ' rất chán ', '🔥': ' rất chán ', '🙏': ' rất chán ',
            '🆗': ' rất tốt ', '😻': ' rất tốt ', '💙': ' rất tốt ', '💟': ' rất tốt ',
            '😚': ' rất tốt ', '❌': ' rất chán ', '👏': ' rất tốt ', ';)': ' rất tốt ', '<3': ' rất tốt ',
            '🌝': ' rất tốt ',  '🌷': ' rất tốt ', '🌸': ' rất tốt ', '🌺': ' rất tốt ',
            '🌼': ' rất tốt ', '🍓': ' rất tốt ', '🐅': ' rất tốt ', '🐾': ' rất tốt ', '👉': ' positive ',
            '💐': ' rất tốt ', '💞': ' rất tốt ', '💥': ' rất tốt ', '💪': ' rất tốt ',
            '💰': ' rất tốt ',  '😇': ' rất tốt ', '😛': ' rất tốt ', '😜': ' rất tốt ',
            '🙃': ' rất tốt ', '🤑': ' rất tốt ', '🤪': ' rất tốt ','☹': ' rất chán ',  '💀': ' rất chán ',
            '😔': ' rất chán ', '😧': ' rất chán ', '😩': ' rất chán ', '😰': ' rất chán ', '😳': ' rất chán ',
            '😵': ' rất chán ', '😶': ' rất chán ', '🙁': ' rất chán ',
            #Chuẩn hóa 1 số sentiment words/English words
            ':))': '  rất tốt ', ':)': ' rất tốt ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
            'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
            ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
            '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
            'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
            'he he': ' rất tốt ','hehe': ' rất tốt ','hihi': ' rất tốt ', 'haha': ' rất tốt ', 'hjhj': ' rất tốt ',
            ' lol ': ' rất chán ',' cc ': ' rất chán ','cute': u' dễ thương ','huhu': ' rất chán ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
            ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
            'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' rất tốt ', 'store': u' cửa hàng ',
            'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
            'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',}
tunoi=[]
with open('TuNoi.csv', 'r', encoding='utf8') as csvFile:
    for row in csvFile:
        tunoi.append(row)
csvFile.close()
tunoi=[i.lower() for i in tunoi]
tunoi=[i.replace('\n',"") for i in tunoi]
tunoi=[i.replace('.','\.') for i in tunoi]
tunoi=[i for i in tunoi if i!=""]
def transform(sentences):
    sentences = sentences.lower()
    for key,value in replace_list.items():
        if key in sentences:
            sentences = re.sub(key,value, sentences)
    changelist=[]
    for i in tunoi:
        matchObj=re.search(i,sentences,re.L)
        if matchObj:
            t=matchObj.group()
            changelist.append(t)
        else:
            continue
    for i in changelist:
        sentences=sentences.replace(i,'<break>')
    sentences=sentences.split('<break>')
    output=[i for i in sentences if i!="" ]
    return output
print('day la dau vao dau tien',dauvaodautien)
dauvao_old=[]
with open('filedauvao.csv', 'w') as csvFile:
    for i in dauvaodautien:
        dauvao_old=transform(i)
        print(dauvao_old)
        writer = csv.writer(csvFile)
        writer.writerows(map(lambda x: [x], dauvao_old))
csvFile.close()
dauvao_new=[]
with open('filedauvao.csv', 'r', encoding='utf8') as file:
    for row in file:
        row = remove_noise(row)
        row = ViTokenizer.tokenize(row)
        dauvao_new.append(row)
csvFile.close()
print(dauvao_new)
one_gram = []
for i in viet_list:
    t = i.count(' ')
    if t == 0:
        one_gram.append(i)
viet_tat_dist = {}
with open('TXT_VietTat.txt', 'r', encoding='utf8') as file:
    for row in file:
        i = row.split(':')
        viet_tat_dist[i[0]] = re.sub('\n', '', i[1])
neg_list_fixed = []
neg_removed_set = []
i = 0
with open('daura.csv', 'w', encoding='utf8') as file1:
    for row in dauvao_new:
        i += 1
        for word in row.split(' '):
            if word.count('_') < 1:
                word = remove_noise(word)
                if word not in one_gram:
                    if word in viet_tat_dist:
                        row = row.replace(word, viet_tat_dist[word])
                    else:
                        neg_removed_set.append(word)
                        row = row.replace(word, '')
        print('replaced ' + str(i) + row)
        row = remove_noise(row)
        if len(row) != 0:
            file1.write(ViTokenizer.tokenize(row) + '\n')
        # file1.write(row + '\n')
        # file2.write(row)
