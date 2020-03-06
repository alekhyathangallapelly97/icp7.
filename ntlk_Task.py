import requests
from bs4 import BeautifulSoup
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

#Downloading all depedencies of NLTK
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#getting the web content
html_doc = requests.get('https://en.wikipedia.org/wiki/Google')

#scrapping content using BeautifulSoup
soup = BeautifulSoup(html_doc.content, 'html.parser')


page = soup.get_text("\n")

# list = []
# for link in soup.find_all('a'):
#     list.append(link.get('href'))
#
# print(list);

# # Writing contents to a file.
with open('input.txt', 'a') as f:
    f.write(page)

input = open('input.txt', 'r')

with open('output.txt', 'w') as f:

    for statement in input:

        #Tokenization
        tokens = nltk.word_tokenize(statement)
        f.write("Tokenization: \n")
        f.write(str(tokens))
        print(tokens)

        #parts of speech
        pos = nltk.pos_tag(tokens)
        f.write("Parts of Speech: \n")
        f.write(str(pos))
        print(pos)

        #Stemming
        stemmer = PorterStemmer()
        f.write("Stemming: \n")
        for token in tokens:
            stemming = stemmer.stem(token)
            f.write(str(stemming))
            print(stemming)

        #Lemmatizaiton
        lemmatizer = WordNetLemmatizer()
        f.write("Lemmatization: \n")
        for token in tokens:
            lemmatization = lemmatizer.lemmatize(token)
            f.write(str(lemmatization))
            print(lemmatization)

        #Named Entity Recognition
        ner = ne_chunk(pos_tag(wordpunct_tokenize(statement)))
        f.write("Named Entity Recognition: \n")
        f.write(str(ner))
        print(ner)

        #Trigrams
        f.write("Trigrams: \n")
        trigrams = nltk.trigrams(statement.split())
        for item in trigrams:
            f.write(str(item))
            print(item)