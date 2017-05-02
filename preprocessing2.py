import nltk

text="Dan's parents were overweight.,Dan was overweight as well.,The doctors told his parents it was unhealthy.,His parents understood and decided to make a change.,They got themselves and Dan on a diet.".split(',')

print [sen.lower() for sen in text]

print [nltk.word_tokenize(sen) for sen in text]

wnl=nltk.WordNetLemmatizer()

print [wnl.lemmatize(sen) for sen in text]

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
lancaster_stemmer.stem('presumably')

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem('presumably')

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
snowball_stemmer.stem('presumably')

# 典型特例，Excited，Lying。目前可能snowball_stemmer是很合适的。
>>> print [lancaster_stemmer.stem(sen) for sen in text]  #变小写
["dan's parents were overweight.", 'dan was overweight as well.', 'the doctors told his parents it was unhealthy.', 'his parents understood and decided to make a change.', 'they got themselves and dan on a diet.']
>>> print [porter_stemmer.stem(sen) for sen in text]  #不会变小写
[u"Dan's parents were overweight.", u'Dan was overweight as well.', u'The doctors told his parents it was unhealthy.', u'His parents understood and decided to make a change.', u'They got themselves and Dan on a diet.']
>>> print [snowball_stemmer.stem(sen) for sen in text] #变小写
[u"dan's parents were overweight.", u'dan was overweight as well.', u'the doctors told his parents it was unhealthy.', u'his parents understood and decided to make a change.', u'they got themselves and dan on a diet.']



#nltk version
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

text="Dan's parents were overweight.,Dan was overweight as well.,The doctors told his parents it was unhealthy.,His parents understood and decided to make a change.,They got themselves and Dan on a diet.".split(',')


for sen in text:
    token_list=nltk.word_tokenize(sen[:-1])
    tagged_sen=nltk.pos_tag(token_list)
    new_sen=[]
    for (word,tag) in tagged_sen:
        if tag[0]=='V':
            lemma_word=wordnet_lemmatizer.lemmatize(word,pos='v')
        else:
            lemma_word=wordnet_lemmatizer.lemmatize(word)
        stem_word=snowball_stemmer.stem(lemma_word)
        new_sen.append(stem_word)
    print " ".join(new_sen)


# stanford version
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import StanfordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = StanfordTokenizer()
eng_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')


text="Dan's parents were overweight.,Dan was overweight as well.,The doctors told his parents it was unhealthy.,His parents understood and decided to make a change.,They got themselves and Dan on a diet.".split(',')


for sen in text:
    token_list=tokenizer.tokenize(sen[:-1])
    tagged_sen=eng_tagger.tag(token_list)
    new_sen=[]
    for (word,tag) in tagged_sen:
        # print word,tag
        if tag[0]=='V':
            
            lemma_word=wordnet_lemmatizer.lemmatize(word,pos='v')
        else:
            lemma_word=wordnet_lemmatizer.lemmatize(word)
        stem_word=snowball_stemmer.stem(lemma_word)
        new_sen.append(stem_word)
    print " ".join(new_sen)
