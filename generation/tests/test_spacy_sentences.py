# Follow https://stackoverflow.com/questions/46290313/how-to-break-up-document-by-sentences-with-spacy
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
from spacy.lang.en import English

nlp_simple = English()
nlp_simple.add_pipe('sentencizer')
# nlp_simple.add_pipe(nlp_simple.create_pipe('sentencizer'))

nlp_better = spacy.load('en_core_web_sm')


text = 'My first birthday was great. My 2. was even better.'

for nlp in [nlp_simple, nlp_better]:
    for i in nlp(text).sents:
        print(i)
    print('-' * 20)


text = "For e.g. 1,2,3 etc. Also you could try this sentence."
print(nlp_better(text).sents)
print(f" Length: {len(list(nlp_better(text).sents))}, sentence list: {list(nlp_better(text).sents)}")

text = "For eg. 1,2,3 etc. and all. Also you could try this sentence."
print(nlp_better(text).sents)
print(list(nlp_better(text).sents))