# Änderung zu bert2408.py: Verwendung anderer Tokenizer, um Problem der falschen Trennung zu begegnen

# hier wird "xlm-roberta-large-finetuned-conll03-german" verwendet
# erster Versuch war mit "bert-base-german-cased"
# --> random results + Meldung: "Some weights of the model checkpoint at bert-base-german-cased
# were not used when initializing BertForTokenClassification"
# Grund: "bert-base-german-cased" ist nicht für token classification vortrainiert

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-german")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-german")

with open("../data_in/Rieger_ Kapitel1.txt", encoding="utf-8") as file:
    text = file.read()
print(text)

my_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
my_nes = my_pipeline(text)
print(my_nes)

