from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Codebeispiel stammt von hier: https://huggingface.co/transformers/main_classes/pipelines.html
# Named entity recognition pipeline, passing in a specific model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("bert-base-german-cased")
my_pipeline = pipeline('ner', model=model, tokenizer="bert-base-german-cased")
my_nes = my_pipeline("Das ist ein Test. Dieser Test wurde in Trier durchgeführt.")
print(my_nes)

# how to train model on downstream-task?
# und warum ist das hier notwendig?
# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py


# relevante pretrained Modelle (enthält keine Community-Modelle):
# Übersicht Modelle: https://huggingface.co/transformers/main_classes/pipelines.html
# Deutsch
    # bert-base-german-cased
    # bert-base-german-dbmdz-cased

# Französisch
    # XLM
    # CamemBERT
    # Flaubert (heterogenes Korpus - gut)

# Multiling:
    # bert-base-multilingual-cased
    # whole world?
