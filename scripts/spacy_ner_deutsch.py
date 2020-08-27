import de_core_news_md

# Sprachmodell wird geladen
nlp = de_core_news_md.load()

# Hier Dateinamen ersetzen
with open("../data_in/rieger.txt", encoding="utf-8") as file:
    text = file.read()
print(text)

doc = nlp(text)

with open("../data_out/spacy_de.txt", "w", encoding="utf-8") as file:
    for entity in doc.ents:
        print(entity.text, entity.label_)
        file.write(entity.text + ";" + entity.label_ + "\n")