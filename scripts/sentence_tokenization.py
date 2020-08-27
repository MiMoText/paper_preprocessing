import de_core_news_md

# Sprachmodell wird geladen
nlp = de_core_news_md.load()

# Hier Dateinamen ersetzen
with open("../data_in/Rieger_ Kapitel1.txt", encoding="utf-8") as file:
    text = file.read()
print(text)

doc = nlp(text)

with open("../data_out/sentences_de.txt", "w", encoding="utf-8") as file:
    for sent in doc.sents:
        print(sent.text)
        file.write(sent.text + "\n")

