# -*- coding: utf-8 -*-

import sys
import codecs
import nltk
import re

# Funzione per l'analisi delle entità nominate (NE), crea la lista dei primi 10 NE più frequenti
def analizzatoreTesto(frasi):
    corpus = []
    testoNER = []
    lista10NER = []
    dictNER = {}
    x = 0
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        corpus = corpus + tokens
        tokensPOS = nltk.pos_tag(tokens)
        testoAnalizzato = nltk.ne_chunk(tokensPOS)
        for nodo in testoAnalizzato:
            NE = " "
            if hasattr(nodo, "label"):
                if nodo.label() in ["PERSON"]:
                    for partNE in nodo.leaves():
                        if "NNP" in partNE[1] or "NNPS" in partNE[1]:
                            if x == 0:
                                NE = partNE[0]
                                x = x + 1
                            else:
                                NE = NE + " " + partNE[0]
                    x = 0
                    testoNER.append(NE)
    lista10NER = ordinaLista(testoNER)
    return lista10NER, corpus

# Funzione per estrazione di determinati NE passati dai parametri, crea una lista di NE corrispondente
def selezione(NE, nodo, Tag, variab, freq):
    if nodo.label() in [Tag]:
        for partNE in nodo.leaves():
            if variab == 0:
                NE = partNE[0]
                variab = variab + 1
            else:
                NE = NE + " " + partNE[0]
        variab = 0
        freq.append(NE)

# Funzione per il calcolo del modello markoviano 0, prendo il corpus e restituisco la probabilità analizzata
def markov(fraseTok, corpus):
    global probTot
    probTot = 0
    if 7 < len(fraseTok) < 13:
        probTot = 1
        i = len(fraseTok)
        while i > 0:
            conto = fraseTok[i - 1]
            probTot = (float(probTot)) * ((float(corpus.count(conto))) / (float(len(corpus))))
            i = i - 1

# Funzione che estrae i dati richiesti dai task del progetto, prende in input le frasi dove compaiono i primi 10 NE estratti
def estrazione(frasi, corpus):
    global lista10NERL, lista10NERP, lista10SOS, lista10Verb, fraseMax
    y, z = 0, 0
    probMax = 0
    listaPOS = []
    freqLoc, freqPers, freqSos, freqVerb = [], [], [], []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        listaPOS = listaPOS + tokensPOS
        testoAnalizzato = nltk.ne_chunk(tokensPOS)
        for nodo in testoAnalizzato:
            NE = " "
            if hasattr(nodo, "label"):
                selezione(NE, nodo, "LOCATION", y, freqLoc)
                selezione(NE, nodo, "PERSON", z, freqPers)
        markov(tokens, corpus)
        if probTot > probMax:
            probMax = probTot
            fraseMax = frase
            fraseMax = fraseMax.encode('utf-8')
    for bigramma in listaPOS:
        if "NN" in bigramma[1]:
            freqSos.append(bigramma[0])
        if "VB" in bigramma[1]:
            freqVerb.append(bigramma[0])
    lista10NERL = ordinaLista(freqLoc)
    lista10NERP = ordinaLista(freqPers)
    lista10SOS = ordinaLista(freqSos)
    lista10Verb = ordinaLista(freqVerb)

# Funzione per la ricerca di giorni mesi o date usando le regex
def dateFrasi(frase):
    iGiorni = re.findall('\s(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\s',frase)
    iMesi = re.findall('\s(?:Jan|Febr)uary|Ma(?:rch|y)|A(?:ugust|pril)|Ju(?:ne|ly)|(?:Septem|Octo|Novem|Decem)ber\s',frase)
    lAnno = re.findall('\s[0-3]?\d[-/][01]?\d[-/][0-2]?\d?\d?\d\s',frase)
    return iGiorni,iMesi,lAnno    

# Funzione per ordinare i primi 10 elementi di n lista
def ordinaLista(testoFreq):
    dict = {}
    listaOrdinare = set(testoFreq)
    lista10TOP = []
    for name in listaOrdinare:
        freqNome = testoFreq.count(name)
        d = {name: freqNome}
        dict.update(d)
    for key in sorted(dict.items(),key = lambda x: x[1], reverse=True)[:10]:
        lista10TOP.append(key)
    return lista10TOP

# Funzione che stampa i task richiesti del progetto, per i primi 10 NE estratti
def frasiNER(frasi, NER, corpus):
    i = 0
    listaGiorno, listaMesi, listaDate = [], [], []
    listaFrasiTempo, listaFrasi = [], []
    lenghtMax, lenghtMin = 0, 200000
    maxFrase, minFrase = " ", " "
    while i < 10:
        print "Per il nome", NER[i][0], "ci sono queste frasi:"
        for frase in frasi:
            token = nltk.word_tokenize(frase)
            lenghtFrase = len(token)
            if NER[i][0] in frase:
                giorno,mese,anno = dateFrasi(frase)
                if (giorno != []):
                    listaGiorno.append(giorno)
                if (mese != []):
                    listaMesi.append(mese)
                if (anno != []):
                    listaDate.append(anno) 
                listaFrasiTempo.append(frase)
                if lenghtFrase > lenghtMax:
                    lenghtMax = lenghtFrase
                    maxFrase = frase
                    maxFrase = maxFrase.encode('utf-8')
                if lenghtFrase < lenghtMin:
                    lenghtMin = lenghtFrase
                    minFrase = frase
                    minFrase = minFrase.encode('utf-8')
        listaFrasi = set(listaFrasiTempo)
        estrazione(listaFrasi, corpus)
        for element in listaFrasi:
            element.encode('utf-8')
        print listaFrasi
        print
        print "Le frasi minori e maggiori di", NER[i][0], "sono le seguenti:"
        print "La maggiore:\n", maxFrase
        print "La minore:\n", minFrase
        print
        print "Lista dei 10 nomi di luogo più frequenti:", lista10NERL
        print "Lista dei 10 nomi di persona più frequenti:", lista10NERP
        print "Lista dei 10 sostantivi più frequenti:", lista10SOS
        print "Lista dei 10 verbi più frequenti:", lista10Verb
        print
        print "Date, Mesi, Giorni trovati:\n", "Giorni:", listaGiorno, "\nMesi:", listaMesi, "\nDate:", listaDate
        print
        print "Frase con probabilità più alta:", fraseMax
        print
        listaFrasiTempo = []
        lenghtMax, lenghtMin = 0, 200000
        maxFrase, minFrase = " ", " "
        i = i + 1

# Main dove leggo i file ed estraggo i dati
def main(file1, file2):
    # Leggo in input i file
    input1 = codecs.open(file1, "r ", "utf-8")
    input2 = codecs.open(file2, "r ", "utf-8")
    raw1 = input1.read()
    raw2 = input2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Divido i file in frasi e conto i token con la funzione definita prima
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    Ner, Corpus = analizzatoreTesto(frasi1)
    Ner2, Corpus2 = analizzatoreTesto(frasi2)
    sys.stdout = open("output2.txt", "w")
    print "Dati del libro", file1, ":"
    frasiNER(frasi1, Ner, Corpus)
    print "Dati del libro", file2, ":"
    frasiNER(frasi2, Ner2, Corpus2)
    sys.stdout.close()

main(sys.argv[1], sys.argv[2])
