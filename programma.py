# -*- coding: utf-8 -*-

import sys
import codecs
import nltk
import math

#Funzione per il calcolo di frasi,token, media di token per frase e media di caratteri per parola
def analizzatoreTesto(frasi):
   testoToken = []
   testoPOS = []
   nFrasi = 0
   lenghtCorpus = 0
   lenghtFrase = 0.0
   for frase in frasi:
      tokens = nltk.word_tokenize(frase)
      tokensPOS = nltk.pos_tag(tokens)
      testoToken = testoToken + tokens
      testoPOS = testoPOS + tokensPOS
      nFrasi = nFrasi + 1
      lenghtCorpus = lenghtCorpus + len(tokens)
      lenghtFrase = lenghtFrase + len(frase)
   return testoToken, testoPOS, nFrasi, lenghtCorpus, lenghtFrase
   
#Funzione per ordinare i primi 10 elementi di un dizionario che vengono successivamente appesi in una lista
def dizionarioLista(dict,lista):
   for key in sorted(dict.items(),key = lambda x: x[1], reverse = True)[:10]:
      lista.append(key)
   return lista

#Funzione per il calcolo di POS, Probabilità massima di bigrammi e LMI
def sequenzaPOS(testoToken,testoPOS,lenghtCorpus):
   listaPOSV, listaPOSN = [], []
   listaBigram, listaProbMax, listaLMI = [], [], []
   dictPOS, dictPROB, dictMUTUAL= {}, {}, {}
   testoPOSDizio = set(testoPOS)
   for bigramma in testoPOS:
      if "NN" in bigramma[1]:
         listaPOSN.append(bigramma[1])
      if "VB" in bigramma[1]:
         listaPOSV.append(bigramma[1])
   rateoNV = len(listaPOSN)/len(listaPOSV)
   lunghezzaTesto = lenghtCorpus*1.0
   for bigramma in testoPOSDizio:
      freqBigram = testoPOS.count(bigramma)*1.0
      freqToken = testoToken.count(bigramma[0])*1.0
      freqPOS = testoPOS.count(bigramma[1])*1.0 
      d = { bigramma : freqBigram }
      d1 = { bigramma : freqBigram/freqToken }
      mi1 = ((freqBigram*lunghezzaTesto)/(freqToken*freqPOS+1.0))
      miFin = math.log(mi1,2)
      d2 = { bigramma : freqBigram*miFin }
      dictPOS.update(d)
      dictPROB.update(d1)
      dictMUTUAL.update(d2)
   dizionarioLista(dictPOS, listaBigram)
   dizionarioLista(dictPROB, listaProbMax)
   dizionarioLista(dictMUTUAL, listaLMI)
   return rateoNV, listaBigram, listaProbMax, listaLMI

#Funzione per il calcolo di hapax per ogni 1000 token e grandezza vocabolario
def hapaxVocabolario(testoToken, lenghtCorpus):
   n = 2000
   y = 1
   distribuzioneH, distr = [], []
   hapax = (len(set(testoToken[0:1000]))*1.0)/(lenghtCorpus*1.0)
   distribuzioneH.append(hapax)
   distr.append(hapax)
   while n < len(testoToken):
      hapax = len(set(testoToken[0:n]))
      distr.append(hapax)
      x = ((distr[y]-distr[y-1])*1.0)/(lenghtCorpus*1.0)
      distribuzioneH.append(x)
      n = n + 1000
      y = y + 1
   vocabolario = set(testoToken)
   grandezzaV = len(vocabolario)
   return grandezzaV, distribuzioneH

#Main dove leggo i file ed estraggo i dati 
def main(file1, file2):
   #Leggo in input i file
   input1 = codecs.open(file1, "r ","utf-8")
   input2 = codecs.open(file2, "r ","utf-8")
   raw1 = input1.read()
   raw2 = input2.read()
   sent_tokenizer= nltk.data.load('tokenizers/punkt/english.pickle')
  #Divido i file in frasi e conto i token con la funzione definita prima
   frasi1 = sent_tokenizer.tokenize(raw1)
   frasi2 = sent_tokenizer.tokenize(raw2)
   tokenTotali_1, POSTotali_1, NFrasiTesto_1, lunghezzaCorpus_1, lunghezzaFrasi_1 = analizzatoreTesto(frasi1)
   vocabolario_1, hapaxOgniN_1 = hapaxVocabolario(tokenTotali_1, lunghezzaCorpus_1)
   rateoPOSNV_1, maxBigramma_1, maxProb_1, LMI_1 = sequenzaPOS(tokenTotali_1, POSTotali_1, lunghezzaCorpus_1)
   tokenTotali_2, POSTotali_2, NFrasiTesto_2, lunghezzaCorpus_2, lunghezzaFrasi_2 = analizzatoreTesto(frasi2)
   vocabolario_2, hapaxOgniN_2 = hapaxVocabolario(tokenTotali_2, lunghezzaCorpus_2)
   rateoPOSNV_2, maxBigramma_2, maxProb_2, LMI_2 = sequenzaPOS(tokenTotali_2, POSTotali_2, lunghezzaCorpus_2)
   sys.stdout = open("output1.txt", "w")
   print "Primo libro analizzato:", file1, "\n", "La lunghezza di token è di:", lunghezzaCorpus_1, "tokens", "con:", NFrasiTesto_1, "Frasi\n","La lunghezza media della frasi è di:", lunghezzaCorpus_1/NFrasiTesto_1, "tokens\n","La media di caratteri per parola è di:", lunghezzaFrasi_1/lunghezzaCorpus_1, "parole\n","Il vocabolario è grande:", vocabolario_1, "parole\n"
   print "La distribuzione Hapax è di:", hapaxOgniN_1, "hapax\n"
   print "Il rateo di sostantivi e verbi è di:", rateoPOSNV_1, "\n"
   print "I 10 bigrammi POS più frequenti sono:", maxBigramma_1, "\n"
   print "I 10 bigrammi POS con probabilità condizionata massima:", maxProb_1, "\n"
   print "I 10 bigrammi POS con Local Mutual Information più alta:", LMI_1, "\n"
   print "Secondo libro analizzato:", file2, "\n","La lunghezza di token è di:", lunghezzaCorpus_2, "tokens", "con:", NFrasiTesto_2, "Frasi\n","La lunghezza media della frasi è di:", lunghezzaCorpus_2/NFrasiTesto_2, "tokens\n","La media di caratteri per parola è di:", lunghezzaFrasi_2/lunghezzaCorpus_2, "parole\n","Il vocabolario è grande:", vocabolario_2, "parole\n"
   print "La distribuzione Hapax è di:", hapaxOgniN_2, "hapax\n"
   print "Il rateo di sostantivi e verbi è di:", rateoPOSNV_2, "\n"
   print "I 10 bigrammi POS più frequenti sono:", maxBigramma_2, "\n"
   print "I 10 bigrammi POS con probabilità condizionata massima:", maxProb_2, "\n"
   print "I 10 bigrammi POS con Local Mutual Information più alta:", LMI_2, "\n"
   sys.stdout.close()
main(sys.argv[1],sys.argv[2])