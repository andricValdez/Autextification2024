
import nltk
import math

def get_ngram_freq(text):    
    unigrams = nltk.word_tokenize(text)
    bigrams = nltk.bigrams(unigrams)

    unigram_freq = nltk.FreqDist(unigrams)
    bigram_freq = nltk.FreqDist(bigrams)
    
    #for k,v in fdist.items():
    #    print(k,v)

    return unigram_freq, bigram_freq



def pmi(words, unigram_freq, bigram_freq):
  prob_word1 = unigram_freq[words[0]] / float(sum(unigram_freq.values()))
  prob_word2 = unigram_freq[words[1]] / float(sum(unigram_freq.values()))
  prob_word1_word2 = bigram_freq[words] / float(sum(bigram_freq.values()))
  return math.log(prob_word1_word2/float(prob_word1*prob_word2),2) 


def main():
        
    text = 'This is a foo bar sentence .\nI need multi-word expression from this text file.\nThe text file is messed up , I know you foo bar multi-word expression thingy .\n More foo bar is needed , so that the text file is populated with some sort of foo bar bigrams to extract the multi-word expression .'
    unigram_freq, bigram_freq = get_ngram_freq(text) 

    for words, freq in bigram_freq.items():
        pmi_val = pmi(words, unigram_freq, bigram_freq)
        print(f'{words}, {freq}, {pmi_val}')


if __name__ == '__main__':
    main()