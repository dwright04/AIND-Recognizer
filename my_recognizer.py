import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for i in range(test_set.num_items):
        bestLogL = -float('inf')
        bestGuess = None
        probDict = {}
        for j,key in enumerate(models.keys()):
            X, lengths = test_set.get_item_Xlengths(i)
            try:
                logL = models[key].score(X, lengths)
            except ValueError:
                logL = -float('inf')
            probDict[key] = logL
            if logL > bestLogL:
                bestLogL = logL
                bestGuess = key
        probabilities.append(probDict)
        guesses.append(bestGuess)
        #guesses.append(test_set.wordlist[0])
    return probabilities, guesses
    #raise NotImplementedError
