import numpy as np
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from demo import *
from keras.utils import to_categorical



def load_glove():
    embeddings_index = {}
    with open("./glove.6B/glove.6B.300d.txt") as f:
        for line in f:
            values = line.split(' ')
            word = values[0] ## The first entry is the word
            coefs = np.asarray(values[1:], dtype='float32') ## These are the vectors representing the embedding for the word
            embeddings_index[word] = coefs
    print('GloVe data loaded')
    return embeddings_index



def featurize_model_outputs(model_predictions, img_name):

    embeddings_index = load_glove()
    texts=[]

    MAX_SEQUENCE_LENGTH = 0
    MAX_NB_WORDS = 10000
    EMBEDDING_DIM = 300

    texts = model_predictions['frame']
    # labels=[x for x in range(len(texts))]
    MAX_SEQUENCE_LENGTH=len(texts)
    # for x in lst_of_words:
    #
    # for x,y in model_predictions.items():
    #
    #     labels.append(x)
    #     texts.append(y)
    #     length = len(y)
    #     if length>MAX_SEQUENCE_LENGTH:
    #         MAX_SEQUENCE_LENGTH=length


    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    #data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    image_to_features={}
    embedding_matrix = np.zeros((len(texts), EMBEDDING_DIM))

    for x in range(len(texts)):
        embedding_vector = embeddings_index.get(texts[x])
        if embedding_vector is not None:
            embedding_matrix[x]=embedding_vector

    image_to_features[img_name]= embedding_matrix
    print(img_name, " has embedding shape: ", embedding_matrix.shape)

    return image_to_features
