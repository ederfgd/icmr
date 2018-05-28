# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import unicodedata
import codecs

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
#  the files in the input directory

def main():
    # --------------------------------------------------------------------------
    # ------- FLAGS ------------------------------------------------------------
    # --------------------------------------------------------------------------
    param_count = 8

    if(len(sys.argv) < param_count)
    {
        print("Não foram passados todos os parâmetros")
        print("Os parâmetros devem ser:\n- arquivo de treino\n- arquivo de saída do modelo\n- quantidade de 'epochs'\n- batch_size (32)\n- training_size (0,67)\n- embed_dim (128)\n- lstm_out (196)\n- Tokenizer max_features (2000)")
        return
    }

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    training_size = float(sys.argv[5])
    embed_dim = int(sys.argv[6])
    lstm_out = int(sys.argv[7])
    max_features = int(sys.argv[8])

    if training_size < 0 or training_size > 1:
        print("O tamanho do treino deve ser um valor em [0, 1]")
        return

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    with codecs.open(input_file, encoding='utf-8') as fd:
        # LEGENDA d in data:
        # d[:,0] = 'sentiment'
        # d[:,1] = 'text'
        data = np.asarray([[y.rstrip() for y in x.split('\t') ] for x in fd], dtype = object)
        sentiment = data[:,0]
        text = data[:,1]

    count_neg = len([d for d in data if d[0] == 'Negative'])
    count_pos = len([d for d in data if d[0] == 'Positive'])
    count_neu = len([d for d in data if d[0] == 'Neutral'])
    #print(len(data[ data[0] == 'Positive']))
    #print(len(data[ data[0] == 'Negative']))

    print("NEGATIVE ",count_neg)
    print("POSITIVE ",count_pos)
    print("NEUTRAL ",count_neu)

    tokenizer = Tokenizer(num_words = max_features, split = ' ')
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X)

    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    Y = pd.get_dummies(sentiment).values
    print(X.shape)
    print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = (1-training_size), random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    model.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, verbose = 2)

    validation_size = 1500

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):
        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("pos_acc", pos_correct/pos_cnt*100, "%")
    print("neg_acc", neg_correct/neg_cnt*100, "%")

    model.save(output_file)

if __name__ == "__main__":
    main()
