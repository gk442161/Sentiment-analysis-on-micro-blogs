import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def Tokenization(EMBEDDING_MATRIX_PATH):
    MAX_NUM_WORDS = 20000
    oov_tok = "<OOV>"  # OUT OF VOCABLARY
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token=oov_tok)
    # TODO insert dataset in tokenizer to fit it
    embedding_df= pd.read_pickle(EMBEDDING_MATRIX_PATH)
    tokenizer.fit_on_texts(embedding_df.columns)
    del embedding_df
    return tokenizer


def sentiment_predict(trained_model,input_data,tokenizer,):
  TRUNCAT_TYPE = 'post'
  PADDING_TYPE = 'post'
  MAX_SEQUENCE_LENGTH = 1000   
  text_seq = tokenizer.texts_to_sequences(input_data)
  text_padded = pad_sequences(text_seq, maxlen=MAX_SEQUENCE_LENGTH, padding=PADDING_TYPE, truncating=TRUNCAT_TYPE)
  pred_val=trained_model.predict(text_padded)

  predicted_val = np.asarray(pred_val)
  pred_class = np.argmax(predicted_val, axis=1)
  pred_class_prob = np.max(predicted_val,axis =1)

  pred_val=np.array(list(map(lambda x: x.argmax(),pred_val)))
  
#   print('predicted sentiment score : ', pred_val)
#   annotator = input('Sentiment score by You in the range [0-10] where 0 is verry negative and 10 is very positive: ')
  return pred_val[0],pred_class_prob,pred_class
  # return 'helo'

