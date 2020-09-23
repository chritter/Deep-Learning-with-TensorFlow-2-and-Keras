import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

max_len = 200
n_words = 10000
dim_embedding = 256
EPOCHS = 20
BATCH_SIZE =500

def load_data():
	#load data
	# Reviews have been preprocessed, and each review is encoded as a list of word indexes (integers).
	# num_words allow to keep only the n_words=10000 most frequent words. all other appear as `oov_char` (0 index) value in the dataset
	(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)
	# Pad sequences with max_len
	X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
	X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
	return (X_train, y_train), (X_test, y_test)

def build_model():
	model = models.Sequential()
	#Input - Emedding Layer
	# the model will take as input an integer matrix of size (batch, input_length)
	# the model will output dimension (input_length, dim_embedding)
    # the largest integer in the input should be no larger
    # than n_words (vocabulary size).

	embedding_layer = layers.Embedding(n_words, 
		dim_embedding, input_length=max_len)

	model.add(embedding_layer)

	model.add(layers.Dropout(0.3))

	model.add(layers.Conv1D(256, 3, padding='valid', 
		activation='relu'))

	#takes the maximum value of either feature vector from each of the n_words features
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	return model

(X_train, y_train), (X_test, y_test) = load_data()
model=build_model()
model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

score = model.fit(X_train, y_train,
 epochs= EPOCHS,
 batch_size = BATCH_SIZE,
 validation_data = (X_test, y_test)
)

score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
