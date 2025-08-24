# -*- coding: utf-8 -*-


file_path = "data/fra.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Nombre de lignes : {len(lines)}")
print("Premières lignes :", lines[:5])


# Séparer uniquement les deux premières colonnes (anglais → français).
# Ignorer la partie licence.

source_texts = []
target_texts = []

for line in lines:
    parts = line.strip().split("\t")
    if len(parts) >= 2:
        source_texts.append(parts[0])
        target_texts.append(parts[1])

print(f"Exemple source : {source_texts[0]}")
print(f"Exemple target : {target_texts[0]}")
print(f"Nombre de paires : {len(source_texts)}")

# préparer les données pour le modèle Seq2Seq



# Ajouter tokens start et end
target_texts = ['\t' + text + '\n' for text in target_texts]

# Créer un vocabulaire unique pour source et target
source_characters = sorted(list(set("".join(source_texts))))
target_characters = sorted(list(set("".join(target_texts))))

num_encoder_tokens = len(source_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in source_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print(f"Nombre de tokens source : {num_encoder_tokens}")
print(f"Nombre de tokens target : {num_decoder_tokens}")
print(f"Longueur max source : {max_encoder_seq_length}")
print(f"Longueur max target : {max_decoder_seq_length}")


#-----------------------------------------------------------------------------
# tokenizer pour les phrases source (français) et target (anglais) 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenizer pour le français
source_tokenizer = Tokenizer(char_level=True)  # caractère par caractère
source_tokenizer.fit_on_texts(source_texts)
source_vocab_size = len(source_tokenizer.word_index) + 1

# Tokenizer pour l'anglais
target_tokenizer = Tokenizer(char_level=True)
target_tokenizer.fit_on_texts(target_texts)
target_vocab_size = len(target_tokenizer.word_index) + 1

# Convertir les phrases en séquences d'entiers
source_sequences = source_tokenizer.texts_to_sequences(source_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Padding pour avoir des séquences de même longueur
max_source_len = max(len(seq) for seq in source_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

source_sequences = pad_sequences(source_sequences, maxlen=max_source_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')



#-----------------------------------------------------------------------------
# modèle Seq2Seq avec Embedding

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

embedding_dim = 128
latent_dim = 256  # taille des états cachés LSTM

# Encodeur
encoder_inputs = Input(shape=(max_source_len,))
encoder_embedding = Embedding(input_dim=source_vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Décodeur
decoder_inputs = Input(shape=(max_target_len,))
decoder_embedding = Embedding(input_dim=target_vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modèle
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


#-----------------------------------------------------------------------------
# sorties pour le décodeur
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Décaler les séquences
decoder_input_sequences = target_sequences[:, :-1]
decoder_target_sequences = target_sequences[:, 1:]

# Pad pour obtenir la même longueur max que le modèle
decoder_input_sequences = pad_sequences(decoder_input_sequences, maxlen=max_target_len, padding='post')
decoder_target_sequences = pad_sequences(decoder_target_sequences, maxlen=max_target_len, padding='post')

# Ajouter la dimension pour sparse_categorical_crossentropy
decoder_target_sequences = np.expand_dims(decoder_target_sequences, -1)



#-----------------------------------------------------------------------------
# Entrainer le modele 
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


batch_size = 64
epochs = 30

history = model.fit(
    [source_sequences, decoder_input_sequences],
    decoder_target_sequences,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)






import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -----------------------------------------------------------------------------
# Sauvegarder le modèle entraîné
model.save("models/seq2seq_fra_en.h5")



# Sauvegarder les tokenizers
import pickle

with open('models/source_tokenizer.pkl', 'wb') as f:
    pickle.dump(source_tokenizer, f)
print("Tokenizer source sauvegardé sous source_tokenizer.pkl")

with open('models/target_tokenizer.pkl', 'wb') as f:
    pickle.dump(target_tokenizer, f)
print("Tokenizer target sauvegardé sous target_tokenizer.pkl")


#------------------------------------------------------------------------------
# Inférence 

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import pickle

latent_dim = 256  # même que ton modèle

# ---- Charger modèle et tokenizers ----
model = load_model("models/seq2seq_fra_en.h5")
with open('models/source_tokenizer.pkl', 'rb') as f:
    source_tokenizer = pickle.load(f)
with open('models/target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

max_source_len = model.input_shape[0][1]
max_target_len = model.input_shape[1][1]

# ---- Encodeur pour inférence ----
encoder_inputs = model.input[0]  # entrée de l'encodeur
encoder_lstm = model.layers[3]
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm.output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# ---- Décodeur pour inférence ----
decoder_inputs_single = Input(shape=(1,))  # 1 token à la fois
decoder_embedding = model.layers[4](decoder_inputs_single)
decoder_lstm = model.layers[5]
decoder_dense = model.layers[6]

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs, state_h, state_c]
)

# ---- Fonction pour traduire ----
def translate_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['\t']
    decoded_sentence = ''
    
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = [char for char, index in target_tokenizer.word_index.items()
                        if index == sampled_token_index][0]
        decoded_sentence += sampled_char
        
        if sampled_char == '\n' or len(decoded_sentence) > max_target_len:
            stop_condition = True
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return decoded_sentence

# ---- Exemple de test ----
test_sentence = "Bonjour."
seq = source_tokenizer.texts_to_sequences([test_sentence])
seq = pad_sequences(seq, maxlen=max_source_len, padding='post')
translation = translate_sequence(seq)

print("Français :", test_sentence)
print("Anglais :", translation)




#------------------------------------------------------------------------------
# Affichage loss accuracy
import matplotlib.pyplot as plt

import pickle

# Sauvegarder
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Plus tard, tu peux le recharger
with open("history.pkl", "rb") as f:
    history_data = pickle.load(f)

# Récupérer les métriques depuis history
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history.get('accuracy', None)  # peut ne pas être présent
val_accuracy = history.history.get('val_accuracy', None)

epochs = range(1, len(loss) + 1)

# --- Affichage Loss ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Loss durant l\'apprentissage')
plt.xlabel('Épochs')
plt.ylabel('Loss')
plt.legend()

# --- Affichage Accuracy (si présente) ---
if accuracy:
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
    plt.title('Accuracy durant l\'apprentissage')
    plt.xlabel('Épochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()



