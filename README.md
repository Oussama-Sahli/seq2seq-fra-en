Seq2Seq Français → Anglais


### Description

Je me suis lancé dans un projet de traduction automatique utilisant un modèle Seq2Seq basé sur LSTM avec TensorFlow/Keras.
Le but était de créer un modèle capable de traduire des phrases simples du français vers l’anglais, entraîné sur le dataset fra.txt.

Étapes du projet
1. Préparation des données

J’ai récupéré le fichier fra.txt contenant des paires français → anglais.

J’ai extrait uniquement les deux premières colonnes (ignorer la licence).

J’ai ajouté des tokens de début \t et de fin \n pour les phrases cibles.

J’ai créé un vocabulaire pour le français (source) et l’anglais (target) caractère par caractère.

2. Tokenization

J’ai utilisé Tokenizer(char_level=True) pour convertir les phrases en séquences d’entiers.

J’ai appliqué du padding pour que toutes les séquences aient la même longueur.

3. Construction du modèle Seq2Seq

Encodeur : Embedding → LSTM

Décodeur : Embedding → LSTM → Dense (softmax)

La taille des états cachés (latent_dim) est de 256 et l’Embedding de 128.

Le modèle a été compilé avec adam et sparse_categorical_crossentropy.

4. Préparation des séquences pour le décodeur

J’ai décalé les séquences cibles d’un pas pour créer decoder_input_sequences et decoder_target_sequences.

J’ai ajouté une dimension pour sparse_categorical_crossentropy.

5. Entraînement

Paramètres : batch_size=64, epochs=30, validation_split=0.1.

J’ai utilisé un GPU pour accélérer l’entraînement.

6. Sauvegarde

J’ai sauvegardé le modèle entraîné : seq2seq_fra_en.h5

J’ai sauvegardé les tokenizers : source_tokenizer.pkl et target_tokenizer.pkl

7. Inférence

J’ai construit des modèles encodeur et décodeur séparés pour la traduction.

J’ai créé une fonction translate_sequence() pour traduire une phrase française en anglais, token par token.

Exemple
test_sentence = "Bonjour."
translation = translate_sequence(seq)
print("Français :", test_sentence)
print("Anglais :", translation)


Résultat obtenu :

Français : Bonjour.
Anglais : hello

Difficultés rencontrées

Construction du modèle d’inférence à partir du modèle entraîné.

Problèmes avec l’accès aux états du LSTM et les dimensions des tenseurs.

Résultat de traduction encore imparfait : le modèle a besoin de plus de données et d’entraînement pour mieux apprendre.

Technologies utilisées

Python 3.9

TensorFlow / Keras

NumPy

Pickle pour sauvegarder les tokenizers
