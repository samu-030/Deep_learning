import tensorflow as tf
import numpy as np

text = "hello world. this is a simple rnn text generator."

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = np.array(chars)

seq_length = 10
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i:i+seq_length])
    next_chars.append(text[i+seq_length])

X = np.zeros((len(sentences), seq_length), dtype=np.int32)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_to_idx[char]
    y[i, char_to_idx[next_chars[i]]] = 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(chars), output_dim=8, input_length=seq_length),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=20, batch_size=16)

def generate_text(model, start, length=50):
    generated = start
    for _ in range(length):
        input_seq = [char_to_idx[c] for c in generated[-seq_length:]]
        input_seq = np.array(input_seq).reshape(1, seq_length)
        
        preds = model.predict(input_seq, verbose=0)[0]
        
        next_idx = np.argmax(preds)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
    return generated

print(generate_text(model, "hello wor"))
