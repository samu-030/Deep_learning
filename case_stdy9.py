import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

G = nx.Graph()
G.add_nodes_from([
    (1, {"text": "I love this new product!"}),
    (2, {"text": "This launch is terrible."}),
    (3, {"text": "Not bad, could be improved."})
])
G.add_edges_from([(1, 2), (2, 3)])

texts = [G.nodes[n]["text"] for n in G.nodes]
labels = ["positive", "negative", "neutral"]   # sample labels

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), padding="post")

label_enc = LabelEncoder()
y = label_enc.fit_transform(labels)

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50, input_length=X.shape[1]),
    LSTM(64),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=5, batch_size=2)

def predict_sentiment(text):
    seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=X.shape[1])
    pred = np.argmax(model.predict(seq))
    return label_enc.inverse_transform([pred])[0]

for node in G.nodes:
    G.nodes[node]["sentiment"] = predict_sentiment(G.nodes[node]["text"])

pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, "sentiment")

nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000)
nx.draw_networkx_labels(G, pos, labels, font_size=12)
plt.show()
