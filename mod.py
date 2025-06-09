import nltk
import re
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance

nltk.download('stopwords')

def clean_sentences(text):
    sentences = text.strip().split(". ")
    cleaned = []
    for sent in sentences:
        words = re.sub(r"[^a-zA-Z]", " ", sent).lower().split()
        if words:
            cleaned.append(words)
    return cleaned, sentences

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []

    all_words = list(set(sent1 + sent2))
    v1 = [0] * len(all_words)
    v2 = [0] * len(all_words)

    for word in sent1:
        if word not in stop_words:
            v1[all_words.index(word)] += 1
    for word in sent2:
        if word not in stop_words:
            v2[all_words.index(word)] += 1

    return 1 - cosine_distance(v1, v2)

def build_similarity_matrix(sentences, stop_words):
    size = len(sentences)
    sim_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i != j:
                sim_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)

    return sim_matrix

def summarize(text, top_n=5):
    stop_words = stopwords.words('english')
    cleaned_sentences, original_sentences = clean_sentences(text)
    
    sim_matrix = build_similarity_matrix(cleaned_sentences, stop_words)
    sentence_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(sentence_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)

    summary = " ".join([ranked_sentences[i][1] for i in range(min(top_n, len(ranked_sentences)))])
    return summary

# === LONG INPUT TEXT ABOUT AI ===
text = """
Artificial Intelligence (AI) is a rapidly advancing field of computer science that focuses on building systems capable of performing tasks that would normally require human intelligence. These tasks include learning from data, understanding natural language, recognizing patterns, solving problems, and even decision making.
AI can be classified into two main categories: narrow AI and general AI. Narrow AI is designed to perform a specific task, such as voice recognition or image classification, and it often surpasses human capabilities in those areas. General AI, on the other hand, refers to a hypothetical machine that possesses the ability to understand, learn, and apply intelligence across a wide range of tasks—similar to human cognitive ability.
One of the key enablers of modern AI is machine learning, a subset of AI that allows machines to learn from data without being explicitly programmed. Within machine learning, deep learning—especially using neural networks—has made tremendous strides in areas like image recognition, language translation, and autonomous driving.
The applications of AI are vast and diverse. In healthcare, AI is used for diagnosing diseases, predicting patient outcomes, and personalizing treatment plans. In the financial industry, it is applied in fraud detection, algorithmic trading, and credit scoring. AI is also transforming education, customer service, manufacturing, and many other industries.
Despite its numerous benefits, AI also presents significant challenges and risks. Ethical considerations such as data privacy, algorithmic bias, and the potential for job displacement must be addressed. There are also concerns about the use of AI in surveillance, misinformation, and autonomous weapon systems.
To ensure responsible development and deployment of AI, governments, organizations, and researchers around the world are working to establish frameworks and regulations. Initiatives like explainable AI, fairness in machine learning, and AI safety research aim to make AI systems more transparent, equitable, and trustworthy.
As AI continues to evolve, it is expected to become even more integrated into everyday life. From virtual assistants and self-driving cars to smart cities and intelligent robots, the impact of AI will be profound and far-reaching. However, it remains crucial to balance innovation with caution and to ensure that AI serves the greater good of humanity.
"""

# === Run summarizer ===
print("\n--- AI SUMMARY ---\n")
print(summarize(text, top_n=5))




