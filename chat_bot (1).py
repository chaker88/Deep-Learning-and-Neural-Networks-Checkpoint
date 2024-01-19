import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the text file and preprocess the data
# the text file is focused en astronomy
with open('chat_bot/astro.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define a function to preprocess each sentence


def preprocess(sentence):
   # Removing illustration or note references
    sentence = re.sub(r'\[Illustration:.*?\]', '', sentence)
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words(
        'english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transform the corpus into TF-IDF vectors
corpus_tfidf = vectorizer.fit_transform(
    [' '.join(sentence) for sentence in corpus])

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    query_str = ' '.join(query)

    # Transform the query into TF-IDF vector
    query_tfidf = vectorizer.transform([query_str])

    # Compute Cosine Similarity between query and each sentence in the corpus
    similarities = cosine_similarity(query_tfidf, corpus_tfidf)

    # Get the indices of sentences sorted by similarity
    sorted_indices = similarities.argsort()[0][::-1]

    # Filter for sentences with high similarity
    relevant_indices = [
        index for index in sorted_indices if similarities[0, index] > 0.3]

    # Randomize the relevant indices to get variation in responses
    random.shuffle(relevant_indices)

    # Pick a relevant sentence based on the randomized indices
    for index in relevant_indices:
        return ' '.join(corpus[index])

    # If no relevant sentence is found, return an empty string
    return 'Please repeat the question'


def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # If a relevant sentence is found, return it
    # Create a pool of potential responses
    responses = [
        "That's an interesting question!",
        "I'm pondering your query.",
        "Let me think about that for a moment.",
        "I'm not sure about that. How about we explore another topic?",
        # Add more potential responses here
    ]

    if len(most_relevant_sentence) != 0:
        return most_relevant_sentence
    else:
        # If no relevant sentence is found, select a random response
        return random.choice(responses)
    # Return the answer


def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    # Get the user's question
    question = st.text_input("You:")
    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)


if __name__ == "__main__":
    main()
