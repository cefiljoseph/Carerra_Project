from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import streamlit as st

# Loading pretrained transformer model for answer generation
answer_generator = pipeline('text-generation', model="gpt2")

# Loading pretrained transformer model for sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# creating a database of questions and answers from pporsche.txt


qa_database = {
    "When did Ferdinand found his iconic sports car company?": "He established Porsche in 1931 in Stuttgart, Germany.",
    "What type of innovative vehicle did Ferdinand Porsche design before starting his own brand?": "He designed an early hybrid car, showcasing his forward-thinking approach.",
    "What is Porsche best known for in the automotive world?": "Porsche is famous for building high-performance sports cars with an emphasis on speed and handling.",
    "Which Porsche model is considered a timeless automotive icon?": "The Porsche 911, introduced in 1963, is their most famous model.",
    "Where does Porsche demonstrate its dominance in the racing world?": "Porsche has won countless races, including the legendary 24 Hours of Le Mans.",
    "Besides sports cars, what other successful vehicle types has Porsche introduced?": "Porsche expanded into the popular SUV market with the Cayenne and Macan.",
    "Which Porsche model represents their electric vehicle innovation?": "The Porsche Taycan showcases their commitment to performance in the electric car market.",
    "How does Porsche extend its design philosophy beyond automobiles?": "Porsche Design creates stylish watches, eyewear, and other lifestyle products.",
    "Where is the must-visit Porsche Museum located?": "The Porsche Museum is located in Stuttgart, Germany.",
    "What are the regular operating hours of the Porsche Museum?": "The museum is open Tuesday to Sunday, 9 am to 6 pm, and closed on Mondays.",
    "When do the ticket desks at the Porsche Museum close?": "Ticket desks close at 5:30 pm.",
    "Where can I check for special opening/closing days around holidays?": "Visit the official Porsche Museum website (https://www.porsche.com/international/aboutporsche/porschemuseum/).",
    "How much is regular adult admission to the Porsche Museum?": "Adult admission is 12 euros.",
    "Are there any discounted ticket prices available?": "Yes, reduced prices apply to students, seniors, those with disabilities, and others. Children under 14 enter for free.",
    "Can I get a discount if I visit the Porsche Museum in the evening?": "Yes, evening tickets (after 5 pm) are available at a reduced price.",
    "How is the exhibition laid out in the Porsche Museum?": "It follows a spiral design; follow it to the left for a chronological experience.",
    "What are the dining and refreshment options at the Porsche Museum?": "The museum has a cafe (Boxenstopp) and a restaurant (Christophorus).",
    "Besides the car exhibits, what else can I enjoy at the museum?": "There's a workshop viewing area, driving simulators, and cars you can sit in.",
    "When is the best time to visit the Porsche Museum for a less crowded experience?": "Try weekday mornings for a quieter visit."
}

#generating embeddings for the entire database
database_questions = list(qa_database.keys())
database_embeddings = model.encode(database_questions)

#Building index using FAISS(vector indexing)
index = faiss.IndexFlatL2(database_embeddings.shape[1])
index.add(np.array(database_embeddings).astype('float32'))

#Function to retrieve the most similar question using vector indexing
def retrieve_most_similar_question(user_question, index, database_embeddings, database_questions):
    user_embedding = model.encode(user_question).reshape(1, -1).astype('float32')
    _, most_similar_index = index.search(user_embedding, 1)
    most_similar_question = database_questions[most_similar_index[0][0]]
    return most_similar_question


# Streamlit app hosting
st.title("Hello, I am Carerra 🚗, Porsche Museum's AI assistant :)")

# User inputing section 
user_question = st.text_input("How can I help you today?")


if user_question:
    # Retrieve the most similar question from the database using vector indexing
    most_similar_question = retrieve_most_similar_question(user_question, index, database_embeddings, database_questions)

    # Retrieve the answer from the database or generate a new answer
    if most_similar_question in qa_database:
        answer = qa_database[most_similar_question]
    else:
        # If not found in the database, generate an answer using a transformer-based language model
        answer = answer_generator(user_question, max_length=100, num_return_sequences=1, top_k=50)[0]['generated_text']

    # Display the answer
    st.subheader("Carerra 🚗")
    st.write(answer)


# Improve UI
st.markdown("---")
st.markdown("**Welcome to Porsche Museum, Stuttgart!**")
st.markdown("Your Porsche adventure awaits you")
st.markdown("---")

'''
**Disclaimer:**

*Carerra 🚗, the Porsche Museum, and Porsche are registered trademarks owned by their respective owners. This document and the associated project are not affiliated with, endorsed by, or sponsored by the trademark owners. Any use of trademark names is for illustrative purposes only, and all rights to these trademarks are explicitly acknowledged.*

*The information provided in this document and associated code is for educational and demonstration purposes. Users should be aware of and adhere to the trademark policies and guidelines set forth by the respective trademark owners. Any reference to specific trademarks is not intended to imply a partnership, endorsement, or association with the trademark owners.*

*It is the responsibility of users to ensure compliance with trademark laws and regulations when using or referring to trademarks mentioned in this document.*
'''