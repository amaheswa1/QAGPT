# Importing the required libraries.
import streamlit as st
import pandas as pd
from transformers import pipeline


# Defining a function to fetch and cache the question-answering model from the transformers pipeline.
@st.cache(allow_output_mutation=True)
def get_model():
    """
    Fetches and caches the question-answering model from the transformers pipeline.

    Returns:
        question_answering_model (transformers.Pipeline): Ttranhe cached question-answering model.
    """
    return pipeline('question-answering')

# Calling the get_model function to retrieve the question-answering model and storing it in 'qa_pipeline'.
qa_pipeline = get_model()

# Setting the title for the Streamlit web application.
st.title('Question Answering Engine')

# Creating a text area input for the user to provide the context for the question.
context = st.text_area('Context', 'Enter the context here...')

# Creating a text input for the user to enter the question.
question = st.text_input('Question', 'Enter your question here...')

# Checking if the 'Generate Answer' button is clicked.
if st.button('Generate Answer'):
    """
    Processes the user-provided context and question using the question-answering model,
    then displays the generated answer.

    If either the context or the question is empty, it will display an error message.

    Parameters:
        None

    Returns:
        None
    """
    # Checking if both context and question are provided.
    if not context or not question:
        # Displaying an error message if either context or question is empty.
        st.write('Please make sure to provide both a context and a question')
    else:
        # Using the question-answering model to generate the answer based on the provided context and question.
        result = qa_pipeline({
            'context': context,
            'question': question
        })
        # Displaying the generated answer.
        st.write(result)
