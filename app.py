import streamlit as st
from gemini_rag import ask_question

st.title("DocuQuery Gemini")

question = st.text_input("Ask a question from your documents")

if st.button("Submit"):
    answer = ask_question(question)
    st.write(answer)