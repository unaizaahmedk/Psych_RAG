# PsychChatMistral RAG: A Psychology Retrieval-Augmented Generation (RAG) Chatbot

## Overview
Psych RAG is a specialized chatbot designed to assist users with psychology-related questions using a Retrieval-Augmented Generation (RAG) approach. The project employs Hugging Face for NLP models, Pinecone for efficient document retrieval, Gradio for an interactive user interface, FastAPI for handling backend API requests, and LangChain to seamlessly integrate these components. 

## Model Used:
mistralai/Mistral-Nemo-Instruct-2407

## Project Structure
- `data/`: Contains the `Psychology.pdf` document that the chatbot is trained on.
- `app/`: Backend of the application, including the main logic and API interactions.
  - `__init__.py`: Initializes the app module.
  - `api.py`: Handles API interactions.
  - `helper.py`: Contains helper functions.
- `frontend/`: Frontend of the application, managing the user interface.
  - `__init__.py`: Initializes the frontend module.
  - `main.py`: Manages the user interface and interaction with the backend.

## Environment Variables
Ensure you have the following variables in your `.env` file:

- `HUGGINGFACE_API_KEY`: Your Hugging Face API key.
- `PINECONE_API_KEY`: Your Pinecone API key.

## Contact
For any inquiries, please contact unaizaahmed.k@gmail.com or open an issue in the repository.


