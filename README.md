# PsychChatMistral RAG: A Psychology Retrieval-Augmented Generation (RAG) Chatbot

## Overview
Psych RAG is a specialized chatbot designed to assist users with psychology-related questions using a Retrieval-Augmented Generation (RAG) approach. The project employs Hugging Face for NLP models, Pinecone for efficient document retrieval, Gradio for an interactive user interface, FastAPI for handling backend API requests, and LangChain to seamlessly integrate these components. 

![Screenshot 2024-08-16 184221](https://github.com/user-attachments/assets/757556bb-bb81-41fb-bc9e-17c734305346)

![Screenshot 2024-08-16 185437](https://github.com/user-attachments/assets/4743699b-0b97-47d7-85f6-3229e66ee331)

![Screenshot 2024-08-16 190318](https://github.com/user-attachments/assets/82ed14f7-1869-4868-9364-60a9580e6e0b)

## Model Used:
mistralai/Mistral-Nemo-Instruct-2407

## Project Structure
- `data/`: Contains the `Psychology.pdf` document, which the chatbot uses as its source material.
- `app/`: This is the backend of the application. It handles all the main logic and interactions with FastAPI. 
  - `__init__.py`: Initializes the app module.
  - `api.py`: Handles API interactions.
  - `helper.py`: Contains helper functions.
- `frontend/`: This is the frontend of the application, built with Gradio. 
  - `__init__.py`: Initializes the frontend module.
  - `main.py`: Manages the user interface and interaction with the backend.
- `requirements.txt` : Lists all the Python packages and dependencies required to run the application.

## Environment Variables
Ensure you have the following variables in your `.env` file:

- `HUGGINGFACE_API_KEY`: Your Hugging Face API key.
- `PINECONE_API_KEY`: Your Pinecone API key.

## Contact
For any inquiries, please contact unaizaahmed.k@gmail.com or open an issue in the repository.


