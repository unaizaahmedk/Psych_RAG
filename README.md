# PsychChatMistral RAG: A Psychology-Focused RAG Chatbot  

**Capstone Project – AI Bootcamp (NUST × AtomCamp)**  
*Selected among the Top 5 projects*  


-----

## Project Overview

**Psych RAG** is a Retrieval-Augmented Generation (RAG) chatbot specifically designed to answer psychology-related questions with contextually accurate responses. The chatbot is built on a clean microservice architecture, leveraging **LangChain**, **FastAPI**, **Gradio**, **Hugging Face models**, and **Pinecone** to ensure fast and reliable performance.

Developed as the capstone project for the **AI Bootcamp at NUST × AtomCamp**, Psych RAG was recognized as a standout project, earning a spot among the **Top 5** in the cohort.

![Screenshot 2024-08-16 184221](https://github.com/user-attachments/assets/757556bb-bb81-41fb-bc9e-17c734305346)
![Screenshot 2024-08-16 185437](https://github.com/user-attachments/assets/4743699b-0b97-47d7-85f6-3229e66ee331)
![Screenshot 2024-08-16 190318](https://github.com/user-attachments/assets/82ed14f7-1869-4868-9364-60a9580e6e0b)


-----

## Key Features

  * **Domain-focused RAG** – Retrieves relevant passages from psychology documents to support high-quality responses.
  * **Mistral-Nemo-Instruct Model** – Utilizes `mistralai/Mistral-Nemo-Instruct-2407` for enhanced reasoning and generation.
  * **Efficient Vector Search** – Powered by Pinecone for semantic retrieval of document chunks.
  * **Clean Microservice Architecture** – A FastAPI backend handles APIs, a Gradio frontend provides the user interface, and LangChain orchestrates the entire process.
  * **User-Friendly Interface** – Allows users to query the chatbot naturally and receive contextual, well-structured answers.

-----

## Project Structure

```bash
.
├── data/
│   └── Psychology.pdf # Source document for the RAG system
├── app/
│   ├── api.py # FastAPI backend routes
│   └── helper.py # Retrieval, embedding, and RAG logic
├── frontend/
│   └── main.py # Gradio UI for chatting
├── requirements.txt # Python dependencies
└── README.md
```

-----

## Setup & Usage

### 1\. Prerequisites

First, create a `.env` file in the project's root directory and add your API keys:

```env
HUGGINGFACE_API_KEY=your_huggingface_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Then, install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```

### 2\. Running the Chatbot

Start the backend server using the following command:

```bash
uvicorn app.api:app --reload
```

In a separate terminal, launch the frontend interface:

```bash
python frontend/main.py
```

Open the local Gradio link in your browser and begin chatting with the bot.

-----

## Workflow – How It Works

The chatbot's RAG pipeline operates through the following steps:

1.  **Document Ingestion** – The `Psychology.pdf` document is loaded and split into manageable text chunks.
2.  **Vector Indexing** – These chunks are embedded using a Hugging Face model and stored in a Pinecone vector index for efficient searching.
3.  **Query Management** – A user's query is embedded and used to retrieve the most relevant context chunks from the Pinecone index.
4.  **Response Generation** – The retrieved context and the user's query are passed to the Mistral-Nemo-Instruct model to generate a relevant and grounded answer.
5.  **UI Presentation** – The final response is presented to the user through a clean Gradio chat interface.

-----

## Highlights & Achievements

  * Ranked among the **Top 5 projects** in the AI Bootcamp.
  * Features a fully functional retrieval and generation pipeline.
  * Designed with a deployment-ready microservice architecture (FastAPI + Gradio).
  * A domain-specific chatbot grounded in high-quality psychological content.
