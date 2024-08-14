from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA


load_dotenv()

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


def load_document(file):
    try:
        loader = PyMuPDFLoader(file, extract_images=True)
        print("PyMuPDFLoader initialized successfully")
        return loader.load()
    except Exception as e:
        print(f"Error loading document: {e}")
        raise


def split_document(data_from_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    return text_splitter.split_documents(data_from_doc)


def create_embeddings():
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Extract the text content from the Document objects
    # texts = [doc.page_content for doc in text_chunks]
    # embeddings = embeddings_model.embed_documents(texts)
    # len(embeddings) = 9672
    # len(embeddings[0]) = 384

    return embeddings_model


def create_vector_store(text_chunks, embeddings_model):
    vector_store = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings_model,
        index_name="psychbot"
    )
    return vector_store


def create_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": 2})


def create_prompt_template():
    template = """
        You are a knowledgeable assistant. You should only provide responses based on the provided context.

        Context: {context}

        Question: {question}

        If the context does not contain relevant information to answer the question, respond with: 
        "I'm sorry, but my knowledge on this topic is limited to the provided document. Is there something else I can assist you with?"

        Only return the Answer below and nothing else.

        Answer:
        """
    return template


def prepare_qa_chain():

    # Full path to the PDF file
    file_path = ("/Users/Unaiza/PycharmProjects/pythonProject/data/Psychology.pdf")

    # Debugging output
    print(f"Checking file path: {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"File exists: {os.path.isfile(file_path)}")

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"File path {file_path} is not a valid file or url")

    try:
        # Load document
        print("Attempting to load document...")
        loaded_document = load_document(file_path)
        print("Document loaded successfully")

        # Split document
        print("Attempting to split document...")
        splitted_document = split_document(loaded_document)
        print("Document split successfully")

        # Create embeddings
        print("Attempting to create embeddings...")
        embeddings_model = create_embeddings()
        print("Embeddings created successfully")

        # Create vector store
        print("Attempting to create vector store...")
        vector_store = create_vector_store(splitted_document, embeddings_model)
        print("Vector store created successfully")

        # Create retriever
        print("Attempting to create retriever...")
        retriever = create_retriever(vector_store)
        print("Retriever created successfully")

        # Create prompt template
        print("Attempting to create prompt template...")
        prompt_template = create_prompt_template()
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        print("Prompt template created successfully")

        # Prepare QA chain
        print("Attempting to prepare QA chain...")
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            model_kwargs={"temperature": 0.7,
                          },
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("QA chain prepared successfully")

        return qa_chain

    except Exception as e:
        print(f"Error in prepare_qa_chain: {e}")
        raise

