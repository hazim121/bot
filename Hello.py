import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import textract
import tempfile

db=None

def extract_text_from_pdf(uploaded_file):
    # Create a temporary file and write the PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())

    # Extract text using textract
    text = textract.process(temp_file.name, encoding='utf-8', errors='replace')

    # Delete the temporary file
    os.unlink(temp_file.name)
    return text.decode("utf-8")


# Define Streamlit app
def main():
    global db
    st.image("Tredence_Analytics_Logo.jpg",width=200)
    st.markdown("<h1 style='text-align: center; font-size: 2em;'>HR Bot</h1>", unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    nav = st.sidebar.radio("Navigation",["Home","Chatbot"])
    if nav == "Home":
        st.image("Tredence_Analytics_Logo.jpg",width=600)
        st.markdown("#### **Problem Statement**")
        long_text1 = '''
In today's dynamic corporate landscape, the ability to efficiently manage and comprehend vast repositories of company policy documents is a challenging task. HR professionals often face difficulties in retrieving specific information from extensive policy manuals, and traditional approaches lack the agility to answer nuanced queries effectively.

The challenge is to design and implement a sophisticated HR Assistant empowered with state-of-the-art Natural Language Processing (NLP) and Document Understanding technologies. 
'''
        st.write(f"""<div style="overflow: hidden; overflow-wrap: break-word; height: 200px;">{long_text1}</div>""",unsafe_allow_html=True,)
        st.markdown("#### **Solution**")
        long_text = '''
The solution is a smart HR Assistant powered by advanced language technology. It reads and understands company policy documents, allowing HR professionals to ask questions naturally. With its ability to provide instant, accurate responses, dynamic updates to reflect changes in policies, and seamless integration with LangChain for enhanced language understanding, the HR Assistant simplifies policy management and improves user interaction in a user-friendly manner.
'''
        st.write(f"""<div style="overflow: hidden; overflow-wrap: break-word; height: 200px;">{long_text}</div>""",unsafe_allow_html=True,)
        st.markdown("")
        st.markdown("#### **Overview of Solution**")   
        st.markdown("##### **Steps Involved:**")
        st.markdown("###### **Embedding Generation with FAISS:**")
        st.markdown('''Utilize the FAISS (Facebook AI Similarity Search) library to generate embeddings from company policy documents.
Leverage FAISS for efficient similarity search, allowing the system to compare and retrieve documents based on their embeddings.''')
        st.markdown("###### **Vector Space Model:**")
        st.markdown('''Implement a vector space model where policy documents are represented as vectors in a high-dimensional space.
Use FAISS to perform nearest-neighbor searches, enabling rapid retrieval of documents with similar embeddings.''')
        st.markdown("###### **Semantic Comparison:**")
        st.markdown('''Apply advanced semantic comparison techniques to assess the similarity between user queries and document embeddings.
Use semantic understanding to enhance the system's ability to provide contextually relevant responses..''')
        st.markdown("###### **LangChain Integration for Question-Answering:**")
        st.markdown('''Integrate LangChain to facilitate advanced question-answering capabilities.
Leverage LangChain's language models to interpret user queries, allowing the system to provide precise and accurate responses based on document embeddings.''')
      
    # Corrected indentation for st.write
    #st.write(f"""<div style="overflow: hidden; overflow-wrap: break-word; height: 200px;">{long_text2}</div>""", unsafe_allow_html=True)
    if nav == "Chatbot":
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
        if uploaded_file is not None:
            
            st.write("File Details:")
            st.write(f"Name: {uploaded_file.name}")
            st.write(f"Type: {uploaded_file.type}")
            st.write(f"Size: {uploaded_file.size} bytes")
            # Read file content
            if uploaded_file.type == "application/pdf":
                #doc = textract.process(uploaded_file)
                text = extract_text_from_pdf(uploaded_file)
                #fil=doc.decode('utf-8')
                #text = pdf_text.read()
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
                def count_tokens(text: str) -> int:
                    return len(tokenizer.encode(text))
    
                text_splitter = RecursiveCharacterTextSplitter(
  
                chunk_size = 512,
                chunk_overlap  = 24,
                length_function = count_tokens,
                )
    
                chunks = text_splitter.create_documents([text])

                from dotenv import load_dotenv

                load_dotenv()


    
                # Embed text and store embeddings
                # Get embedding model
                embeddings = OpenAIEmbeddings()  
                # Create vector database
                db = FAISS.from_documents(chunks, embeddings)
            else:
                st.warning("Unsupported file type. Please upload a pdf file.")
        else: 
            st.warning("Please upload a file.")
        
        
        st.title("HR Bot")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Hi! How can i help you?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = model_bot(prompt,db)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
       
# Function to load original data (replace this with your actual function)
def load_original_data():
    np.random.seed(42)
    data = {
        'CustomerID': range(1, 101),
        'Age': np.random.randint(18, 65, 100),
        'Gender': np.random.choice(['Male', 'Female'], size=100),
        'Income': np.random.randint(30000, 100000, 100),
        'PurchaseAmount': np.random.uniform(10, 500, 100)
        }
    customer_df = pd.DataFrame(data)
    return customer_df
def model_bot(prompt,db):
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    query = prompt    

    docs = db.similarity_search(query) 

    ans=chain.run(input_documents=docs, question=query)  


    return ans
if __name__ == "__main__":
    openai_api_key=st.secrets["key"]
    os.environ["OPENAI_API_KEY"] = openai_api_key
    main()
