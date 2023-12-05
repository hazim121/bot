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
import io
import tempfile



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
    nav = st.sidebar.radio("Navigation",["Home","Upload PDF/DOC","Chatbot"])
    if nav == "Home":
        st.image("Tredence_Analytics_Logo.jpg",width=600)
        st.markdown("#### **Problem Statement**")
        long_text1 = '''
Problem statement bfkdsvdev bdsvkdsvdkjv
'''
        st.write(f"""<div style="overflow: hidden; overflow-wrap: break-word; height: 200px;">{long_text1}</div>""",unsafe_allow_html=True,)
        st.markdown("#### **Solution**")
        long_text = '''
solution being offred sdhbcv shbvcdsjv hbckjdsbc
'''
        st.write(f"""<div style="overflow: hidden; overflow-wrap: break-word; height: 200px;">{long_text}</div>""",unsafe_allow_html=True,)
        st.markdown("")
        st.markdown("#### **Overview of Solution**")   
        st.markdown("##### **Steps Involved:**")
        st.markdown("###### **Scaling Data:**")
        st.markdown('''The function starts by scaling the numerical columns ('Age', 'Income', 
'PurchaseAmount') of the input DataFrame using Min-Max scaling.''')
        st.markdown("###### **GAN Architecture:**")
        st.markdown('''It then defines functions to build the generator, discriminator, and the 
GAN model. The generator creates synthetic data, the discriminator evaluates 
whether the data is real or synthetic, and the GAN combines these two networks.''')
        st.markdown("###### **Model Compilation:**")
        st.markdown('''The discriminator is compiled with binary crossentropy loss, and the 
Adam optimizer. The GAN is compiled with the same optimizer and loss function 
but with the discriminator weights frozen.''')
        st.markdown("###### **Training Loop:**")
        st.markdown('''The GAN is trained through a loop of a specified number of epochs. In each 
epoch, it generates synthetic data, updates the discriminator using both real 
and generated data, and updates the generator to fool the discriminator.''')
        st.markdown("###### **Data Generation:**")
        st.markdown('''After training, the function generates synthetic data by feeding random 
noise through the trained generator. The generated data is then inverse transformed 
to the original scale.''')
        st.markdown("###### **Data Post-Processing:**")
        st.markdown('''The synthetic DataFrame is created and further processed, rounding 
'Age' to integers, clipping 'Income' and 'PurchaseAmount' to reasonable 
ranges, and assigning 'Male' or 'Female' genders randomly.''')
        st.markdown('''In summary, this program leverages a GAN to create synthetic data 
closely resembling the input dataset's statistical properties, offering 
a method for privacy-preserving data generation or augmentation in various
applications.''')
    # Corrected indentation for st.write
    #st.write(f"""<div style="overflow: hidden; overflow-wrap: break-word; height: 200px;">{long_text2}</div>""", unsafe_allow_html=True)
    if nav == "Upload PDF/DOC":
    
        
        
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
        
        
        


    if nav == "Chatbot":
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
