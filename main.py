import os

import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


with st.sidebar:
    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.xml` file", type="xml")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get("file_cache", {}):

                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir, required_exts=[".xml"], recursive=True
                        )
                    else:
                        st.error(
                            "Could not find the file you uploaded, please check again..."
                        )
                        st.stop()

                    docs = loader.load_data()

                    # setup llm & embedding model
                    llm = Ollama(model="llama3.1", request_timeout=1200.0)
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True
                    )
                    # Creating an index over loaded data
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    # Create the query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                        "Context information about file is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, create a technical document with the following structure:\n"
                        "- Introduction: Begin with an introduction explaining that the document provides an overview of scripts related to different modules within the system, derived from the provided XML files.\n"
                        "- Sections: Create a separate section for each module present in the XML file. If the XML contains data for multiple modules (e.g., Incident, Problem, Change), create individual sections for each. If only one module is present, create a section for that module only.\n"
                        "- For each section, include:\n"
                        "  - Name: Section Name (e.g., Incident, Problem, Change, etc), if that module's data exist in the file.\n"
                        "  - Description: A brief description of the module's purpose.\n"
                        "  - Scripts:\n"
                        "    - Provide a table summarizing the scripts related to the module. The table should include columns for Script Name, Sys ID, Table, Functionality, and Purpose.\n"
                        "    - If no scripts are found for a module, Skip generating that particular Section\n"
                        "Ensure that each section is created based on the XML content and accurately reflects the data present. Do not miss any scripts or data, and be clear and concise in your responses, and in the table include every single script present for each present section.\n"
                        "The document should contain all relevant modules and scripts found in the XML file. If a module has no scripts, note that accordingly.\n"
                        "Remember to provide the scripts details in a tabular format only."
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to generate!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Generate technical document using XML")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
