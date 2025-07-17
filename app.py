import streamlit as st
import os
import tempfile
import gc
import base64
import time

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import SerperDevTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool as SimpleDocumentSearchTool
from src.agentic_rag.tools.custom_tool_complex import DocumentSearchTool as ComplexDocumentSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def load_llm():
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.7,
    )
    return llm

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks(pdf_tool):
    """Creates a Crew with the given PDF tool (if any) and a web search tool."""
    web_search_tool = SerperDevTool()

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first. "
            "If you are not able to retrieve the information from the PDF search tool, "
            "then try to use the web search tool. Consider the conversation history: {chat_history} "
            "to provide contextually relevant information."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            "You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base. "
            "You maintain awareness of the ongoing conversation to provide better context."
        ),
        verbose=True,
        tools=[t for t in [pdf_tool, web_search_tool] if t],
        llm=load_llm()
    )

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            "based on the user query: {query}. Consider the conversation history: {chat_history} "
            "to maintain context and provide follow-up responses. If you are not able to retrieve the "
            'information then respond with "I\'m sorry, I couldn\'t find the information '
            'you\'re looking for."'
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses. "
            "You excel at maintaining conversational flow and providing contextual answers "
            "based on previous interactions in the chat."
        ),
        verbose=True,
        llm=load_llm()
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant information from the available "
            "sources for the user query: {query}. Consider the conversation history: {chat_history} "
            "to understand the context and provide relevant information."
        ),
        expected_output=(
            "The most relevant information in the form of text as retrieved "
            "from the sources, considering the conversation context."
        ),
        agent=retriever_agent
    )

    response_task = Task(
        description=(
            "Synthesize the final response for the user query: {query}. "
            "Use the conversation history: {chat_history} to maintain context and provide "
            "a natural, conversational response that builds upon previous interactions."
        ),
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            "from the right source for the user query: {query}. The response should "
            "maintain conversational context and flow naturally from previous messages. "
            "If you are not able to retrieve the information, then respond with: "
            '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,  # or Process.hierarchical
        verbose=True
    )
    return crew

# ===========================
#   Streamlit Setup
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

def reset_chat():
    st.session_state.messages = []
    st.session_state.crew = None  # Reset crew when chat is reset
    gc.collect()

def format_chat_history(messages, max_messages=5):
    """Format recent chat history for context."""
    if not messages:
        return "No previous conversation."
    
    # Get the last few messages for context (excluding the current one)
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    formatted_history = "Recent conversation:\n"
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        formatted_history += f"{role}: {content}\n"
    
    return formatted_history

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("📄 Document Management")
    
    # Show conversation statistics
    if st.session_state.messages:
        num_exchanges = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        st.metric("Conversation Exchanges", num_exchanges)
    
    # Document source selection
    doc_source = st.radio(
        "Choose document source:",
        ["Upload New Document", "Use Existing GroundX Document"],
        help="Upload a new PDF or select from previously indexed GroundX documents"
    )
    
    if doc_source == "Upload New Document":
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        tool_choice = st.radio(
            "Choose Document Analysis Type:",
            ('Simple PDF Processing', 'Complex Document Analysis (with images)'),
            key='tool_choice_radio',
            on_change=reset_chat # Reset chat and tool when choice changes
        )
        st.session_state.tool_choice = 'simple' if tool_choice == 'Simple PDF Processing' else 'complex'
        
        # Show which RAG system will be used
        if tool_choice == 'Simple PDF Processing':
            st.info("🔧 **RAG System:** Chroma (Local, Fast)")
        else:
            st.info("🔧 **RAG System:** GroundX (Cloud, Advanced)")
            st.caption("⏱️ Note: GroundX processing takes longer but handles images and complex layouts better.")

        st.markdown("""
            <style>
            .stRadio > label > div:first-child {
                padding-right: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:
            # If there's a new file and we haven't set pdf_tool yet...
                    # If there's a new file and we haven't set pdf_tool yet...
            # If there's a new file and we haven't set pdf_tool yet, or if the tool choice has changed
            if st.session_state.pdf_tool is None or st.session_state.get('previous_tool_choice') != tool_choice:
                st.session_state.previous_tool_choice = tool_choice # Store current choice for next check
                # Clear existing tool and crew to force re-initialization
                st.session_state.pdf_tool = None
                st.session_state.crew = None  # Reset crew when switching RAG systems

                # tool_choice is already defined above

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    try:
                        if tool_choice == 'Simple PDF Processing':
                            with st.spinner("Indexing PDF with Chroma... Please wait..."):
                                st.session_state.pdf_tool = SimpleDocumentSearchTool(file_path=temp_file_path)
                            st.success("✅ PDF indexed with Chroma! Ready to chat.")
                        else:
                            with st.spinner("Uploading and processing document with GroundX... This may take a few minutes..."):
                                st.session_state.pdf_tool = ComplexDocumentSearchTool(file_path=temp_file_path)
                            st.success("✅ PDF fully processed with GroundX! Ready for complex document analysis.")
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        st.session_state.pdf_tool = None

            # Optionally display the PDF in the sidebar
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    
    else:  # Use Existing GroundX Document
        st.subheader("📚 Existing GroundX Documents")
        
        if st.button("🔄 Refresh Document List"):
            if 'existing_buckets' in st.session_state:
                del st.session_state.existing_buckets
        
        # Load existing buckets if not already loaded
        if 'existing_buckets' not in st.session_state:
            with st.spinner("Loading existing GroundX documents..."):
                try:
                     st.session_state.existing_buckets = ComplexDocumentSearchTool.list_existing_buckets()
                except Exception as e:
                    st.error(f"❌ Error loading existing documents: {str(e)}")
                    st.session_state.existing_buckets = []
        
        # Display existing buckets and documents
        if st.session_state.get('existing_buckets'):
            bucket_options = {}
            
            for bucket in st.session_state.existing_buckets:
                if bucket.get('documents'):
                    for doc in bucket['documents']:
                        if doc['status'] == 'complete':
                            display_name = f"{doc['name']} (Bucket: {bucket['bucket_name'][:20]}...)"
                            bucket_options[display_name] = {
                                'bucket_id': bucket['bucket_id'],
                                'document_name': doc['name'],
                                'created_at': doc['created_at']
                            }
            
            if bucket_options:
                selected_doc = st.selectbox(
                    "Select a document:",
                    list(bucket_options.keys()),
                    help="Choose from previously indexed documents"
                )
                
                if selected_doc and st.button("Use Selected Document"):
                    # Reset crew when switching to existing document
                    if 'crew' in st.session_state:
                        del st.session_state.crew
                    
                    try:
                        bucket_id = bucket_options[selected_doc]['bucket_id']
                        st.session_state.pdf_tool = ComplexDocumentSearchTool(bucket_id=bucket_id)
                        st.session_state.tool_type = "complex"
                        st.session_state.pdf_processed = True
                        
                        st.success(f"✅ Using existing document: {bucket_options[selected_doc]['document_name']}")
                        st.info("🚀 **RAG System**: GroundX (Existing Document)")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading existing document: {str(e)}")
            else:
                st.info("📝 No completed documents found in GroundX. Upload a new document to get started.")
        else:
            st.info("📝 No existing GroundX buckets found. Upload a new document to get started.")
    
    # Show conversation stats
    if st.session_state.messages:
        st.markdown("---")
        st.markdown(f"💬 **Conversation:** {len(st.session_state.messages)//2} exchanges")


    st.markdown("---")
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        reset_chat()
        st.rerun()
    
    if st.session_state.messages:
        st.caption("💡 Tip: I remember our conversation context for better responses!")

# ===========================
#   Main Chat Interface
# ===========================
st.markdown("""
    # 🤖 Agentic RAG Chatbot
    ### Chat with your documents using AI agents
""")

# Show welcome message if no messages yet
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if st.session_state.pdf_tool:
            st.markdown("👋 Hello! I'm ready to help you with questions about your uploaded document. What would you like to know?")
        else:
            st.markdown("👋 Hello! Please upload a PDF document in the sidebar to get started. Once uploaded, I'll be able to answer questions about its content!")

# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.pdf_tool:
    prompt = st.chat_input("💭 Ask me anything about your document...")
else:
    prompt = st.chat_input("Please upload a PDF document first to start chatting...")
    if prompt:
        st.warning("⚠️ Please upload a PDF document in the sidebar before asking questions.")
        prompt = None  # Prevent processing

if prompt:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build or reuse the Crew (only once after PDF is loaded)
    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

    # 3. Get the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the complete response first
        with st.spinner("Thinking..."):
            # Format chat history for context (excluding the current message)
            chat_history = format_chat_history(st.session_state.messages[:-1])  # Exclude current user message
            inputs = {
                "query": prompt,
                "chat_history": chat_history
            }
            result = st.session_state.crew.kickoff(inputs=inputs).raw
        
        # Split by lines first to preserve code blocks and other markdown
        lines = result.split('\n')
        for i, line in enumerate(lines):
            full_response += line
            if i < len(lines) - 1:  # Don't add newline to the last line
                full_response += '\n'
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.15)  # Adjust the speed as needed
        
        # Show the final response without the cursor
        message_placeholder.markdown(full_response)

    # 4. Save assistant's message to session
    st.session_state.messages.append({"role": "assistant", "content": result})
