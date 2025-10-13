# LangGraph Projects Repository

This repository contains a collection of LangGraph projects organized by complexity, from basic implementations to advanced applications.

## 📁 Repository Structure

Projects are organized in separate branches, each representing different levels of complexity:

- **Basic Projects**: Simple LangGraph implementations and concepts
- **Intermediate Projects**: More complex workflows and integrations
- **Advanced Projects**: Sophisticated applications with multiple components

## 🚀 What You'll Find Here

- **LangGraph Workflows**: Various types of graph-based AI workflows
- **Agent Implementations**: Different agent patterns and architectures
- **Integration Examples**: LangGraph with various tools and APIs
- **Best Practices**: Code examples following LangGraph conventions
- **Real-world Applications**: Practical use cases and implementations

## 🌿 Branch Organization

Each branch contains a complete, self-contained project with:
- Clear documentation
- Working code examples
- Requirements and setup instructions
- Usage examples

## 🛠️ Technologies Used

- **LangGraph**: Core framework for building stateful, multi-actor applications
- **Python**: Primary programming language
- **Various AI/ML Libraries**: Depending on project requirements
- **APIs and Tools**: Integration with external services

## 🔄 Project Evolution

Projects are designed to build upon each other, with each branch introducing new concepts and complexity levels. Start with the basic branches and work your way up to more advanced implementations.

## 📂 Current Implementations

### `react_agent/` - ReAct Agent Implementation

A manual implementation of the ReAct (Reasoning and Acting) pattern using LangChain. This folder demonstrates how ReAct agents work under the hood:

**Core Components:**
- **`main.py`**: Complete ReAct agent using LangChain's built-in `create_react_agent` with Tavily search
- **`basic_tool_calling.py`**: Manual step-by-step implementation showing the ReAct loop
- **`basic_tool_calling_using_callbacks.py`**: Same implementation with callback handlers for monitoring
- **`prompt.py`**: ReAct prompt template with the classic Thought/Action/Observation format
- **`schemas.py`**: Pydantic models for structured agent responses
- **`callbacks.py`**: Custom callback handler for logging LLM interactions
- **`function_call_evolution.py`**: Modern approach using `.bind_tools()` instead of ReAct prompts

**How ReAct Works:**
1. **Thought**: Agent analyzes the question and decides what to do
2. **Action**: Chooses a tool to call (or determines it has the final answer)
3. **Action Input**: Provides parameters for the tool
4. **Observation**: Receives the tool's result
5. **Loop**: Repeats until the agent concludes it has enough information
6. **Final Answer**: Provides the structured response

The implementation shows both the high-level LangChain approach and the manual step-by-step process, making it perfect for understanding ReAct internals.

**Evolution to Modern Function Calling (`function_call_evolution.py`):**

Instead of using the traditional ReAct prompt pattern, this file demonstrates the modern approach using `.bind_tools()`. This represents a significant evolution in how we handle tool calling:

**Traditional ReAct Approach:**
- Requires complex prompt templates with specific formatting
- Developer must manually parse agent responses
- More error-prone and verbose
- Requires explicit handling of `AgentAction` and `AgentFinish` types

**Modern `.bind_tools()` Approach:**
- **Simplified Implementation**: No need for complex prompt templates
- **LLM-Native Function Calling**: The LLM handles function calling natively behind the scenes
- **Automatic Tool Management**: The LLM automatically determines when to call functions, what parameters to use, and assigns unique IDs
- **Reduced Developer Responsibility**: Less manual parsing and error handling required
- **More Reliable**: Leverages the model's built-in function calling capabilities

**Key Benefits:**
- **Cleaner Code**: Eliminates the need for ReAct prompt templates and manual parsing
- **Better Integration**: Uses the model's native function calling features
- **Automatic ID Management**: The LLM automatically assigns and tracks tool call IDs
- **Simplified Debugging**: Easier to trace tool execution flow
- **Future-Proof**: Aligns with modern LLM capabilities and best practices

This evolution represents a shift from developer-managed tool calling to LLM-managed tool calling, where the responsibility for determining when and how to call tools is delegated to the language model itself.

### `intro-vector-dbs/` - Vector Database Introduction

A comprehensive introduction to vector databases and RAG (Retrieval Augmented Generation) using LangChain. This folder demonstrates how to work with vector stores for document retrieval and question-answering:

**Core Components:**
- **`ingestion.py`**: Document ingestion pipeline using Pinecone vector database
- **`retrieval.py`**: RAG implementation with both built-in chains and custom LCEL chains
- **`chat_with_pdf.py`**: Complete PDF-to-vector workflow using FAISS for local storage

**Vector Database Workflow:**
1. **Document Loading**: Load PDFs or text files using LangChain loaders
2. **Text Splitting**: Split documents into manageable chunks (1000 tokens with 200 overlap)
3. **Embedding Generation**: Convert text chunks to vector embeddings using OpenAI
4. **Vector Storage**: Store embeddings in vector database (Pinecone or FAISS)
5. **Retrieval**: Query vector store for semantically similar documents
6. **RAG Generation**: Combine retrieved context with LLM for contextual answers

**Key Features:**
- **Multiple Vector Stores**: Examples with both Pinecone (cloud) and FAISS (local)
- **Document Types**: Support for PDFs and text files
- **RAG Patterns**: Both LangChain's built-in chains and custom LCEL implementations
- **Local Storage**: FAISS index persistence for offline usage
- **Chunking Strategies**: Configurable text splitting with overlap for context preservation

This implementation provides a solid foundation for understanding how vector databases enable semantic search and retrieval-augmented generation in AI applications.

---

*This repository serves as a comprehensive collection of LangGraph implementations, showcasing various patterns, techniques, and real-world applications.*
