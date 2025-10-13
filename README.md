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

**How ReAct Works:**
1. **Thought**: Agent analyzes the question and decides what to do
2. **Action**: Chooses a tool to call (or determines it has the final answer)
3. **Action Input**: Provides parameters for the tool
4. **Observation**: Receives the tool's result
5. **Loop**: Repeats until the agent concludes it has enough information
6. **Final Answer**: Provides the structured response

The implementation shows both the high-level LangChain approach and the manual step-by-step process, making it perfect for understanding ReAct internals.

---

*This repository serves as a comprehensive collection of LangGraph implementations, showcasing various patterns, techniques, and real-world applications.*
