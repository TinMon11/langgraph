from dotenv import load_dotenv

load_dotenv()
from ingestion import retriever

from ...nodes.generate import generation_chain
from ..retrieval_grader import retrieval_grader
from ..hallucination_grader import hallucination_grader, GradeHallucinations


def test_retrieval_grader_answer_yes() -> None:
    question = "Agent memory"
    documents = retriever.invoke(question)
    doc_txt = documents[0].page_content

    result = retrieval_grader.invoke({"documents": doc_txt, "question": question})
    assert result.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "What is the capital of France?"
    documents = retriever.invoke(question)
    doc_txt = documents[0].page_content

    result = retrieval_grader.invoke({"documents": doc_txt, "question": question})
    assert result.binary_score == "no"


def test_generation_chain() -> None:
    question = "Agent memory"
    documents = retriever.invoke(question)
    result = generation_chain.invoke({"question": question, "context": documents})
    print("Result", result)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score == "yes"


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert res.binary_score == "no"
