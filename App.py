import os
import uuid
import streamlit as st
from pydantic import BaseModel
from typing import Literal
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

BOT_NAME = "Blog Generator (with Human Review)"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title=BOT_NAME, page_icon="📝")
st.title("📝 " + BOT_NAME)


class BlogState(BaseModel):
    topic: str = ""
    audience: str = ""
    research_output: str = ""
    writer_output: str = ""
    editor_output: str = ""
    human_feedback: str = ""
    stage: str = ""


@st.cache_resource
def build_graph():
    llm = ChatGroq(model="openai/gpt-oss-20b")

    def researcher_node(state: BlogState) -> BlogState:
        prompt = f"""You are a researcher. Gather key points for a blog on:
"{state.topic}" for audience: {state.audience}. List 5-7 research points."""
        state.research_output = llm.invoke(prompt).content
        return state

    def writer_node(state: BlogState) -> BlogState:
        prompt = f"""Write a full blog post for {state.audience} using this research:
{state.research_output}
Apply this feedback if present: {state.human_feedback}"""
        state.writer_output = llm.invoke(prompt).content
        return state

    def editor_node(state: BlogState) -> BlogState:
        prompt = f"""Polish this blog post (grammar, flow). Apply feedback if present: {state.human_feedback}
Draft: {state.writer_output}"""
        state.editor_output = llm.invoke(prompt).content
        return state

    def research_review(state: BlogState) -> BlogState:
        feedback = interrupt({"stage": "researcher_review", "output": state.research_output,
                               "question": "Approve or give feedback?"})
        state.human_feedback = feedback
        return state

    def writer_review(state: BlogState) -> BlogState:
        feedback = interrupt({"stage": "writer_review", "output": state.writer_output,
                               "question": "Approve or give feedback?"})
        state.human_feedback = feedback
        return state

    def editor_review(state: BlogState) -> BlogState:
        feedback = interrupt({"stage": "editor_review", "output": state.editor_output,
                               "question": "Approve or give feedback?"})
        state.human_feedback = feedback
        return state

    def route_after_research(state: BlogState) -> Literal["writer", "researcher"]:
        return "writer" if "approve" in state.human_feedback.lower() else "researcher"

    def route_after_writer(state: BlogState) -> Literal["editor", "writer"]:
        return "editor" if "approve" in state.human_feedback.lower() else "writer"

    def route_after_editor(state: BlogState) -> Literal["_end_", "editor"]:
        return "_end_" if "approve" in state.human_feedback.lower() else "editor"

    graph = StateGraph(BlogState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("research_review", research_review)
    graph.add_node("writer", writer_node)
    graph.add_node("writer_review", writer_review)
    graph.add_node("editor", editor_node)
    graph.add_node("editor_review", editor_review)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "research_review")
    graph.add_conditional_edges("research_review", route_after_research,
                                 {"writer": "writer", "researcher": "researcher"})
    graph.add_edge("writer", "writer_review")
    graph.add_conditional_edges("writer_review", route_after_writer,
                                 {"editor": "editor", "writer": "writer"})
    graph.add_edge("editor", "editor_review")
    graph.add_conditional_edges("editor_review", route_after_editor,
                                 {"_end_": END, "editor": "editor"})

    return graph.compile(checkpointer=MemorySaver())


finalGraph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.started = False
    st.session_state.finished = False

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if not st.session_state.started:
    topic = st.text_input("Blog topic")
    audience = st.text_input("Target audience")
    if st.button("Start") and topic and audience:
        result = finalGraph.invoke({"topic": topic, "audience": audience}, config=config)
        st.session_state.started = True
        st.rerun()

elif not st.session_state.finished:
    snap = finalGraph.get_state(config)
    if snap.interrupts:
        payload = snap.interrupts[0].value
        st.subheader(f"Stage: {payload.get('stage')}")
        st.write(payload.get("output", ""))
        st.caption(payload.get("question", ""))

        feedback = st.text_input("Type 'approve' or give feedback")
        if st.button("Submit"):
            result = finalGraph.invoke(Command(resume=feedback), config=config)
            if not finalGraph.get_state(config).interrupts:
                st.session_state.finished = True
            st.rerun()
    else:
        st.session_state.finished = True
        st.rerun()

else:
    final_state = finalGraph.get_state(config).values
    st.success("Blog finalized!")
    st.write(final_state.get("editor_output", ""))
    if st.button("Start a new blog"):
        for key in ["thread_id", "started", "finished"]:
            del st.session_state[key]
        st.rerun()
