import os
import json
import uuid
import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv
import boto3
import faiss
import numpy as np

load_dotenv()

st.set_page_config(page_title="AI Virtual Agent", page_icon="🤖")
st.title("🤖 AI Virtual Agent")


def get_api_key():
    try:
        value = st.secrets["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"]
    except Exception:
        value = os.environ.get("OPENAI_API_KEY")
    return value.strip() if isinstance(value, str) else None


def get_aws_setting(key):
    try:
        value = st.secrets[key]
    except Exception:
        value = os.environ.get(key)
    return value.strip() if isinstance(value, str) else None


def has_aws_credentials():
    return bool(get_aws_setting("AKIA3G6DSZYNOOXI7SZ") and get_aws_setting("dh41ncM3cUc6LyfQv45DFlHZvrp6rc0ioeuEVT"))


def get_nova_client(region_name="us-east-1"):
    return boto3.client(
        "nova-act",
        region_name=region_name,
    )


def start_nova_workflow(workflow_definition_name, region_name="us-east-1", model_id=None):
    client = get_nova_client(region_name=region_name)
    params = {
        "workflowDefinitionName": workflow_definition_name,
        "clientToken": str(uuid.uuid4()),
    }
    if model_id:
        params["modelId"] = model_id
    return client.create_workflow_run(**params)


def create_nova_session(workflow_definition_name, workflow_run_id, region_name="us-east-1"):
    client = get_nova_client(region_name=region_name)
    return client.create_session(
        workflowDefinitionName=workflow_definition_name,
        workflowRunId=workflow_run_id,
        clientToken=str(uuid.uuid4()),
    )


def invoke_nova_act_step(
    workflow_definition_name,
    workflow_run_id,
    session_id,
    act_id,
    region_name="us-east-1",
    previous_step_id=None,
    call_results=None,
):
    client = get_nova_client(region_name=region_name)
    params = {
        "workflowDefinitionName": workflow_definition_name,
        "workflowRunId": workflow_run_id,
        "sessionId": session_id,
        "actId": act_id,
    }
    if previous_step_id:
        params["previousStepId"] = previous_step_id
    if call_results is not None:
        params["callResults"] = call_results
    return client.invoke_act_step(**params)


def inspect_nova_api(region_name="us-east-1"):
    """Return a summary of the nova-act client's available operations.

    This is displayed in the Streamlit UI for debugging and confirmation.
    """
    client = get_nova_client(region_name=region_name)
    ops = list(client.meta.service_model.operation_names)
    # Provide a small sample of operation input/output member names for quick inspection
    details = {}
    for op_name in ops:
        try:
            op_model = client.meta.service_model.operation_model(op_name)
            input_members = []
            output_members = []
            if op_model.input_shape:
                input_members = list(op_model.input_shape.members.keys())
            if op_model.output_shape:
                output_members = list(op_model.output_shape.members.keys())
            details[op_name] = {"input": input_members, "output": output_members}
        except Exception:
            details[op_name] = {"input": None, "output": None}
    return {"operations": ops, "details": details}


api_key = get_api_key()
if not api_key:
    st.warning("OpenAI API key not found. Set OPENAI_API_KEY in Streamlit secrets or environment.")
    if os.path.exists(".env"):
        st.info("Your local `.env` file exists. Ensure it contains a valid OPENAI_API_KEY without quotes.")
    else:
        st.info(
            "Create a `.env` file from `.env.example` and add your OpenAI key:\n\n"
            "copy .env.example .env\n\n"
            "Then edit `.env` and set OPENAI_API_KEY=sk-..."
        )
    # Allow reloading .env at runtime without restarting Streamlit
    if st.button("Reload config"):
        try:
            load_dotenv(override=True)
            st.success("Reloaded .env.")
            # Some Streamlit versions expose `experimental_rerun`; others do not.
            if hasattr(st, "experimental_rerun"):
                try:
                    st.experimental_rerun()
                except Exception:
                    st.info("Reload requested; please restart the Streamlit server to apply changes.")
            else:
                st.info("Reloaded .env. Please restart the Streamlit server to apply changes.")
        except Exception as e:
            st.error(f"Reload failed: {e}")

    st.stop()

openai.api_key = api_key
client = OpenAI(api_key=api_key)


def load_faiss_index(index_path="rag_index/index.faiss", meta_path="rag_index/meta.jsonl"):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(index_path)
    metas = []
    with open(meta_path, "r", encoding="utf-8") as mf:
        for line in mf:
            metas.append(json.loads(line))
    return index, metas


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("RAG / Retrieval settings")
    rag_enabled = st.checkbox("Enable retrieval-augmented generation (RAG)", value=False)
    index_path = st.text_input("FAISS index path", value="rag_index/index.faiss")
    meta_path = st.text_input("Metadata path", value="rag_index/meta.jsonl")
    top_k = st.number_input("Top K retriever", min_value=1, max_value=20, value=4)

    st.header("AWS Nova workflow")
    workflow_definition_name = st.text_input("Nova workflow name", value="premkumar")
    model_id = st.text_input("Nova model ID (optional)", value="")
    nova_region = st.text_input("AWS region", value="us-east-1")
    workflow_run_id = st.text_input(
        "Workflow run ID",
        value=st.session_state.get("nova_workflow_run_id", ""),
    )
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.get("nova_session_id", ""),
    )
    act_id = st.text_input("Act ID to invoke", value=st.session_state.get("nova_act_id", ""))
    previous_step_id = st.text_input("Previous step ID (optional)", value="")

    if rag_enabled:
        idx, metas = load_faiss_index(index_path=index_path, meta_path=meta_path)
        if idx is None:
            st.warning("FAISS index or metadata not found at provided paths. Run `rag_ingest.py` first.")
        else:
            st.success(f"Loaded FAISS index with {len(metas)} chunks")
    else:
        idx, metas = None, None

    if st.button("Start Nova workflow run"):
        if not has_aws_credentials():
            st.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in environment or Streamlit secrets.")
        elif not workflow_definition_name:
            st.error("Enter a valid Nova workflow name.")
        else:
            try:
                result = start_nova_workflow(
                    workflow_definition_name,
                    region_name=nova_region,
                    model_id=model_id or None,
                )
                workflow_run_id = result.get("workflowRunId")
                st.session_state["nova_workflow_run_id"] = workflow_run_id
                st.success("Started Nova workflow run")
                st.json(result)
            except Exception as e:
                st.error(f"Nova workflow start failed: {e}")

    if st.button("Create Nova session"):
        if not has_aws_credentials():
            st.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in environment or Streamlit secrets.")
        elif not workflow_definition_name or not workflow_run_id:
            st.error("Enter a valid Nova workflow name and workflow run ID.")
        else:
            try:
                result = create_nova_session(
                    workflow_definition_name,
                    workflow_run_id,
                    region_name=nova_region,
                )
                session_id = result.get("sessionId")
                st.session_state["nova_session_id"] = session_id
                st.success("Created Nova session")
                st.json(result)
            except Exception as e:
                st.error(f"Nova session creation failed: {e}")

    if st.button("Invoke Nova act step"):
        if not has_aws_credentials():
            st.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in environment or Streamlit secrets.")
        elif not workflow_definition_name or not workflow_run_id or not session_id or not act_id:
            st.error("Provide workflow name, run ID, session ID, and act ID.")
        else:
            try:
                result = invoke_nova_act_step(
                    workflow_definition_name,
                    workflow_run_id,
                    session_id,
                    act_id,
                    region_name=nova_region,
                    previous_step_id=previous_step_id or None,
                    call_results=[],
                )
                st.success("Invoked Nova act step")
                st.json(result)
            except Exception as e:
                st.error(f"Nova invoke step failed: {e}")

    # Diagnostics: inspect nova-act operations and shapes
    if st.button("Inspect Nova API"):
        try:
            if not has_aws_credentials():
                st.warning("AWS credentials not found — inspection will still attempt to list operations but API calls may fail.")
            info = inspect_nova_api(region_name=nova_region)
            st.subheader("Nova operations")
            st.write(info["operations"])
            st.subheader("Operation details (input/output members)")
            st.json(info["details"])  
        except Exception as e:
            st.error(f"Failed to inspect Nova API: {e}")

prompt = st.chat_input("Ask me anything...")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_embedding_openai(text, model="text-embedding-3-small"):
    resp = openai.Embedding.create(model=model, input=text)
    return np.array(resp["data"][0]["embedding"], dtype=np.float32)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_snippets = []
    if rag_enabled and 'idx' in locals() and idx is not None:
        try:
            q_emb = get_embedding_openai(prompt)
            q_emb = q_emb.reshape(1, -1)
            distances, indices = idx.search(q_emb, k=top_k)
            for i in indices[0]:
                if i < 0 or i >= len(metas):
                    continue
                context_snippets.append(metas[i]["text"])
        except Exception as e:
            st.error(f"Retriever failed: {e}")

    # Compose the prompt for the LLM
    if context_snippets:
        context_combined = "\n\n---\n\n".join(context_snippets)
        final_prompt = f"Use the following context to answer the question. If the context isn't relevant, answer from general knowledge.\n\nContext:\n{context_combined}\n\nQuestion: {prompt}"
    else:
        final_prompt = prompt

    with st.spinner("Thinking..."):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=final_prompt,
            )
        except Exception as e:
            st.error(f"LLM request failed: {e}")
            response = None

    answer = None
    if response is not None:
        answer = getattr(response, "output_text", None)
        if not answer:
            try:
                answer = response.output[0].content[0].text
            except Exception:
                answer = str(response)

    if not answer:
        answer = "(no response)"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
