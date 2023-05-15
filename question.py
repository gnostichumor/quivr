import anthropic
import streamlit as st
from streamlit.logger import get_logger
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatAnthropic
from langchain.vectorstores import SupabaseVectorStore
from prompt_manager import get_all_prompt_titles, get_prompt_from_title, find_input_variables

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
openai_api_key = st.secrets.openai_api_key
anthropic_api_key = st.secrets.anthropic_api_key
logger = get_logger(__name__)


def count_tokens(question, model):
    count = f'Words: {len(question.split())}'
    if model.startswith("claude"):
        count += f' | Tokens: {anthropic.count_tokens(question)}'
    return count


def chat_with_doc(model, vector_store: SupabaseVectorStore):
    question = st.text_area("## Ask a question")
    button = st.button("Ask")
    count_button = st.button("Count Tokens", type='secondary')
    if button:
        if model.startswith("gpt"):
            logger.info('Using OpenAI model %s', model)
            qa = ConversationalRetrievalChain.from_llm(
                OpenAI(
                    model_name=st.session_state['model'], openai_api_key=openai_api_key, temperature=st.session_state['temperature'], max_tokens=st.session_state['max_tokens']), vector_store.as_retriever(), memory=memory, verbose=True)
            result = qa({"question": question})
            logger.info('Result: %s', result)
            st.write(result["answer"])
        elif anthropic_api_key and model.startswith("claude"):
            logger.info('Using Anthropics model %s', model)
            qa = ConversationalRetrievalChain.from_llm(
                ChatAnthropic(
                    model=st.session_state['model'], anthropic_api_key=anthropic_api_key, temperature=st.session_state['temperature'], max_tokens_to_sample=st.session_state['max_tokens']), vector_store.as_retriever(), memory=memory, verbose=True, max_tokens_limit=102400)
            result = qa({"question": question})
            logger.info('Result: %s', result)
            st.write(result["answer"])

    if count_button:
        st.write(count_tokens(question, model))

def prompted_chat(model, prompt_db):
    col1, col2 = st.columns(2)

    with col1:
        prompts = get_all_prompt_titles(prompt_db)
        prompt_title = st.radio("Choose a prompt", prompts)
    with col2:
        if not prompt_title:
            st.write("No prompt selected")
        st.write(get_prompt_from_title(prompt_title, prompt_db))
    template = str(get_prompt_from_title(prompt_title, prompt_db))
    input_variables = find_input_variables(template)
    if input_variables == []:
        prompt = PromptTemplate(input_variables=[], template=template)
    else:
        prompt = PromptTemplate(input_variables=input_variables, template=template)

    question = st.text_area("## Ask a question")
    button = st.button("Ask")
    count_button = st.button("Count Tokens", type='secondary')
    if button:
        if model.startswith("gpt"):
            logger.info('Using OpenAI model %s', model)
            llm = OpenAI(model_name=st.session_state['model'], openai_api_key=openai_api_key, temperature=st.session_state['temperature'], max_tokens=st.session_state['max_tokens'])
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            result = llm_chain.run(question)
            logger.info('Result: %s', result)
            st.write(result)
        elif anthropic_api_key and model.startswith("claude"):
            logger.info('Using Anthropics model %s', model)
            qa = ConversationalRetrievalChain.from_llm(
                ChatAnthropic(
                    model=st.session_state['model'], anthropic_api_key=anthropic_api_key, temperature=st.session_state['temperature'], max_tokens_to_sample=st.session_state['max_tokens']), vector_store.as_retriever(), memory=memory, verbose=True, max_tokens_limit=102400)
            result = qa({"question": question})
            logger.info('Result: %s', result)
            st.write(result)

    if count_button:
        st.write(count_tokens(question, model))
