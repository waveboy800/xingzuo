import time
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import load_tools, initialize_agent, AgentType
import streamlit as st
import datetime
import os
from cachetools import LRUCache
from serpapi import GoogleSearch


# Load env vars
load_dotenv()


llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=2048)

cache = LRUCache(maxsize=128)


def get_mouthdate(today):
    key = (today)
    if key in cache:
        return cache[key]

    # Prompt templates
    mouthdate_template = PromptTemplate(
        input_variables=['today'],
        template='根据{today} 查询对应的月日是什么'
    )

    mouthdate_chain = LLMChain(llm=llm, prompt=mouthdate_template, verbose=True, output_key='mouthdate')

    reply = mouthdate_chain({'today': today})

    result = reply['mouthdate']
    if not isinstance(result, dict) or 'text' not in result:
        result = {'text': '未找到历史记录'}
    cache[key] = result
    return result

def get_history(mouthdate):
    key = (mouthdate)
    if key in cache:
        return cache[key]

    # Prompt templates
    history_template = PromptTemplate(
        input_variables=['mouthdate'],
        template='根据{mouthdate} 查询当天发生的历史是什么'
    )

    history_chain = LLMChain(llm=llm, prompt=history_template, verbose=True, output_key='history')

    reply = history_chain({'mouthdate': mouthdate})

    result = reply['history']
    if not isinstance(result, dict) or 'text' not in result:
        result = {'text': '未找到历史记录'}
    cache[key] = result
    return result


def get_constellation(birthday, today):
    key = (birthday, today)
    if key in cache:
        return cache[key]

    # Prompt templates
    constellation_template = PromptTemplate(
        input_variables=['birthday'],
        template='根据您的出生日期 {birthday} 查询您是什么星座'
    )

    constellation_chain = LLMChain(llm=llm, prompt=constellation_template, verbose=True, output_key='constellation')

    reply = constellation_chain({'birthday': birthday})

    result = reply['constellation']
    cache[key] = result
    return result


def get_astrology_horoscope(constellation, today):
    key = (constellation, today)
    if key in cache:
        return cache[key]

    # Prompt templates
    horoscope_template = PromptTemplate(
        input_variables=['constellation', 'today'],
        template='请假设你是星座运势专家，给出 {constellation}在{today}的本日运势'
    )

    horoscope_chain = LLMChain(llm=llm, prompt=horoscope_template, verbose=True, output_key='horoscope')
    
    reply = horoscope_chain({'constellation': constellation, 'today': today})

    result = reply['horoscope']
    cache[key] = result
    return result
    

def on_submit(prompt):
    with st.spinner('正在查询，请稍候...'):
        today = str(datetime.date.today().strftime("%Y-%m-%d"))
        today = today[:10]

        constellation = get_constellation(prompt, today)
        horoscope = get_astrology_horoscope(constellation, today)
        mouthdate = get_mouthdate(today)
        today_history = get_history(mouthdate)

        with st.expander("您是什么星座"):
            st.info(constellation)

        with st.expander("您本日星座运势"):
            st.info(horoscope)

        with st.expander("相同月日发生的历史"):
            if today_history:
                st.info(f"相同月日发生的历史：{today_history['text']}")
            else:
                st.warning("未找到该历史")



st.title('查询您的本日星座运势')

prompt = st.text_input('输入您的出生日期，可以查询您本日星座运势', key='prompt', max_chars=12)

if prompt:
    on_submit(prompt)