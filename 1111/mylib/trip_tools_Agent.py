from dotenv import load_dotenv
import os
import json
import logging
import requests
from datetime import datetime
import pytz
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_community import GoogleSearchAPIWrapper

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# tool 함수 
# python trip_tools_Agent.py 실행 될때
if __name__=='__main__':
    from trip_tools import get_current_datetime, get_current_weather, google_search, calculate
else:   # 다른 소스파일에서 import 할때
    from mylib.trip_tools import get_current_datetime, get_current_weather, google_search, calculate

# 에이전트 설정
from langchain.agents import create_agent    # 추가

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

# 도구 목록
tools = [
    get_current_datetime,
    get_current_weather,
    google_search,
    calculate,
]

system_prompt = """당신은 여행 및 일상의 계획에 매우 도움이 되는 어시스턴트입니다.
사용자의 요청에 따라 다음의 도구들을 활용하여 답변합니다.
1. 현재 날짜 및 시간 조회
2. 실시간 날씨 조회
3. 구글 검색을 통한 정보 검색
4. 수학 계산

사용자의 질문을 분석하여 가장 적절한 도구를 선택하고 조합하여 사용하세요.
한국어로 친절하고 상세하게 답변하세요.
여러 도구가 필요하면 순차적으로 사용하세요."""

# 에이전트와 InMemorySaver 사용 (랭체인 버전 1.0.5 )
#    ㄴ pip install langgraph  
from langgraph.checkpoint.memory import InMemorySaver
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()   # 수정
)

def run_agent_memory(user_input:str, thread_id:str):   # 수정
    """thread_id 설정하기"""
    config = {"configurable": {"thread_id": thread_id}}    # 수정

    """에이전트 실행"""
    print(f"\n{'='*70}")
    print(f"👤 사용자: {user_input}")
    print(f"{'='*70}")
    
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config)   

        print(f"\n{'='*70}")
        print(f"🤖 어시스턴트:\n{response}")
        print(f"{'='*70}\n")
        
        final_answer = response['messages'][-1].content

        return final_answer

    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}\n")
        return None

# 테스트
if __name__ == "__main__":   # import 할 때는 실행 안 합니다.
#     # 5. 정보 검색과 날씨
    final_answer=run_agent_memory("충북 괴산의 날씨를 알려주고, 2025년 11월 괴산의 축제 일정을 알려줘",thread_id='abc123')
    print(f'💬 : {final_answer}')
    final_answer=run_agent_memory("방금 내가 물어본 지역이 어디지?",thread_id='abc123')
    print(f'💬 : {final_answer}')