# -*- coding: utf-8 -*-
"""
LangChain 날씨 조회 봇 - tool_calls  버전
ToolMessage 사용 연습
"""

from dotenv import load_dotenv
import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
import logging
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================
# 1. 도구 함수 정의
# ============================================

@tool
def get_coordinates(location: str) -> str:
    """
    주어진 장소 이름을 기반으로 위도와 경도를 조회합니다.
    OpenStreetMap Nominatim API를 사용합니다.
    
    Args:
        location: 조회할 장소명 (예: '서울', 'LA', 'Paris')
    
    Returns:
        "위도,경도" 형식의 문자열 또는 에러 메시지
    """
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {
            "User-Agent": "LangChainWeatherBot/1.0 (weather@bot.com)"
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        res = response.json()
        
        if res:
            lat = res[0]['lat']
            lon = res[0]['lon']
            logger.info(f"{location}의 위도: {lat}, 경도: {lon}")
            return f"{lat},{lon}"
        else:
            return f"'{location}'의 좌표를 찾을 수 없습니다."
            
    except requests.exceptions.Timeout:
        logger.error("요청 시간 초과")
        return "요청 시간 초과. 잠시 후 다시 시도해주세요."
    except requests.exceptions.RequestException as e:
        logger.error(f"네트워크 오류: {e}")
        return "네트워크 오류가 발생했습니다."
    except Exception as e:
        logger.error(f"좌표 조회 중 오류: {e}")
        return "좌표를 조회하는 중 오류가 발생했습니다."


@tool
def get_weather_info(lat_lon: str) -> str:
    """
    위도와 경도를 기반으로 현재 날씨 정보를 조회합니다.
    OpenWeatherMap API를 사용합니다.
    
    Args:
        lat_lon: "위도,경도" 형식의 문자열
    
    Returns:
        포맷된 날씨 정보 문자열
    """
    try:
        lat, lon = lat_lon.split(',')
        api_key = os.getenv("OPENWEATHER_API_KEY", "bee5fad369e27a7ced91d32f284a1217")
        
        url = (
            f"http://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=kr"
        )
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        res = response.json()
        
        if res.get('cod') == '401':
            return "API 키가 유효하지 않습니다."
        
        if 'weather' in res:
            weather_desc = res['weather'][0]['description']
            temp = res['main']['temp']
            feels_like = res['main']['feels_like']
            humidity = res['main']['humidity']
            location_name = res.get('name', 'Unknown')
            
            result = (
                f"날씨: {weather_desc}\n" 
                f"기온(°C): {temp}°C\n"
                f"체감온도(°C): {feels_like}°C\n"
                f"습도(%): {humidity}%\n"
                f"지역: {res.get('name', 'Unknown')}"
                )
            logger.info(f"날씨 정보 조회 성공: {location_name}")
            return result
        else:
            return "날씨 정보를 불러올 수 없습니다."
            
    except ValueError:
        logger.error("잘못된 좌표 형식")
        return "좌표 형식이 잘못되었습니다. '위도,경도' 형식을 사용하세요."
    except requests.exceptions.Timeout:
        logger.error("날씨 API 요청 시간 초과")
        return "요청 시간 초과. 잠시 후 다시 시도해주세요."
    except requests.exceptions.RequestException as e:
        logger.error(f"네트워크 오류: {e}")
        return "네트워크 오류가 발생했습니다."
    except Exception as e:
        logger.error(f"날씨 정보 조회 중 오류: {e}")
        return f"오류가 발생했습니다: {str(e)}"


# ============================================
# 2. LLM 모델 및 도구 설정
# ============================================

def setup_weather_agent():
    """
    날씨 조회 에이전트를 설정합니다.
    
    Returns:
        tuple: (llm_with_tools, tools_dict)
    """
    llm = ChatOpenAI(model="gpt-4.1-mini")
    tools = [get_coordinates, get_weather_info]
    # 도구 설정된 LLM
    llm_with_tools = llm.bind_tools(tools)
    tools_dict = {
        'get_weather_info': get_weather_info,
        'get_coordinates': get_coordinates
    }
    
    logger.info("날씨 에이전트 설정 완료")
    return llm_with_tools, tools_dict


# ============================================
# 3. 도구 실행 
# ============================================

def execute_tools(llm_with_tools, tools_dict, response, messages):
    """
    LLM이 호출한 도구들을 실행하고 ToolMessage로 응답합니다.
    
    ✅ tool_call_id와 ToolMessage를  매칭
    
    Args:
        llm_with_tools: 도구가 바인딩된 LLM
        tools_dict: 도구 이름 - 함수 매핑 딕셔너리
        response: LLM의 응답 (tool_calls 포함)
        messages: 현재까지의 메시지 히스토리
    
    Returns:
        list: ToolMessage가 추가된 메시지 리스트
    """
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        logger.warning("도구 호출이 없습니다.")
        return messages
    
    logger.info(f"도구 호출 발견: {len(response.tool_calls)}개")
    
    for tool_call in response.tool_calls:
        try:
            #  중요: tool_call 구조 분해
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")  # ← 이 ID가 매칭 키
            
            logger.info(f"도구 실행: {tool_name}")
            logger.info(f"  - ID: {tool_call_id}")
            logger.info(f"  - Args: {tool_args}")
            
            if tool_name not in tools_dict:
                logger.error(f"알 수 없는 도구: {tool_name}")
                error_msg = ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=f"알 수 없는 도구입니다: {tool_name}"
                )
                messages.append(error_msg)
                continue
            
            # 도구 실행
            selected_tool = tools_dict[tool_name]
            tool_result = selected_tool.invoke(tool_args)
            
            logger.info(f"✅도구 실행 결과: {tool_result[:100]}...")  # 처음 100자만 로깅
            
            # 중요: ToolMessage로 감싸기
            # tool_call_id를 이용해 도구 호출과 응답을 매칭
            tool_message = ToolMessage(
                tool_call_id=tool_call_id,      # ← 도구 호출 ID와 매칭
                name=tool_name,                  # ← 도구 이름
                content=str(tool_result)         # ← 도구 실행 결과 (문자열)
            )
            messages.append(tool_message)
            
        except Exception as e:
            logger.error(f"도구 실행 중 오류: {e}")
            error_msg = ToolMessage(
                tool_call_id=tool_call.get("id", "unknown"),
                name=tool_call.get("name", "unknown"),
                content=f"도구 실행 중 오류가 발생했습니다: {str(e)}"
            )
            messages.append(error_msg)
    
    return messages


# ============================================
# 4. 메인 실행 함수
# ============================================

def ask_weather(user_question: str) -> str:
    """
    사용자 질문에 대해 날씨 정보를 제공합니다.
    
    Args:
        user_question: 사용자의 질문 (예: "LA 날씨는 어때?")
    
    Returns:
        LLM이 생성한 최종 답변
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🔵 사용자 질문: {user_question}")
    logger.info('='*60)
    
    try:
        # 1. 에이전트 설정 : 🔄 setup_weather_agent() 함수 호출
        llm_with_tools, tools_dict = setup_weather_agent()   
        
        # 2. 초기 메시지 구성
        messages = [
            SystemMessage(
                "너는 친절한 날씨 정보 제공 봇이야. "
                "사용자가 요청한 지역의 날씨를 정확하고 친절하게 알려줘. "
                "항상 한국어로 답변하고, 이모지를 사용해서 보기 좋게 표현해."
            ),
            HumanMessage(user_question),
        ]
        logger.info("메시지 초기화 완료")
        
        while True:
            # 3. LLM 호출
            response = llm_with_tools.invoke(messages)  # 사용자 메시지 전달
            if not getattr(response, "tool_calls", None):
                break

            messages.append(response)   # 4. 응답 기록
            logger.info(f"응답 받음 - tool_calls: {len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}개")
            # 5. 도구 실행 : 🔄 execute_tools() 함수 호출
            messages = execute_tools(llm_with_tools, tools_dict, response, messages)
            
            logger.info("🟢 messages:" + "\n\t".join(type(msg).__name__  for msg in messages))
            logger.info('='*60)
        
        return response.content
        
    except Exception as e:
        logger.error(f"🔴 오류 발생: {e}", exc_info=True)
        return f"오류가 발생했습니다: {str(e)}"


# ============================================
# 5. 테스트
# ============================================
