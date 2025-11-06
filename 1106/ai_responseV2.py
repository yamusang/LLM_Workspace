from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()
client=OpenAI() 

def get_ai_response(question, functions=None):
   response = get_first_response_tz(question=question)
   fn_name = getattr(response.choices[0].message.function_call, "name", None)
   if fn_name:
    # 함수 호출 : get_current_time_tz, get_current_date_tz는 인자가 필요합니다.
    #function_call.arguments를 dict로 변환
    
    tz = json.loads(response.choices[0].message.function_call.arguments).get("timezone", "Asia/Seoul")
    func_response =  globals()[fn_name]()  
    followup_response=get_followup_response_tz(fn_name,func_response)
    return followup_response.choices[0].message.content
   else:
    return response.choices[0].message.content

def get_followup_response_tz(fn_name, func_response):
    followup_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.using locale language."},
            {"role": "user", "content": f'{fn_name} 함수를 실행한 결과 {func_response} 이용하여 최종 응답을 만들어줘.'}
        ],
        functions=myfunctions
    )
    return followup_response

def get_first_response_tz(question):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are a helpful assistant.using locale language."},
      {"role": "user", "content": question}
    ],
    functions=myfunctions,
    function_call="auto"  
  )
  return response

from datetime import datetime
import pytz
def get_current_time(timezone='Asia/Seoul'):
  tz = pytz.timezone(timezone) # str을 타임존 객체로 변경 
  now = datetime.now().strftime('%H:%M:%S')
  print(f"🔄log: 현재 시간(tz) : {now} {tz}")
  return now

def get_current_date(timezone='Asia/Seoul'):
  tz = pytz.timezone(timezone) # str을 타임존 객체로 변경  
  now = datetime.now().strftime('%Y 년 %m 월 %d 일')
  print(f"🔄log: 현재 시간(tz) : {now} {tz}")
  return now

# functions (Chat Completions용)
myfunctions = [
    {
        "name": "get_current_time_tz",
        "description": "현재 시간 출력 HH:MM:SS format",
        # 함수의 인자를 정의
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Time zone in 'Area/Location' format, e.g., 'Asia/Seoul', 'America/New_York'. Default is 'Asia/Seoul'."
                }
            },
            "required": ['timezone']
        }
    },
    { 
        "name": "get_current_date_tz",
        "description": "현재 날짜 출력 YYYY 년 MM 월 DD 일 format",
        # "parameters": {"type": "object", "properties": {}}
    }
]