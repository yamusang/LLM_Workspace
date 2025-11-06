from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client=OpenAI() 

def get_ai_response(question, functions=None):
   response = get_first_response(question=question)
   fn_name = getattr(response.choices[0].message.function_call, "name", None)
   if fn_name:
    func_response =  globals()[fn_name]()  
    followup_response=get_followup_response(fn_name,func_response)
    return followup_response.choices[0].message.content
   else:
    return response.choices[0].message.content

def get_followup_response(fn_name, func_response):
    followup_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.using locale language."},
            {"role": "user", "content": f'{fn_name} 함수를 실행한 결과 {func_response} 이용하여 최종 응답을 만들어줘.'}
        ],
        functions=myfunctions
    )
    return followup_response

def get_first_response(question):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are a helpful assistant.using locale language."},
      {"role": "user", "content": question}
    ],
    functions=myfunctions,
    function_call="auto"  # auto, {name: 'get_current_time'}
  )
  return response

from datetime import datetime
def get_current_time():
  now = datetime.now().strftime('%H:%M:%S')
#   print(f"현재 시간 : {now}")
  return now

def get_current_date():   # 나중에 추가해서 함수 선택 테스트
  now = datetime.now().strftime('%Y 년 %m 월 %d 일')
#   print(f"현재 시간 : {now}")
  return now

# functions (Chat Completions용)
myfunctions = [
    {
        "name": "get_current_time",
        "description": "현재 시간 출력 HH:MM:SS format",
        # "parameters": {"type": "object", "properties": {}}
    },
    { 
        "name": "get_current_date",
        "description": "현재 날짜 출력 YYYY 년 MM 월 DD 일 format",
        # "parameters": {"type": "object", "properties": {}}
    }
]