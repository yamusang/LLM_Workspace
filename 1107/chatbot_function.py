import uuid

# 메뉴 데이터
menu = {
    "슈퍼 슈프림": 10000,
    "아이리쉬 포테이토": 11000,
    "스파이스 치킨": 12000,
    "슈프림 알프레도": 13000
}

size_price = {"레귤러": 0, "미디엄":2000, "라지": 5000, "패밀리": 10000}

# function call 함수
def get_menu() -> dict:
    return {"menu":menu, "size":size_price}

# 가격 계산
def get_order_price(pizza_name:str, pizza_size:str="레귤러",quantity:int=1)->dict:
    base = menu.get(pizza_name,0)    # 없는 피자이름이면 0
    extra = size_price.get(pizza_size, 0)   
    unit_price = base + extra
    sub_total = unit_price * quantity
    return {
        "pizza_name": pizza_name,
        "pizza_size": pizza_size,
        "quantity": quantity,
        "unit_price": unit_price,   # 기본가격 + 사이즈 + (토핑)
        "sub_total": sub_total      # x 수량
    }

# 구매 확정 : orders 는 주문 내역을 저장하는 딕셔너리
def set_order(pizza_name:str, pizza_size:str="레귤러",quantity:int=1, orders:dict=None)->dict:
    if orders is None:  # 추가주문이면 None 이 아닙니다.
        orders = {"order_id": None, "content":[], "payment":0}

    base = menu.get(pizza_name,0)    # 없는 피자이름이면 0
    extra = size_price.get(pizza_size, 0)   
    unit_price = base + extra
    sub_total = unit_price * quantity

    if not orders.get("order_id"):  # 처음 주문이면 None
        orders["order_id"] = str(uuid.uuid4())   # 주문번호 난수 문자열로 생성

    orders['content'].append({
        "order": {"pizza_name": pizza_name,"pizza_size": pizza_size, "quantity": quantity },
        "unit_price": unit_price,   # 기본가격 + 사이즈 + (토핑)
        "sub_total": sub_total 
        })

    orders['payment'] = sum(item['sub_total'] for item in orders['content'])
    return {
        "pizza_name": pizza_name,
        "pizza_size": pizza_size,
        "quantity": quantity,
        "unit_price": unit_price,   # 기본가격 + 사이즈 + (토핑)
        "sub_total": sub_total      # x 수량
    }

# 주문 완료 
def complete_order(orders:dict) -> dict:
    if not orders or ("order_id" not in orders):
        return {'error': '주문 내역이 존재하지 않습니다.'}
    if not orders.get('content'):    # content 리스트가 비어있으면
        return {'error': '주문 내역이 존재하지 않습니다.'}
    last_order = orders.copy()

    # orders 초기화
    orders.clear()   # dict 의 모든 항목 삭제
    orders.update({"order_id": None, "content":[], "payment":0})
    return {'message': '주문이 완료되었습니다.','orders': last_order}

# function call 기능은 위의 3개 함수를 제어하는 함수를 추가
# 테스트 1: process_message 함수 없이도 동작하는지 !
def process_message(msg:str, orders:dict):
    msg = msg.strip()
    if '메뉴' in msg:
        m = get_menu()

        # 이것이 없으면 실행할 때 마다 메뉴 출력을 gpt 가 다르게 합니다.
        menu_str = "\n".join([f'{k} : {v}원' for k,v in m['menu'].items()])
        size_str = "\n".join([f'{k} : {v}원' for k,v in m['size'].items()])
        return f"현재 메뉴 입니다. : \n{menu_str} \n{size_str}",orders
    elif '주문' in msg:
        orders = set_order('슈퍼 수프림','레귤러',1,orders)
        return f'슈퍼 수프림 1판을 주문했습니다.\n 총 가격 {orders['payment']} 원입니다.',orders
    elif '완료' in msg:
        result = complete_order(orders)
        if 'error' in result: # dict result 에 key 'error' 가 있으면
            return result['error'],orders
        else:
            payment = result['orders']['payment']
            return f'{result['message']}\n 총 결제금액: {payment} 원 입니다.',orders
    else:
        return "원하시는 피자 메뉴나 주문을 말씀해 주세요.",orders 