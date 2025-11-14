"""
PDF 매뉴얼 기반 RAG 시스템 - Pinecone + LangChain 1.x
한글 특화 임베딩 모델 + 이미지 무시 
"""

import os
from typing import List
from dotenv import load_dotenv

# LangChain 1.x 
# PDF 파일을 로드하여 텍스트를 추출하는 문서 로더입니다. PyPDF2 기반으로 작동
# pdfplumber 활용 : PDF에서 더 정밀하게 한글 텍스트를 추출
#     ㄴ pip install langchain_huggingface pdfplumber 
# 긴 텍스트를 적절한 길이로 나누는 텍스트 분할기
# OpenAI의 GPT 모델을 LangChain에서 사용할 수 있도록 래핑한 클래스
# Pinecone 벡터 데이터베이스와 연동하여 임베딩 벡터를 저장하고 검색할 수 있게 해주는 벡터 저장소 객체
# 챗봇 프롬프트를 템플릿 형태로 구성
# 입력을 그대로 출력하는 단순한 Runnable 객체입니다. 파이프라인 구성 시 유용
# 모델의 출력 결과를 문자열로 파싱하는 파서
# 임베딩 모델을 추상화한 기본 클래스
# Hugging Face의 사전학습 임베딩 모델을 LangChain에서 사용할 수 있도록 해주는 클래스 (한글 임베딩 모델 지원)
from langchain_community.document_loaders import PyPDFLoader   
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
#from langchain_core.embeddings import embedding 상속받아서 Embedding 객체로 정의한 클래스
#↙koreanEmbeddings 속성 embeddings
class KoreanEmbeddings(Embeddings):
    """한글 특화 임베딩 모델 래퍼"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """
        한글 임베딩 모델 초기화
        
        Args:
            model_name: HuggingFace 모델명
            - "intfloat/multilingual-e5-large": 다국어 (한글 우수), 높은 성능
            - "intfloat/multilingual-e5-base": 경량 버전
            - "BAAI/bge-m3": 한글 성능 우수
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # GPU 사용 시 "cuda"
            encode_kwargs={"normalize_embeddings": True}
        )
    
    # pdf 텍스트를 벡터로 변환 : str -> list[float]
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서(문자열 리스트) 텍스트를 벡터로 변환"""
        return self.embeddings.embed_documents(texts)
    
    # 질문(쿼리)를 벡터로 변환
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트를 벡터로 변환"""
        return self.embeddings.embed_query(text)

#(1)pdf를 텍스트로 변환 -> 임베딩 -> 벡터 db에 저장
#(2)벡터 db에 저장된 인덱스에 조회 -> llm에 전달하여 사용자 응답
# PdfRAGSystem의 속성 : index_name 문자열, pc(파인콘 객체), embeddings,llm,vectorstore
class PdfRAGSystem:
    """PDF 매뉴얼 기반 RAG 시스템 (한글 특화)"""
    
    def __init__(
        self,
        index_name: str = "recipe-book-index",
        embedding_model: str = "intfloat/multilingual-e5-large",
        llm_model: str = "gpt-4.1-mini"
    ):
        self.index_name = index_name
        
        # Pinecone 초기화
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # 한글 특화 임베딩 모델 초기화
        print(f"Loading Korean embeddings model: {embedding_model}")
        self.embeddings = KoreanEmbeddings(model_name=embedding_model)
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0
        )
        
        # 벡터 스토어 초기화
        self.vectorstore = None

    def create_index(self, dimension: int = 1024):
        """
        Pinecone 인덱스 생성
        
        Note: 한글 모델(multilingual-e5-large)은 1024 차원 사용
        """
        #인덱스 이름 모두 가져오기
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        #기존 인덱스에 있는 index_name인지 검사. 없을 때만 인덱스 생성
        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Index created successfully!")
        else: #index_name이 존재한다면
            print(f"Index '{self.index_name}' already exists.")
#청킹을 위해 splitter를 생성 -> PDF 텍스트를 청킹
    def load_and_split_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 800,
        chunk_overlap: int = 200
    ) -> List:
        """
        PDF 파일 로드 및 청크 분할 (이미지 무시)
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 청크 크기 (한글은 500~ 800으로 조정)
            chunk_overlap: 청크 오버랩(chunk_size의 10~20%)
        """
        print(f"Loading PDF: {pdf_path}")
        
        # PDF 로더 
        loader = PDFPlumberLoader(pdf_path)
        # loader를 이용하여 텍스트 추출 -> Document 타입 객체로 변환
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages, type: {type(documents)}")
        
        # 옵션: 이미지 텍스트 제거: 빈 페이지나 텍스트 없는 페이지 필터링
        # documents = self._filter_image_only_pages(documents)
        print(f"After filtering image-only pages: {len(documents)} pages")
        
        # 텍스트 분할 (한글 최적화)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # 조각의 길이를 계산하는 함수
            separators=[  # 텍스트를 나눌 때 사용할 우선순위 구분자.순서대로 적용
                "\n\n",      # 단락 구분
                "\n",        # 줄 바꿈
                ". ",        # 마침표 (한글)
                "! ",        # 느낌표
                "? ",        # 물음표
                " ",         # 공백
                ""           # 문자 단위 : 위 구분자들로도 나눌 수 없을 경우, 최종 적용
            ]
        )
        #청킹(문서 쪼개기)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        return chunks

    #옵션 : 필수 사항은 아님.(빈페이지, 그림만 있는 페이지가 많으면 적용하기)
    def _filter_image_only_pages(self, documents: List) -> List:
        """
        이미지만 있는 페이지 필터링
        텍스트가 거의 없는 페이지(이미지 위주)를 제거
        """
        filtered_documents = []
        
        for doc in documents:
            # 페이지 텍스트 길이 확인
            text_length = len(doc.page_content.strip())
            
            # 최소 50자 이상의 텍스트가 있는 페이지만 유지
            if text_length >= 50:
                filtered_documents.append(doc)
            else:
                page_num = doc.metadata.get("page", "unknown")
                print(f"  Skipping page {page_num} (mostly image)")
        
        return filtered_documents

    def create_vectorstore(self, documents: List):
        """
        벡터 스토어 생성 및 문서 임베딩
        한글 특화 임베딩 모델 사용
        """
        print("Creating vector store and embedding documents with Korean model...")
        
        # 문서를 벡터로 생성하여 지정된 index 이름으로 벡터db에 저장
        self.vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        
        print("Vector store created successfully!")
        return self.vectorstore
    
    def load_vectorstore(self):
        """기존 벡터 스토어 로드"""
        print("Loading existing vector store...")
        
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        print("Vector store loaded!")
        return self.vectorstore

    def create_rag_chain(self):
        """RAG 체인 생성 (한글 최적화 프롬프트)"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_vectorstore() first.")
        
        # 임베딩된 검색(retriever) 객체 생성 : 파이프라인으로 사용
        # docs = vector_store.similarity_search(query=question[0], k=5, namespace="wiki-ns1") 는 query 로 유사도 검색
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  
        )
        
        # 한글 프롬프트 : docstring 에서는 f'' 없이 입력변수 {} 지정
        template = """당신은 제품 매뉴얼 기반 질의응답 전문가입니다.
주어진  바탕으로 사용자의 질문에 정확하고 상세하게 답변해주세요.

답변 시 주의사항:
- 컨텍스트에 정보가 있으면 그것을 기반으로 답변하세요.
- 컨텍스트에 없는 내용은 "제공된 매뉴얼에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
- 한글로 명확하고 이해하기 쉽게 설명하세요.
- 필요하면 단계별 설명을 제공하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        # ChatPromptTemplate : LangChain에서 대화형 LLM(Chat Model)을 위한 프롬프트 템플릿을 구성
        # from_template() : 프롬프트 문자열로 ChatPromptTemplate을 생성. 
        prompt = ChatPromptTemplate.from_template(template)
        '''
        또는
        prompt = ChatPromptTemplate.from_messages([
          ("system", template),
          ("human", "질문: {question}")
         ])
        '''
        
        # 문서 포맷팅 함수 : create_rag_chain 함수안에서 정의
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # RAG 체인 구성(랭체인 표현식) :  첫번째 | 두번쨰 | 세번쨰 | 네번쨰    의 파이프라인 구성
        rag_chain = (
            {
                "context": retriever | format_docs, 
                #체인안의 체인 : 사용자 쿼리를 검색할 수 있게 벡터화하여 유사도 검색 실행
                #                검색 실행 결과 중 PAGE_CONTENT 속성만 가져오기
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        rag_chain = self.create_rag_chain()
        response = rag_chain.invoke(question)
        return response
    
    def query_with_sources(self, question: str):
        """소스 문서와 함께 답변 생성"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        # 관련 문서 검색
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        docs = retriever.invoke(question)
        
        # 답변 생성
        answer = self.query(question)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }

#위의 PdfRagSystem 클래스를 이용하여 RAG 시스템 실행
def main(run:bool,pdf_path:str,index_name:str,questions:List[str]):
    """메인 실행 함수"""
    
    # RAG 시스템 초기화
    rag_system = PdfRAGSystem(
        index_name=index_name  , #"product-manual-index",
        embedding_model="intfloat/multilingual-e5-large",  # 한글 특화 모델
        llm_model="gpt-4.1-mini"
    )
    
    # 1. 인덱스 생성 (최초 1회만)
    rag_system.create_index(dimension=1024)  # multilingual-e5-large는 1024 차원
    
    # 2. PDF 로드 및 처리 (최초 1회만)
    # pdf_path = "cuckoo_레시피.pdf"  # PDF 파일 경로
    
    # 새로운 문서 임베딩하기
    if run:  # 최초 실행시 True로 변경. 참이면 index_name에 대해 업서트(create_vectorstore 메소드 실행)
        documents = rag_system.load_and_split_pdf(
            pdf_path=pdf_path,
            chunk_size=800,  # 한글용으로 조정
            chunk_overlap=200
        )
        rag_system.create_vectorstore(documents)
    else:#거짓이면
        # 기존 벡터 스토어 로드
        rag_system.load_vectorstore()
    
    # 3. 질의응답
    print("\n" + "="*60)
    print("PDF 매뉴얼 RAG 시스템 (한글 특화)")
    print("="*60)
    
    for question in questions:
        print(f"\n질문: {question}")
        print("-" * 60)
        
        # 답변만 받기
        answer = rag_system.query(question)
        print(f"답변: {answer}\n")
    

