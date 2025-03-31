import re
from transformers import AutoTokenizer
if __name__=="__main__": 
    # GPT-2 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    def extract_answer_token_length(text):
        """
        주어진 텍스트에서 "A:" 이후의 텍스트를 추출하고,
        해당 부분의 토큰 길이를 반환합니다.
        """
        match = re.search(r"A:\s*(.*)", text, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            tokens = tokenizer.tokenize(answer_text)
            token_count = len(tokens)
            return answer_text, token_count
        else:
            return None, 0

    # 텍스트 파일 읽기 (예시 파일명: data.txt)
    with open("cqa/cqa_vanilla_10/cqa_vanilla_10_0/correct_data.txt", "r", encoding="utf-8") as file:
        content = file.read()

    # 각 데이터 블록은 "<|end_of_text|>" 로 구분되어 있다고 가정
    blocks = content.split("<|end_of_text|>")

    max_token_length = 0  # 전체 블록 중 최대 토큰 개수를 저장할 변수
    max_sentence=''
    gen_length=0

    for i, block in enumerate(blocks):
        # "A:" 이후 텍스트와 토큰 수 계산
        answer_text, answer_token_count = extract_answer_token_length(block)
        # 블록 전체의 토큰 수 계산
        block_tokens = tokenizer.tokenize(block)
        block_token_count = len(block_tokens)    
        
        # 최대 토큰 개수 갱신
        if block_token_count > max_token_length:
            max_token_length = block_token_count
            gen_length=answer_token_count
            max_sentence=answer_text

    print("전체 블록 중 최대 토큰 개수 (max token length):", max_token_length)
    print("전체 블록 중 최대 생성토큰 개수 (gen_length):", gen_length)
    print("전체 블록 중 최대 토큰 결과 (max_sentence):", max_sentence)
