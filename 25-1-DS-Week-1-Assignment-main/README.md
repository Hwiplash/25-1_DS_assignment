# 25-1-DS-Week-1-Assignment

원작자: 24기 DS 부팀장 박동연

안녕하세요 DS팀에 오신 것을 정말 환영합니다.

OT인데 과제가 있다니 죄송하네요

내리갈굼입니다 :)

### 본격적인 과제 시작 이전에 짚어 두셔야 할 점

- 해당 과제는 Attention의 메커니즘은 물론, **Transformer의 구조를 완벽히 이해**하고 있다는 가정 하에 풀 수 있고, 그래야만 합니다.
    - 혹시 이해가 부족하다면 과제를 시작하기 전 꼭 공부해주세요.
- ChatGPT, Github Copilot 등 문명의 이기 없이 구현해주세요.
    - 여러분의 양심에 맡기겠습니다. DS 와서 Transformer도 안 만들어봤다..? ㅠㅠ

## 제출 파일 구조

```bash
my_transformer/
├── __init__.py
├── attention.py
├── decoder.py
├── embeddings.py
├── encoder.py
├── feedforward.py
├── my_transformer.py
├── normalization.py
└── residual.py

wandb/
main.ipynb

```


## 제출 방법

1. git clone https://github.com/imsuviiix/25-1-DS-Week-1-Assignment.git
2. 각 모듈 속 비어있는 **#TODO를 완성**하기
    1. 전부 typing이 되어 있고, typing을 안 고치시는게 전체 모듈을 수정하지 않는 방법
    2. one line! 은 진짜 한 줄 컷. return이 None 아니라면 return문 하나만 써주면 끝
3. main.ipynb 실행
4. 위 파일 구조 내 **전체 파일**을 본인 레포에 제출, 캐시는 제외해주세요
5. 여기에 WandB 대시보드를 캡쳐해서, 자신이 왜 하이퍼 파라미터를 다음과 같이 설정했는지 짤막한 리포트를 작성해주세요!

## 힌트 및 팁

1. 모듈들을 실제 transformer가 동작하는 순서대로 구현하면 수월
    1. 실제 동작하는 순서가 애매하면 모듈 파일의 의존성 참고
2. torch, math, torch.nn 에서 기본으로 제공해주는 함수들 적극 이용
    1. matmul, softmax, transpose, Dropout….
    2. 안 쓰시고 구현하셔도 됩니다.. 리스펙..! 
3. import 부분은 일부러 안 지웠습니다.
4. decoder.py가 Transformer의 꽃이라 텅 비워놨는데, 너무 어려우시면 encoder의 코드를 참고하시면 도움이 될 것 같습니다.
5. pytorch 에 대해 잘 모르시면 
    1. https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    2. https://pytorch.org/docs/stable/tensors.html

## 기한

~03/17 23:59
