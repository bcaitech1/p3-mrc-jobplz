# boostcamp AI tech P3 MRC (team: jobplz)
부스트 캠프 AI tech 프로젝트3 기계독해 팀 jobplz의 코드 repo입니다.

# 개요

"한국에서 가장 오래된 나무는 무엇일까?" 이런 궁금한 질문이 있을 때 검색엔진에 가서 물어보신 적이 있을텐데요, 요즘엔 특히나 놀랍도록 정확한 답변을 주기도 합니다. 어떻게 가능한 걸까요?

질의 응답(Question Answering)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 그 중에서도 Open-Domain Question Answering 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가가 되어야하기에 더 어려운 문제입니다.

![image](https://user-images.githubusercontent.com/9896149/120650563-e03c3200-c4b8-11eb-8386-e106b44b2c89.png)


본 대회에서 우리가 만들 모델은 두 stage로 구성되어 있습니다. 첫 번째 단계는 질문에 관련된 문서를 찾아주는 "retriever"단계이고요, 다음으로는 관련된 문서를 읽고 간결한 답변을 내보내 주는 "reader" 단계입니다. 이 두 단계를 각각 만든 뒤 둘을 이으면, 어려운 질문을 던져도 척척 답변을 해주는 질의응답 시스템을 여러분 손으로 직접 만들게 됩니다. 더 정확한 답변을 내주는 모델을 만드는 팀이 우승을 하게 됩니다.

![image](https://user-images.githubusercontent.com/9896149/120650591-e6caa980-c4b8-11eb-8b78-de38ec4fdd79.png)

# Repository 설명

각 폴더에는 각 팀원의 코드가 구현되어 있습니다.

각 팀원의 결과물을 hard voting하여 최종 스코어 **EM : 63.89%, F1 :	76.45%** 를 달성하였습니다.

# [프로젝트 개요](https://github.com/bcaitech1/p3-mrc-jobplz/blob/main/project_overview.pdf)

# 팀원
|Name|Code Directory|Github|
|------|---|---|
|김원배|[wonbae](wonbae)|[Github](https://github.com/wonbae)|
|한현우|[hyunwoo](hyunwoo)|[Github](https://github.com/CodeNinja1126)|
|박상기|[SangKi](SangKi)|[Github](https://github.com/sangki930)|
|최석환|[cshwan130](cshwan130)|[Github](https://github.com/loyalsp13)|
|장동재|[dongjae](dongjae)|[Github](https://github.com/DongjaeJang)|
|김명수|[myeongsoo](myeongsoo)|[Github](https://github.com/Kim-Myeong-Soo)|
