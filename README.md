# recommender-system
Recommender System implemented with Item and User based Collaborative filtering


## Test Environment
- OS: windows 10
- language: python 3.7.3

## Requirements
- numpy 1.16.4

## How to run program
```𝐩𝐲𝐭𝐡𝐨𝐧  𝐫𝐞𝐜𝐨𝐦𝐦𝐞𝐧𝐝𝐞𝐫.𝐩𝐲  [𝐛𝐚𝐬𝐞 𝐟𝐢𝐥𝐞 𝐧𝐚𝐦𝐞]  [𝒕𝒆𝒔𝒕 𝒇𝒊𝒍𝒆 𝒏𝒂𝒎𝒆]```

## Goal
  - traing data를 사용해 test data의 각 case[user, item]의 movie rating(1~5점)을 예측
  
## Prediction Algorithm
  - Item based collaborative filtering
    - 모든 item과 target item 사이의 similarity를 구함
    - item rating의 cosine similarity 사용    
![sim1](https://user-images.githubusercontent.com/37183417/63418053-6949bc80-c43d-11e9-8395-c697e28c148a.JPG)
    - target user가 이미 rating한 item 중에서 similarity를 기준으로 target item의 neighbor item들을 구함
    - predicted rating value는 neighbor item들의 rating value의 weighted sum(similarity)
    
  - User based collaborative filtering
    - target item을 rating한 다른 user(neighbor user)와 target user간의 similarity를 구함
    - user의 average로 보정한 similarity 사용
![sim2](https://user-images.githubusercontent.com/37183417/63418057-6bac1680-c43d-11e9-8b99-11e6adbb801b.JPG)
    - predicted rating value는 neighbor user의 target item에 대한 rating value의 weighted sum(similarity)
    
최종적인 predicted value는 두가지 방법의 결과물을 weighted sum(1:1)하여 산출
 
## Test Method and Result
- Test Method


  예측결과의 test method로 RMSE(Root Mean Square Error) 사용
  
- Result


    test|test data|RMSE
    ----|---------|---
    test1|u1.test|0.9856
    test2|u2.test|0.9718
    test3|u3.test|0.9621
    test4|u4.test|0.9610
    test5|u5.test|0.9676
    
  
    
  

