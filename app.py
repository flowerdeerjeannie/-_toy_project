from konlpy.tag import Okt
from flask import Flask, render_template, request

import os
import joblib
import re

app = Flask(__name__)
app.debug = True

tfidf_vector = None
model_lr = None

okt = Okt()

def tw_tokenizer(text):
     tokenizer_ko = okt.morphs(text)
     return tokenizer_ko

# 직렬화 된 모델을 가져오기 위해서 모델 형성때 사용한 tokenizer도 필요함 

def load_lr():
     global tfidf_vector, model_lr
     tfidf_vector = joblib.load(os.path.join(app.root_path, "model/tfidf_vect.pkl"))
     model_lr = joblib.load(os.path.join(app.root_path, "model/lr.pkl"))

# 누구에게나 어디에서나 운영체제를 타지 않는 일관된 동작을 위하여 경로를 os 상으로 설정해 줌

def lt_transform(review):
     review = re.sub(r"\d+", " ", review)
     test_matrix=tfidf_vector.transform([review])
     return test_matrix

@app.route("/predict", methods=["GET", "POST"]) 

def npl_predict(): 
     if request.method == "GET":
          return render_template("predict.html")
     else:
          review = request.form["review"]
          review_matrix = lt_transform(review)
          review_result = model_lr.predict(review_matrix)[0]
          review_result = "긍정" if review_result else "부정"
          result = {
               "review" : review,
               "review_result" : review_result
          }
          return render_template("predict_result.html", review=result)

#form["review"] 는 키 처럼 생각하면 됨
#method는 from flask의 request에 의해 사용할 수 있는 것 - predict의 <textarea class="form-control" rows="5" name="review"></textarea>와 매칭
#post는- <form action="/predict" method="post">와 매칭해서 동작
#get으로 들어오는지 post로 들어오는지 알 수 없기 때문에 판단해줄 context가 필요함
#result 받아서, predict_result.html 에서 받았는데 그걸 받아줄 애가 있어야지, html의 {{review}}
#review=result 키와 밸류의 매칭처럼 생각하면 됨
#model_lr.predict(review_matrix)[0]은 1아니면 0의 값인데 array 형태라서 빼내줄려고 [0]으로 정해주는거임
#A if 조건(참(1)이면 A 거짓(0)이면 B) else B

@app.route("/")
def index():
     # 해당 테스트코드는 잘 작동하지만 별도의 함수로 빠져나가야 됨
     test_str = "이 영화 재미있어요! 하하하 "
     test_matrix = lt_transform(test_str)
     result = model_lr.predict(test_matrix)
     print(result)
     return render_template("index.html")

# 변수, 함수, app.run 순서 지키기
 
if __name__ == "__main__":
     load_lr()
     app.run(port=5001)