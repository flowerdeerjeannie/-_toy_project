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
          return render_template("predict_result.html", review=review)

#method는 from flask의 request에 의해 사용할 수 있는 것임
#get으로 들어오는지 post로 들어오는지 알 수 없기 때문에 판단해줄 context가 필요함

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