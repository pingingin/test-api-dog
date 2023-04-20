from flask import Flask, request
from predict import dog_breed_classifier

app = Flask(__name__)

@app.route('/')
def home():
   return "hello dog breed"

@app.route('/api/dog_breed', methods=['GET','POST'])
def dog_breed_classification():
    if request.method == 'GET':
        data = request.args
        data.to_dict(flat=False)
        url_img = data['url_img']
        result = dog_breed_classifier(url_img)
        return result

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=6000)
    # app.debug = False
    # app.run()