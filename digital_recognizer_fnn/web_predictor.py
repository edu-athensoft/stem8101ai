"""
this module is a demo for seq2seq question-answer. It will
launch a simple website that receives users' requests and return answers

Yimin Nie: ymnie888@gmail.com
"""
from flask import Flask, request, render_template, jsonify, make_response
from digital_recognizer_fnn.model_fnn import FFNN
from digital_recognizer_fnn.predictor_fnn import predict_single_img
from digital_recognizer_fnn import settings
import tensorflow as tf


app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
n_in = settings.n_in
n_out = settings.n_out
hidden_layers = settings.hidden_layers
nn = FFNN(n_in=n_in, n_out=n_out, hidden_layers=hidden_layers, reg_type='multiclass',
          optimizer=tf.train.AdamOptimizer)


@app.route('/')
def home():
    """
    home landing page
    :return:
    """
    return render_template('home.html')


@app.route('/about')
def about():
    """
    about info
    :return:
    """
    return "This is an internal test for chat bot"


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        my_img = request.form['img_name']
        res = predict_single_img(my_img, settings.n_in, settings.n_out, settings.hidden_layers)
        return res
    #     return render_template('results.html',
    #                            content=my_img,
    #                            prediction=pred_label,
    #                            probability=round(pred_proba * 100, 2))
    # return render_template('img_recog.html', form=form)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    """
    this is the main function to run the server
    :return:
    """
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
