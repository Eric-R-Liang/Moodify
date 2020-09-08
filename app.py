import flask
from Model import predict
from Model.Mood_predict import predict_mood

app = flask.Flask(__name__,template_folder='templates')


@app.route('/',methods=['GET','POST'])
def main():
    global user_input
    global mood
    user_mood = None
    if flask.request.method == 'GET':
        return (flask.render_template('main2.html',flag=True))
    if flask.request.method == 'POST':
        if (flask.request.form['submit'] == "Submit Mood"):
            user_input_mood = flask.request.form["mood"]
            user_mood = predict_mood(user_input_mood)
            return flask.render_template('main2.html',flag = False,user_mood = user_mood,user_input=None)
        elif (flask.request.form['submit'] == "Submit Lyrics"):
            user_artist_choice = flask.request.form["Artists"]
            user_input_lyrics = flask.request.form["lyrics"]
            user_input = predict.predict(user_artist_choice,user_input_lyrics)
            return flask.render_template('main2.html',flag=False, user_mood = "yes",user_input=user_input)
        else:
            return flask.render_template('main2.html',flag=True,user_mood = user_mood,user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
 