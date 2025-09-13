from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/experiment")
def experiment():
    return render_template("experiment.html")

@app.route("/effects")
def effects():
    return render_template("effects.html")

@app.route("/rl")
def rl():
    return render_template("rl.html")

if __name__ == "__main__":
    app.run(debug=True)
