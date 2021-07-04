from flask import Flask, jsonify
"""import csv
with open("movies.csv") as f:
    reader = csv.reader(f)
    data = list(reader)
    allmovies = data[1:]"""
import pandas as pd
data = pd.read_csv("movies.csv")

allmovies = []
likedmovies = []

for i in data:
    allmovies.append(i)  

app = Flask(__name__)

@app.route('/get-movie')
def get_movie():
    return  jsonify({
        "data": allmovies[1:],
        "status": "success"
    })

@app.route('/liked-movie', methods=['POST'])
def liked():
    movie = allmovies[0]
    allmovie = allmovies[1:]
    likedmovies.append(movie)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)