from flask import Flask, render_template, request
import functions, re

if __name__ == "__main__":

    app = Flask(__name__)

    @app.route("/")
    def site():
        return render_template("index.html")

    @app.route("/",methods=["POST"])
    def my_post():

        query = request.form.get("query")
        knn = request.form.get("KNN")
        dist = request.form.get("distances")
        distances, result = functions.nearest_k(query, int(knn), dist)
        final = []
        count = 0
        for res in result:
            # print(r)
            text = open('data/{}.txt'.format(res), 'r')
            content = text.read()
            content = re.sub("\n",'<br>', content)
            str = dict()
            str["id"] = res
            str["distance"] = round(distances[count],2)
            str["text"] = content.split("\n ")
            str["link"] = "Link: data/{}.txt".format(res)
            final.append(str)
            text.close()
            count += 1
        return render_template("index.html", result=final)


    app.run()

