from project import app
from project.my_func import *
from flask import render_template, request, redirect, url_for, flash, make_response
from werkzeug.utils import secure_filename

upload_folder = "project/temp/"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from pdf2image import convert_from_path
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

@app.route('/', methods = ["GET", "POST"])
def upload_file():
    if request.method == "POST":
        print(request.form.get("checked"))
        if request.form.get("checked") is None:
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print("saving "+ filename+ " as " + upload_folder+filename)
                file.save(os.path.join(upload_folder, filename))
                dpi = request.form.get("dpi")
                ncol = request.form.get("ncol")
                nrows = request.form.get("nrows")
                direction = request.form.get("direction")
                nquestions = request.form.get("nquestions")
                topics = request.form.get("topics")
                uniqueId = request.form.get("uniqueId")
                return redirect(url_for('uploaded_file', filename=filename, dpi=dpi, ncol=ncol, nrows=nrows, direction=direction, nquestions=nquestions, topics=topics, uniqueId=uniqueId))
        else:
                filename = "greyquiz.pdf"
                dpi = request.form.get("dpi")
                ncol = request.form.get("ncol")
                nrows = request.form.get("nrows")
                direction = request.form.get("direction")
                nquestions = request.form.get("nquestions")
                topics = request.form.get("topics")
                uniqueId = request.form.get("uniqueId")
                return redirect(url_for('uploaded_file', filename=filename, dpi=dpi, ncol=ncol, nrows=nrows, direction=direction, nquestions=nquestions, topics=topics, uniqueId=uniqueId))
        print("all file queries failed")
    else:
        return render_template("input.html")



@app.route('/uploaded', methods = ["GET", "POST"])
def uploaded_file():
    filename = request.args.get("filename")
    dpi = int(request.args.get("dpi"))
    print("dpi:", dpi)
    print("dpitype:", type(dpi))
    nrows = request.args.get("nrows")
    print("nrows", nrows)
    ncol = int(request.args.get("ncol"))
    direction = request.args.get("direction")
    print("dir:", type(direction))
    nquestions = int(request.args.get("nquestions"))
    print("nq:", type(nquestions))
    topic_text = request.args.get("topics")
    print(topic_text)
    uniqueId = request.args.get("uniqueId")
    print(uniqueId)
    if topic_text:
        print("getting topics")
        topic_nestlist = [item.split(":") for item in topic_text.split(";")]
        for item in topic_nestlist:
            item[0] = item[0].strip()
            item[1] = item[1].split(",")
            item[1] = [int(x)-1 for x in item[1]]
        topics = dict(topic_nestlist)
    else:
        print("no topics")
        topics = False

    thresh = 0.3

    filename = filename[:-4]

    if not os.path.exists(upload_folder+filename):
        os.mkdir(upload_folder+filename)
    print(filename)
    load_and_save_pdf(upload_folder+filename, f"{upload_folder}{filename}/{filename}", dpi = dpi)
    print("images saved. Moving on")

    if uniqueId and os.path.exists(f'{upload_folder}{uniqueId}_scores'):
        scores_to_date = joblib.load(f'{upload_folder}{uniqueId}_scores')
        if len(glob.glob(f"{upload_folder}{filename}/{filename}*.png")) != scores_to_date.shape[0]:
            print("new file:", len(glob.glob(f"{upload_folder}{filename}/{filename}*.png")))
            print("old file:", scores_to_date.shape[0])
            return     '''<!doctype html>
                <title>Error</title>
                <h1>Combining assignments with different numbers of students is not currently supported.</h1>
                <h2>Please go back and try again<h2>
                <a href="http://hjohnsen.site">Go back<a>'''



    # The template filter is in the target0.png file.
    # These coordinates form a tight bound
    filt = plt.imread("project/assets/target0.png")[36:86, 12:62]   #filter = targets[28:83, 61:116]
    print("Filter loaded")
    grey_filter =  np.average(filt, weights=[0.299, 0.587, 0.114], axis=2)
    #plt.imshow(grey_filter, cmap="gray")

    boxsize= grey_filter.shape[0]
    offset = int(boxsize/10)
    # empirically determined for this 50x50 filter
    offset_of_center= 18
    width_of_center = 18

    clf = joblib.load('project/assets/classifier.pkl')
    print("classfier loaded")
    student_scores = np.empty([1,nquestions])
    print(f"looking for {upload_folder}{filename}/{filename}*.png")
    for png_file in glob.glob(f"{upload_folder}{filename}/{filename}*.png"):
        print("loading png_file")
        img = np.average(plt.imread(png_file), weights=[0.299, 0.587, 0.114], axis=2)
        corr = get_corr(img, grey_filter, boxsize, offset)
        list_of_locations, list_of_images = get_loc_im(img, corr, offset, boxsize, offset_of_center, width_of_center, nrows, ncol, direction, thresh=thresh)
        #print(list_of_locations)
        X = np.empty([1,324])
        for im in list_of_images:
            #plt.imshow(im)
            #plt.show()
            #print(clf.predict(im.reshape(1,-1)))

            X = np.append(X, im.reshape(1,-1), axis=0)
        X = X[1:, :]
        scores = clf.predict(X)
        print(scores)
        if scores.shape[0] < nquestions:
            print("Error: not all questions have been found!")
        student_scores = np.append(student_scores, scores.reshape(1,-1), axis=0)

    student_scores = student_scores[1:, :]

    if uniqueId and os.path.exists(f'{upload_folder}{uniqueId}_scores'):
        scores_to_date = joblib.load(f'{upload_folder}{uniqueId}_scores')
        if scores_to_date.shape[1] > student_scores.shape[1]:
            student_scores = pad_np(student_scores, scores_to_date.shape[1])
        elif scores_to_date.shape[1] < student_scores.shape[1]:
            scores_to_date = pad_np(scores_to_date, student_scores.shape[1])
        scores_to_date = np.dstack((scores_to_date, student_scores))
        joblib.dump(scores_to_date, f'{upload_folder}{uniqueId}_scores')
        try:
            topics_to_date = joblib.load(f'{upload_folder}{uniqueId}_topics')
            if topics:
                ntest = scores_to_date.shape[2]-1
                topics_to_date[ntest]=topics
                joblib.dump(topics_to_date, f'{upload_folder}{uniqueId}_topics')
        except:
            print("no topics")
            topics_to_date = None
            if topics:
                ntest = scores_to_date.shape[2]-1
                joblib.dump({ntest:topics}, f'{upload_folder}{uniqueId}_topics')

    elif uniqueId:
        joblib.dump(student_scores, f'{upload_folder}{uniqueId}_scores')
        if topics:
            joblib.dump({0:topics}, f'{upload_folder}{uniqueId}_topics')
        scores_to_date = None
        topics_to_date = None
    else:
        print("no uniqueId")
        scores_to_date = None
        topics_to_date = None

    # np.save(f"{upload_folder}{filename}scores", student_scores)
    # filename= f"{upload_folder}{filename}scores"
    # print("saving npfile as ", filename)
    #
    # print("reading npfile as ", filename+".npy")
    # student_scores = np.load(filename+".npy")
    # print(student_scores)

    hardest_qs = get_below_avg_qs(student_scores)
    worst_students = get_below_avg_students(student_scores)

    class_plots= []
    topic_plots= []
    student_plots = []

    plot_q_accuracies(student_scores)
    class_plots.append(mysavefig())

    plot_student_accuracies(student_scores)
    class_plots.append(mysavefig())

    if topics:
        plot_topic_scores(topics, student_scores)
        class_plots.append(mysavefig())

    if scores_to_date is not None:
        plot_scores_to_date(scores_to_date)
        class_plots.append(mysavefig())

    if scores_to_date is not None and topics_to_date is not None:
        plot_topics_over_time(scores_to_date, topics_to_date)
        class_plots.append(mysavefig())

    if topics:
        for topic in topics.keys():
            plot_student_accuracies(student_scores[:,topics[topic]], title="Test scores for "+ topic)
            topic_plots.append(mysavefig())
        for number, student_row in enumerate(student_scores):
            plot_topic_scores(topics, student_row, title=f"Student {str(number+1)}")
            student_plots.append(mysavefig())

    # print(scores_to_date)
    # print(topics_to_date)
    if scores_to_date is not None and topics_to_date is not None:
        student_plots.extend(plot_student_topics_over_time(scores_to_date, topics_to_date))


    dict_of_plots = {"hardest_qs": hardest_qs, "worst_students": worst_students, "class_plots": class_plots, "topic_plots": topic_plots, "student_plots": student_plots}

    return render_template("index.html", dict_of_plots=dict_of_plots) #list_of_plots=list_of_plots)



    #return redirect(url_for('make_plots', filename=filename))

@app.route('/plots/')
def make_plots():
    filename = request.args.get("filename")
