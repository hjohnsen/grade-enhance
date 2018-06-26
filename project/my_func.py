from pdf2image import convert_from_path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

def load_and_save_pdf(input_file, output_prefix, dpi=72):
    pages = convert_from_path(input_file+".pdf", dpi=dpi)
    if len(pages)==0:
        print("removing .pdf")
        pages = convert_from_path(input_file, dpi=dpi)
    if len(pages)==0:
        print("failed to read pdf")
    for num, page in enumerate(pages):
        page.save(f'{output_prefix}{num}.png', 'PNG')
        print("saved images")

def corr_funct(patch1, patch2):
    """calculates similarity between two patches"""
    std1 = np.std(patch1)
    std2 = np.std(patch2)
    m1 = np.mean(patch1)
    m2 = np.mean(patch2)
    avg=np.mean((patch1-m1)*(patch2-m2))
    return avg/(std1*std2+0.001)

def get_corr(img, grey_filter, boxsize, offset):
    """takes in an image vector, a filter, and returns correlations"""
    i = 0
    j = 0

    nx = (img.shape[1]-boxsize)/offset
    ny = (img.shape[0]-boxsize)/offset

    corr = np.zeros((int(ny)+1, int(nx)+1))

    for y in range(int(ny)):
        i = y*offset

        for x in range(int(nx)):
            j = x*offset
            #print(x,y)
            corr[y,x] = corr_funct(img[i:i+boxsize, j:j+boxsize], grey_filter)

    # plt.figure(figsize=(20,10))
    # plt.subplot(1,3,1)
    # plt.imshow(corr)
    # #plt.colorbar()
    # plt.subplot(1,3,2)
    # plt.imshow(corr>0.2, cmap="gray_r")
    # plt.subplot(1,3,3)
    # plt.imshow(corr>0.3, cmap="gray_r")

    return corr

def new_point(loc_list, i,j, boxsize):
    new = True
    for (a,b) in loc_list:
        if np.abs(a-i)<boxsize and np.abs(b-j)<boxsize:
            new = False
    return new

def get_ordered(img, list_of_locations, nrows, ncol, direction="down"):
    if ncol == 1:
        return sorted(list_of_locations, key=lambda tup: (tup[0]))
    if nrows == 1:
        return sorted(list_of_locations, key=lambda tup: (tup[1]))
    if direction == "across":
        return sorted(list_of_locations, key=lambda tup: (tup[0]//(img.shape[0]/nrows), tup[1]//(img.shape[1]/ncol)))
    elif direction == "down":
        return sorted(list_of_locations, key=lambda tup: (tup[1]//(img.shape[1]/ncol),tup[0]//(img.shape[0]/nrows)))

def get_loc_im(img, corr, offset, boxsize, offset_of_center, width_of_center, nrows, ncol, direction, thresh=0.2):
    list_of_locations = []
    list_of_images = []

    for i, ar in enumerate((corr>thresh)):
        for j, elem in enumerate(ar):
            if elem:
                origi = i*offset+offset_of_center
                origj = j*offset+offset_of_center
                if new_point(list_of_locations, origi, origj, boxsize):
                    list_of_locations.append((origi, origj))
                    #list_of_images.append(img[origi:origi+width_of_center, origj:origj+width_of_center])
    list_of_locations = get_ordered(img, list_of_locations, nrows, ncol, direction)
    for origi,origj in list_of_locations:
        list_of_images.append(img[origi:origi+width_of_center, origj:origj+width_of_center])

    return list_of_locations, list_of_images


def plot_q_accuracies(student_scores):
    question_accuracies = np.mean(student_scores, axis=0)*100
    nquestions = question_accuracies.shape[0]
    plt.bar(np.arange(nquestions)+1, question_accuracies)
    plt.xlabel("Question number")
    plt.ylabel("% Class correct")
    plt.xticks(np.arange(1, nquestions+1))
    plt.xlim([0, nquestions+1])
    plt.ylim([-1,101])
    plt.hlines(np.mean(question_accuracies), 1, nquestions)
    plt.title("Class accuracy on each question")
    #plt.show()


def get_below_avg_qs(student_scores):
    question_accuracies = np.mean(student_scores, axis=0)*100
    per = np.percentile(question_accuracies, [25,50,75])
    IQR_thresh = per[1]-(per[2]-per[0])/2
    worst_qs = np.argwhere(question_accuracies<IQR_thresh)
    print("worst to best questions:", np.argsort(question_accuracies)+1)
    print("far below avg questions:", worst_qs.reshape(-1)+1)
    return worst_qs.reshape(-1)+1

def plot_student_accuracies(student_scores, title="Test score for each student"):
    student_accuracies = np.mean(student_scores, axis=1)*100
    nstudents = student_accuracies.shape[0]
    plt.bar(np.arange(nstudents)+1, student_accuracies)
    plt.xlabel("Student number")
    plt.ylabel("Test score")
    plt.xticks(np.arange(1, nstudents+1))
    plt.xlim([0, nstudents+1])
    plt.ylim([-1,101])
    plt.hlines(np.mean(student_accuracies), 1, nstudents)
    plt.title(title)


def get_below_avg_students(student_scores):
    student_accuracies = np.mean(student_scores, axis=1)*100
    avg = np.mean(student_accuracies)
    per = np.percentile(student_accuracies, [25,50,75])
    IQR_thresh = per[1]-(per[2]-per[0])/2
    worst_students = np.argwhere(student_accuracies<IQR_thresh)
    print("worst to best student IDs:", np.argsort(student_accuracies)+1)
    print("far below avg students:", worst_students.reshape(-1)+1)
    return worst_students.reshape(-1)+1

def plot_topic_scores(topics, student_scores, title = "Class-wide"):
    topic_scores = {}
    if len(student_scores.shape)==2:
        for topic in topics.keys():
            topic_scores[topic]=np.mean(student_scores[:,topics[topic]])*100
            #print(topic, topic_scores[topic])
    elif len(student_scores.shape)==1:
        for topic in topics.keys():
            topic_scores[topic]=np.mean(student_scores[topics[topic]])*100
    plt.bar(range(len(topic_scores)), list(topic_scores.values()), align='center')
    plt.xticks(range(len(topic_scores)), list(topic_scores.keys()), rotation=45, ha="right")
    plt.xlabel("Topic")
    plt.ylabel("% accuracy")
    plt.ylim([-1,101])
    plt.title(title+ " accuracy for each topic")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ["pdf"]

def mysavefig():
    png_output = BytesIO()
    plt.savefig(png_output)
    png_output.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(png_output.getvalue()).decode('utf8')
    plt.clf()
    return figdata_png

def plot_topics_over_time(scores_to_date, topics_to_date, how="class", studentid=0):
    # create a unique set of topics that have ever been covered
    topic_set=set()
    for test in topics_to_date.values():
        #print(list(test.keys()))
        topic_set.update(list(test.keys()))
    print(topic_set)

    for topic in topic_set:
        #print(topic)
        tests_for_this_topic = [] #will hold test "names", 1-indexed
        scores_for_this_topic = []
        for test_id in topics_to_date.keys():
            try:
                if how == "class":
                    print(np.mean(scores_to_date[:,topics_to_date[test_id][topic],test_id])*100)
                    scores_for_this_topic.append(np.mean(scores_to_date[:,topics_to_date[test_id][topic],test_id])*100)
                    print(scores_for_this_topic)
                elif how == "student":
                    scores_for_this_topic.append(np.mean(scores_to_date[studentid,topics_to_date[test_id][topic],test_id])*100)
#                     topic_list.append(topic)
                tests_for_this_topic.append(test_id+1)
            except KeyError:
                pass
        plt.plot(tests_for_this_topic, scores_for_this_topic, label = topic, alpha=0.5)
    plt.xticks(range(1,scores_to_date.shape[2]+1))
    plt.ylim([-1,101])
    if how == "class":
        plt.title("Classwide performance")
    elif how == "student":
        plt.title(f"Performance for student {studentid+1}")
    plt.legend()

def plot_student_topics_over_time(scores_to_date, topics_to_date):
    im_list = []
    class_size = scores_to_date.shape[0]
    for student in range(class_size):
        plot_topics_over_time(scores_to_date, topics_to_date, how="student", studentid=student)
        im_list.append(mysavefig())
    return im_list

def plot_scores_to_date(scores_to_date):
    #for row in np.mean(scores_to_date, axis=1):
    #    plt.plot(row)
    plt.plot(np.mean(np.mean(scores_to_date, axis=1), axis=0)*100)
    ntests = np.mean(np.mean(scores_to_date, axis=1), axis=0).shape[0]
    plt.xticks(range(ntests), range(1, ntests+1))
    plt.ylim([-1,101])
    plt.title("Classwide performance over time")

def pad_np(array, goal_width):
    pad_width = goal_width - array.shape[1]
    if len(array.shape) == 2:
        return np.concatenate([array, np.empty([array.shape[0], pad_width])], axis=1)
    else:
        pad_depth = array.shape[2]
        return np.concatenate([array, np.empty([array.shape[0], pad_width, pad_depth])], axis=1)
