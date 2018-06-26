# Grade Enhance
A pipeline that automatically extracts grading marks and turns them into easy insights about your students' progress

This is a webapp I created as my project during the Insight Data Science Fellowship. It is currently hosted at [http://hjohnsen.site](http://hjohnsen.site)

# Why
After finishing my PhD, I spent two years teaching 3rd, 5th, and 6th grade math and science. Something that frustrated me was that I spent a lot of time grading question-by-question, but ultimately recorded overall scores. Just looking at trends in overall scores, it's impossible to know what topics need to be retaught, but keeping track of which topics each student missed by hand would be prohibitively tedious. As a compromise, I would often scan my student's graded assessments so that I could go back and double-check what they missed before meeting with a student or their parents. Or, I would compromise and create a multiple-choice assessment, even when it might not have been the most informative, because it made it easier to see how my class was doing.

Teachers are good at working with and understanding students and their challenges. Computers are good at keeping track of data. I wanted to create a way that teachers could spend more time doing what they do best while avoiding the tedium of data entry and analysis.

# How to use
A teacher adds a target symbol (see [target.png](https://github.com/hjohnsen/grade-enhance/blob/master/target.png)) next to each question on their assignment. This could be added digitally before printing, or pasted on any pre-existing assignment before making copies. The teacher should grade by adding an x inside the target for missed questions and not writing anything in the target next to correct questions, scan the stack of assignments, and upload it to the Flask app. 

The app will analyze the grading marks and create plots that easily show which questions and topics were the most challenging, which students struggled the most, and trends over time for overall accuracy and topic-specific accuracy. It also plots by topic and by student. These analyses can prevent students from being overlooked and help teachers prioritize which topics or questions to review in class.

# Dependencies
## This app is created in Python 3.6.5 and imports the following modules:
* Flask==1.0.2
* matplotlib==2.2.2
* numpy==1.14.3
* pandas==0.23.0
* pdf2image==0.1.7
* Pillow==5.1.0
* scikit-learn==0.19.1
* Werkzeug==0.14.1

## and the following built-in modules:
* base64
* glob
* io
* os

# Limitations
This currently works with one-page assignments only. It does not extract student names and uses the page order to assign student numbers. It's recommended to always sort in gradebook order or alphabetical order before scanning so that the order is consistent from assignment to assignment. It does not accomodate changing numbers of tests (such as when one or more students were absent).
Grading mark classification highly accurate, but it is not 100%. Therefore, the output from this app is best used as a guide for the teacher to review, but should not be entered directly into a gradebook. The accuracy might be sensitive to scanner settings, so if it is giving strange results, you could try using different scanner settings.
