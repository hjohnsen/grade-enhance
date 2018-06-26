from flask import Flask
app = Flask(__name__)
from project import views

# upload_folder = "project/temp/"
# allowed_extensions = ["pdf"]
#
#
# app.config['upload_folder'] = upload_folder
