import os

PROJECT_NAME = "app"
PROJECT_PATH = os.getcwd()

while os.path.basename(os.getcwd()) != PROJECT_NAME:
    os.chdir("..")
    PROJECT_PATH = os.getcwd()

DATA_PATH = os.path.join(PROJECT_PATH, "static", "data")
OUTPUT_PATH = os.path.join(DATA_PATH, "output")

