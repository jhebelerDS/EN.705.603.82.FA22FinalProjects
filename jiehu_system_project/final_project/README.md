## Interpreting Black Box Models
### Description

This project examines the breast cancer Wisconsin original dataset available from UCI Machine Learning Repository (https://archive-beta.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), and uses various techniques to explain the predictions made by popular machine learning models. Both white box and black box models are used for comparison, and their results are analayzed.

### Prerequisites
In order to run this container you will need to have docker installed:

* Windows
* OS X
* Linus

### Project Organization
|- data - name of file folder containing the input file |
    | - breast-cancer-wisconsin.data - name of input file used for analysis 
|- comparisons.png - name of image file containing results 
|- Dockerfile - a text document containing all of the commands to build a docker image 
|- explain_model.py - name of Python Script containing the different modules to explain black box models 
|- hu_final_project.ipynb - Jupyter Notebook containing explanation and analysis of the project 
|- main.py - name of Python Script that contains the entry point to the program 
|- model.py - name of Python Script containing modules for different machine learning models 
|- README.md - name of README file that introduces and explains the project
|- requirements.txt - name of text file that contains list of packages needed to run the docker application

### Steps to Execution

1.  download docker image
2.  navigate to working directory
3.  docker run -it -v c:\Users\crc70\final_project\output:/output jiehu2022/final_project:10.0 
    (replace the directory before the colon with your working directory)
4.  graphics are saved to the output folder

