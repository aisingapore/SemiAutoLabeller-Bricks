# Project - Auto Labeller

## Introduction

### Background
It can be useful for businesses to label text data for identification, sorting or strategic purposes. Traditionally, businesses employ word matching (building a label dictionary from scratch) or manual labour to put labels on their existing storage. However, this tends to be resource heavy and can be cumbersome to implement.

This project aims to ease this process and make labelling easy. With this \[semi\] auto labelling tool, users simply have to pick from a list of recommended words to form their label dictionary and allow the model to form an enriched dictionary. The model will then utilise the enriched dictionary to label the input text dataset.


### Possible applications
| Data  |  Use Case |
|---|---|
| Email messages from suppliers or customers  | - Better archive and store email messages on local file system </br> - Group suppliers or customers to better understand collaboration partners  |
|  Service tickets for customer complaints  |  - Group customer complaints to identify problematic areas </br> -  Group service approaches to identify the best service approaches  |
|  Customer feedback for products or services  |  - Identify potential new product categories  </br> -  Group feedbacks with labels to identify performance of each product label  |


## Getting Started
### Environment Setup
You will require the following system set up.
1. Install python3 [here](https://realpython.com/installing-python/)
2. Install pip3 for your [windows](https://www.liquidweb.com/kb/install-pip-windows/) or [linux](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
3. Install python virtualenv [here](https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-virtualenv-with-Python-3)
4. Install git [here](https://git-scm.com/downloads)


### Step-by-step setup
1. Clone project to the local and cd into project
```
git clone ***
```
2. Create a python virtual environment within project folder
```
virtualenv -p python3 env
```
3. Activate your virtual environment
```
# Linux
source env/bin/activate

# Windows
env\Scripts\activate
```
4. Install python dependencies
```
pip3 install -r requirements.txt

# or
pip install -r requirements.txt
```

5. Run Jupyter Notebook
```
jupyter notebook
```

6. Run the following code within Jupyter to install nltk
```
import nltk   
nltk.download('all')
```

### Notebook Instruction
1. Walk through the demo in bricks_demo_auto_label.ipynb to gain an intuition of the steps required to operate this auto labelling tool.
2. Walk through the sample notebook in bricks_auto_label.ipynb. This notebook allows you to experiment with the labelling tool and evaluate its usefulness for your company.

## Technical Documentation

### Folder Structure
* bricks_demo_auto_label.ipynb - demo code using the original example for users to get an intuition for the applications of this auto labelling tool.
* bricks_auto_label.ipynb - base code to allow users to play with and experiment with the auto labeller

### Labels Dictionary
It is important to identify your desired *keys* and *labels*
* Manually mix and match keywords to create the dictionary `labels.csv` with desired categories, with a list of keywords for each category 
* Notebook takes in `data/labels.csv` to proceed with the semi-supervised labeling


### How to use this Model
The primary function of this model takes in an input (news.csv) and labels it using the labels.csv. It outputs labelled.csv and if ground truth is available, the score.csv.
* inputs:
    * news.csv - dataset containing string to be labelled
        * can contain as many row as needed (recommended less than 10k rows)
        * contains text to be labelled
    * labels.csv - labels for different identified classification (e.g. finance, sports, politics)
        * contains number of columns corresponding to categories
        * contains keys in each column relating to each category
* outputs:
    * labelled.csv - labelled dataset containing labels for the specified input labels from labels.csv. 
        * Contains the same number of rows as news.csv
    * score.csv - model performance for the labels. Only available if you have ground truth for the input.

### Code Tested
Code tested with python 3.5.5 running on Azure Data Science Virtual Machine (Ubuntu 16.04)


## Credits

### Author
<p>Lin Laiyi, Senior AI Apprentice at AI Singapore, NUS MSBA 2017/2018</p>
<p>LinkedIn: https://www.linkedin.com/in/laiyilin/</p>
<p>Portfolio of selected analytics project: https://drive.google.com/file/d/1fVntFEvj6us_6ERzRmbU85EOeZymFxEm/view</p>
<p> Edited by Jway Jin Jun on Aug 2019, AI Engineer at AI Singapore. </p>
<p> Find the original presentation slide [here](https://docs.google.com/presentation/u/1/d/1hQED4ZZqzcwgq6-jgtw3MbRWRPN6CRTOs_zbVQQu_YU/edit#slide=id.p) </p>

### Additional Notes

<p> Project is editted for the purpose of the Bricks project to demonstrate and enable AI Technologies </p>
<p> Find the original project [here](https://github.com/lylin17/auto_label) </p>


### Software License


### Dataset sources
* news - [github](https://github.com/robsalgado/personal_data_science_projects/tree/master/mulitclass_text_class), [medium](https://medium.com/@robert.salgado/multiclass-text-classification-from-start-to-finish-f616a8642538)
* toxic comments - [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
*