# Semi-supervised Consistent Labeling of Short Text

**Unlabeled data**
- Exists large repositories of unlabeled free text such as customer reviews, call center transcripts, scientific article abstracts, social media comments, tweets etc. 
- Most text-based models are supervised, hence, such textual data mostly unused

**Manual labeling**
- Resort to manual labeling of a small subset (~10%) of data for supervised learning
- Manual labeling is: 
    - **Tedious** 
        - Nobody really wants to do it and even if someone is forced to do it, labeling quality generally deteriorates over time
        - Some documents requires labeling by domain experts, who will usually not have the luxury of time to label the documents one by one
        - *Context drift*: new labels would emerge over time as the business evovles, not sustainable to repeat manual labeling everytime
    - **Highly subjective**
        - Different reviewers can label the same text differently especially for multi-label datasets (one piece of text can belong to multiple categories)
        - The same reviewer can even label the same text differently at different times
        - Manually labeled dataset are **highly inconsistent** and there are reservations about using these inconsistent manual labels as ground truth

**Short Text**
- Documments with a small number of words (<100 words after text preprocessing) are common
- Unsupervised methods such as topic modeling does not perform well in the case of short text due to highly sparse document-term matrix


## Solution

#### Prerequisites:

1. Install dependencies using pip:

<pre><code>pip install -r requirements.txt</code></pre>

2. Download movies21500.csv from https://drive.google.com/file/d/1U_Y3z1mecNtuiabUHA_TpryExVNtSbmH/view?usp=sharing and put it in this repository folder

#### Solution Details and Scripts

<p> Detailed description of the project provided in auto_label.ipynb
	- Notebook outputs keywords (prelim_keywords.csv) from preliminary topic modeling
	- Manually mix and match keywords to create dictionary (dict.csv) with desired categories, with a list of keywords for each category 
	- Notebook takes in dict.csv to proceed with the semi-supervised labeling </p>

## Built With

Code tested with python 3.5.5 running on Azure Data Science Virtual Machine (Ubuntu 16.04)

## Author

<p>Lin Laiyi, Senior AI Apprentice at AI Singapore, NUS MSBA 2017/2018</p>
<p>LinkedIn: https://www.linkedin.com/in/laiyilin/</p>
<p>Portfolio of selected analytics project: https://drive.google.com/file/d/1fVntFEvj6us_6ERzRmbU85EOeZymFxEm/view</p>