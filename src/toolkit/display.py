# implementation from https://en.wikipedia.org/wiki/Model–view–controller
from ipysheet import sheet, cell, column, cell_range
from IPython.display import display, Markdown, clear_output

import ipysheet
import ipywidgets as widgets
import pandas as pd

# Imported functions to run model
from .autolabel import Preprocessor, AutoLabeller, check_labels
from .autolabel import recommend_words

from sklearn.naive_bayes import MultinomialNB

out = widgets.Output() # this needs to be outside of the class

class MLModel():
    def __init__(self):
        self.stopwords_path = "data/stopwords.csv"
        return
    
    def run(self, data, label):
        corpus = data['content']
        
        # Initialise Labels
        label = check_labels(data, label)

        # Text Preprocessing
        preprocessor = Preprocessor()
        preprocessed_corpus = preprocessor.corpus_preprocess(corpus=corpus, stopwords_path=self.stopwords_path)
        preprocessor.corpus_replace_bigrams(corpus=preprocessed_corpus, min_df=50, max_df=500)
        data['content'] = preprocessor.corpus_replace_bigrams(corpus=preprocessed_corpus, min_df=50, max_df=500)

        # Enrich Labels
        autoLabeller = AutoLabeller(label, corpus, data)
        enriched_label = autoLabeller.train()

        # Predict results
        mnb = MultinomialNB()
        ypred = autoLabeller.apply(mnb, "content")
        labelled_data = data[["content"]].join(ypred)
        
        return enriched_label, labelled_data

class Model():
    """ Class to solely handle the data, this can include widgets
    """
    def __init__(self):
        return
    
    def create_header(self, header: str, description: str = ""):
        """ creates a HTML widget to display text
        """
        header = widgets.HTML(
            value="<h1>{}</h1><font size=3>{}</font>".format(header, description),
            placeholder="{}".format(header),
            description="",
        )
        return header
    
    def create_sheet(self, label_path: str):
        """ creates ipysheet from file path to csv file
        """
        df = pd.read_csv(label_path)
        sheet = ipysheet.from_dataframe(df)
        return sheet
    
    def create_sheets(self, label_paths: list):
        """ creates dictionary of ipysheet using labelspath and titles
        """
        sheets = {}
        for i, label_path in enumerate(label_paths):
            sheet = self.create_sheet(label_path)
            sheets[self.titles[i]] = sheet
        return sheets
    
    def create_dfs(self, data_paths: list):
        """ creates dataframes from list of paths to csv files
        """
        dfs = {}
        for i, data_path in enumerate(data_paths):
            df = pd.read_csv(data_path)
            dfs[self.titles[i]] = df
        return dfs
    
    def initialise(self, titles: list, data_paths: list, reco_paths: list, label_paths: list):
        self.titles = titles
        self.datas = self.create_dfs(data_paths)
        self.recommendations = self.create_dfs(reco_paths)
        self.sheets = self.create_sheets(label_paths)
        return

    
class View():
    """ Class solely responsible for the UI display of the project
    """
    def __init__(self):
        return
    
    def display_output(*items):
        """
        Displays output in sequence that items where fed in
        """
        with out:
            out.clear_output()
            for item in items[1:]:
                display(item)
        return    
    
    def first_render(self, titles: list, toolHeader: str, data: pd.DataFrame, recommendationHeader: str,
                     recommendation: pd.DataFrame, labelHeader: str, sheet: ipysheet.sheet):
        """ creates the layout of the UI
        """
        
        # dropdown menu
        menu = widgets.Dropdown(options=titles,
                                value=titles[0],
                                description='Input Data:')
        
        # normal button
        button = widgets.Button(description='Run Model',
                                button_style='info')
        

        hbox = widgets.HBox([menu, button])
        vbox = widgets.VBox([toolHeader, hbox, out])
        
        self.menu = menu
        self.runButton = button
        self.display_output(data, recommendationHeader, recommendation, labelHeader, sheet)
        return vbox

    
class Controller():
    """ Class solely responsible for the triggering of updates for UI of the project
    """
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.mlmodel = MLModel()
        return
      
    
    def button_clicked(self, _):
        # "linking function with output"
        with out:
            self.view.display_output(
                self.model.datas[self.view.menu.value],
                self.model.recommendationHeader, 
                self.model.recommendations[self.view.menu.value],
                self.model.labelHeader, 
                ipysheet.to_dataframe(self.model.sheets[self.view.menu.value])
            )
        # runs the model
        enriched_label, labelled_data = self.mlmodel.run(self.model.datas[self.view.menu.value][['content']].copy(deep=True),                       
                                         ipysheet.to_dataframe(self.model.sheets[self.view.menu.value]))
        
        with out:
            self.view.display_output(
                self.model.datas[self.view.menu.value],
                self.model.recommendationHeader, 
                self.model.recommendations[self.view.menu.value],
                self.model.labelHeader, 
                self.model.sheets[self.view.menu.value],
                self.model.enrichedHeader,
                enriched_label, 
                self.model.labelledHeader,
                labelled_data
            )
        return
    
    def drop_down_updated(self, _):
        """ Update data when drop_down_menu_updated
        """
        self.view.display_output(self.model.datas[self.view.menu.value],
                                 self.model.recommendationHeader, 
                                 self.model.recommendations[self.view.menu.value],
                                 self.model.labelHeader, 
                                 self.model.sheets[self.view.menu.value])
        return
    
    def wire_components(self):
        """ Gives components their effects when clicked
        """
        self.view.runButton.on_click(self.button_clicked)
        self.view.menu.observe(self.drop_down_updated)
        
        return
    
    def render(self):
        """ calls first rendering of the view
        """
        firstItem = self.model.titles[0]
        view = self.view.first_render(self.model.titles,
                                      self.model.toolHeader,
                                      self.model.datas[firstItem],
                                      self.model.recommendationHeader, 
                                      self.model.recommendations[firstItem],
                                      self.model.labelHeader, 
                                      self.model.sheets[firstItem])
        self.wire_components()
        
        return view


def run_model():
    titles = ['News', 'Movies']
    label_paths = ["data/news500_labels.csv", "data/movies500_labels.csv"]
    data_paths = ["data/news500.csv", "data/movies500.csv"]
    reco_paths = ["data/news500_matrix10.csv", "data/movies500_matrix10.csv"]

    model = Model()

    model.initialise(titles, data_paths, reco_paths, label_paths)
    model.toolHeader = model.create_header("Semi Automatic Labeller (Corgie) - Demo",
                                        """
                                        Tool designed to label short text, we will use a subset of the movies dataset and news dataset to demonstrate the potential of the tool.
                                        Do note that you just need the 'content' column to utilise this tool.
                                        """)
    model.labelHeader = model.create_header("Label Dictionary",
                                            "This is the list of labels for the document, you may modify this label dictionary before running the model.")
    model.enrichedHeader = model.create_header("Enriched Dictionary", 
                                            "This is an enriched version of the input label dictionary.")
    model.recommendationHeader = model.create_header("List of recommended words",
                                                    "In each row you will see words relating to a topic or theme.")
    model.labelledHeader = model.create_header("Predictions", 
                                            "This are the predictions of our tool.")
    view = View()
    controller = Controller(model, view)

    comp = controller.render()
    
    return comp