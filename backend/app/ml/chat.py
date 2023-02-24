import os
from json import load

# This file is dedicated to exposing a function fastapi can use to call the 'chat_input' function in the jim_ml.ipynb.
# Python files apparently cant just import ipynb files, even though they both house python code, so this file is meant to bridge that gap.

# The following code will open the 'jim_ml.ipynb file as a json, since ipynb's raw data is a json file.
base_path = os.path.abspath(os.getcwd())
current_dir =  base_path + '/app/ml/'
filename = current_dir + 'jim_ml.ipynb'

with open(filename) as fp:
    nb = load(fp)

# The 'cells' key in the ipynb json contains the python code in the form of an array of cells
# Here we are running all the cells in the current scope so that once the chat function below is ran,
# it can call variables and functions written in the jim_ml.ipynb file.
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        exec(source, globals(), locals())

async def chat(input):
    try:
        # The Jupyter notebook will need a 'chat_input(input)' function
        # that takes in an input string and
        # responds with the chatbot's response as a string
        return chat_input(input)
    except Exception as e:
        error = str(e)
        print(error)
        return {"error": "Something went wrong calling the 'chat_input' from the jupyter notebook"}
    
