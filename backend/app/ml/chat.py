# from jim_ml import chat_imput
# def chat(input):
#     print(input)
#     return chat_input(input)

import os
from json import load

base_path = os.path.abspath(os.getcwd())
current_dir =  base_path + '/app/ml/'
filename = current_dir + 'jim_ml.ipynb'

with open(filename) as fp:
    nb = load(fp)

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
    
