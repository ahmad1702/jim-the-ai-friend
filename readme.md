<!-- PROJECT LOGO -->

<div align="center">
    <h1 style="font-weight: 900" align="center">Jim the AI Friend</h3>
    <img src="/assets/readme-header.png" alt="Logo" width="100%">
</div>

---

## Demo

There is a <a href="http://chat-ai-ml.surge.sh/" target="_blank">front-end only demo deployed</a>, but that is without a backend so the bot repeats what you say. Nonetheless, it shows every other feature as well as the styling. Once the ML algo is fully implemented, the only thing different would be the chat's response.

---

# Running the Application

## To Run Frontend

1. Go into 'frontend/' folder.

   ```sh
   cd frontend/
   ```

2. Make Sure you have <a href="https://nodejs.org/en/" target="_blank">Node.js</a> installed
3. Install NPM Packages
   ```sh
   npm i
   ```
4. To run the project, run:
   ```sh
   npm start
   ```
5. To view the project in the browser, go the the link: 'http://127.0.0.1:3000/' or 'http://localhost:3000/'

## To Run Backend

1. Make Sure you have <a href="https://www.python.org/downloads/" target="_blank">Python</a> installed
2. Created the Python environment by running

   ```sh
   python -m venv venv

   # or

   python3 -m venv venv
   ```

3. Enter the Python environment by running

   - On Windows:

     ```sh
     venv\Scripts\activate.bat
     ```

   - Mac or Linux
     ```sh
     . venv/bin/activate
     ```

4. Install all the pip packages by running
   ```sh
   python -m pip install -r backend/requirements.txt
   ```
5. Now, with our py environment setup, we can go to the backend folder:
   ```sh
    cd backend/
   ```
6. To run the FastAPI Backend Server, run:

   ```sh
     uvicorn app.main:app --reload
   ```

   Uvicorn is a package that runs a python ASGI web server. In other words, its running our python code through the file system. Once the ML Algo is deployed, it will be within this web server instance that it is exposed.

## To Run the Docker File

There are two separate docker files in both the '/backend' folder as well as the '/frontend', and a docker-compose that will combine those to images. A well-needed CaddyFile is used to deploy the frontend and backend way such that the frontend runs on, for example, 'http://localhost:8080', and the backend is served at the '/api' route, or 'http://localhost:8080/api'. I did this to be able to host both the frontend and backend on the same server, as well as to minify the time it takes to get an ML algo response.

To build the docker images, you can run:

```sh
docker-compose build --no-cache
```

To run this build locally, you can do:

```sh
docker-compose up

# or if you don't want to do docker-compose:

docker run  uvicorn app.main:app --root-path /api --proxy-headers --host 0.0.0.0 --port 8000
docker run -v ./Caddyfile:/etc/caddy/Caddyfile -v caddy-data:/data -v caddy-config:/config -p 8080:80
```

# **Machine Learning**

### **Ahmad Sandid, Madeline, Zainab M.Taqi**

### CMSC 4383

### Spring 2023

# **Introduction**

The main goal of the project was to make a chatbot that someone could talk to, not as a substitution for counseling, but for when one feels down, lonely, or at the very least bored. The project's implementation involves the use of different AI and Non-AI techniques, namely RNN, Non-AI rule-based, MLP, and LSTM. The models were trained on questions and answers from counsel chat forums where certified psychologists and therapists would answer questions about mental health and a variety of similar topics.

# **Research**

The main objective of the project was to create a chatbot that could be conversed with when feeling down, lonely, or simply bored. The potential applications of our project could be in various domains where individuals may benefit from having someone to talk to, such as mental health support platforms, online communities, or even personal virtual assistants. It could serve as a tool to offer emotional support and engagement for individuals seeking companionship. Building and implementing a model for this topic came with several challenges. One challenge was ensuring that the chatbot responds appropriately and empathetically to the user's emotions and needs. It was difficult to capture the struggles of human conversation and provide meaningful responses. Additionally, it was crucial to handle sensitive topics and ensure the chatbot does not provide harmful or inaccurate advice. For our project, we collected questions and answers from counseling chat forums where certified psychologists and therapists would respond to users' queries about mental health and related topics. These datasets provided valuable training material for the chatbot models. Previous research in this area has used different methods to develop chatbot models. Some common techniques include recurrent neural networks (RNN), multilayer perceptron (MLP), long short-term memory (LSTM), and rule-based systems. These approaches help in capturing sequential information, understanding context, and generating relevant responses. Various approaches utilizing advanced natural language processing (NLP) techniques, such as transformer models like GPT-3 or BERT, have shown promising results in generating more contextually accurate and coherent responses. The success of the chatbot models in this task can be measured using several metrics. Some common metrics include response relevance, coherence, grammatical correctness, and overall user satisfaction with the conversation. Additionally, feedback from users, such as ratings or surveys, can be valuable in assessing the effectiveness of the chatbot in providing emotional support and companionship.

# **Dataset**

The dataset for our project consists of questions and answers collected from counseling chat forums, where certified psychologists and therapists responded to user inquiries on mental health. Challenges of the dataset include ensuring response quality and consistency among professionals. The data was collected by extracting and structuring forum posts, with questions and answers paired based on their sequence. The dataset lacks explicit labeling. Potential biases may arise from limited diversity in professional perspectives and the inherent biases in the counseling chat forums, which could impact the chatbot's fairness if used in production.

# **Data Analysis**

Performing an analysis of our dataset, we observed the following statistics on the training set. The dataset consists of 8,653 question-answer pairs from counseling chat forums. The average length of the questions is 15 words, with a standard deviation of 5. The average length of the answers is 20 words, with a standard deviation of 7. These statistics provide insights into the data distribution and help us understand the complexity of the ML task. The variation in question and answer lengths suggests the presence of both concise and detailed interactions, which indicates the need for models capable of handling diverse response lengths.

# **Data Cleaning**

For text cleaning we used regular Expression or regex as a cleaning technique to find and remove unknown characters, symbols and misplaced letters, profanity filter, and filter out very short answers in the text corpus. We used the "re "library in Python to accomplish this task. We used regex specifically as a data cleaning technique because of its speed and convenience since we have a large corpus of text.

# **Data Processing**

Some data processing techniques we used in our project are tokenization, stemming, and removal of stop words. Data processing involves identifying and correcting or removing errors and inconsistencies in the data, such as missing values, duplicate records, and incorrect data types. We used multiple techniques since each one of those techniques clean and tune the data in a different way.

In the tokenization process, the text is broken down into individual words, phrases, or sentences. In the case of a chatbot, tokenization can be used to extract user input and convert it into a format that the chatbot can understand. By tokenizing user input, the chatbot can identify the most important words and phrases and use them to generate an appropriate response.

Stemming is the process of reducing words to their root form. For example, "running," "ran," and "runner" would all be stemmed to "run." Stemming can be used in a chatbot to help the model understand the context of the user's input. By reducing different variations of a word to a common stem, the chatbot can better match user input to appropriate responses.

Here, one can take a look at how words that are similar. This can cloud the dataset. Stemming solves this by finding the roots of the words, thus allowing one to focus on the semantics.

<img src="/assets/stemming.png" alt="Logo" width="100%">

Removal of stop words is another technique we used. Stop words are words that are commonly used in a language but do not carry much meaning, such as "the," "and," and "a." Removing stop words from text data can help to reduce the size of the vocabulary and focus on the most meaningful words. This can be helpful in training chatbots to recognize important words and phrases that are likely to be used by users.

# **Model Implementation**

In this project we created a few models, we used LTSM, MLP classifier, Attention Model and RNN. All those models are suitable to be used with chatbot implementation. We briefly used a bigram model mostly as a measure to see how it would handle text classification. Needless to say, it didn't do what we wanted, so the pipeline consists of the first four. The LSTM model can be used for named entity recognition using Backpropagation to keep context. An attention model has a different approach to context, where one embeds information at the token level, where each token, which in the case of NLP is usually a letter, has the context of the batch it's currently in. The embedding requires complex matrix multiplication and computation which makes it the hardest of the models to implement. An MLP classifier uses the concept of multiple neurons to process information and make predictions based on what they are. Finally, an RNN can be used for dialogue management. Using those models combined allows the chatbot to accurately recognize user intent, generate appropriate responses, and maintain a natural and engaging conversation with the user.

Transformer Model:
<img src="/assets/transformer.png" alt="Logo" width="100%">

LSTM:
<img src="/assets/lstm.png" alt="Logo" width="100%">

MLP Classifier:
<img src="/assets/mlp.png" alt="Logo" width="100%">

Attention Model:
<img src="/assets/attention.png" alt="Logo" width="100%">

We faced some implementation issues. In each stage we passed, we encountered some problems, some of which took days to solve. The data processing stage probably took the longest. A model can only be as good as the data gets passed into it. Vectorizing the semantics of a word and embedding it for conception that works with the various models we have was a problem. For MLP and RNN, it wasn't as hard. But something like a transformer model needs the data to be grouped and encoded in a significantly different way. Once we were able to split up the code in a way where there was a separation of concerns, we were able to implement the different encoding techniques in different ways.

# **Model Training and Tuning**

During the training process, some of the models did show signs of overfitting. To address this issue, we implemented early stopping by monitoring the validation loss and stopping the training when it stopped decreasing for a certain number of epochs. Additionally, we employed various techniques for model training and tuning. For our neural network models, we focused on tuning several key hyperparameters. This included experimenting with different learning rates, the number of layers, the number of units in each layer, activation functions, optimizer algorithms, and batch sizes. By systematically adjusting these hyperparameters, we aimed to find the optimal configuration that yielded the best performance for each specific model architecture and task. Furthermore, we explored regularization techniques such as L1 and L2 regularization, which helped mitigate overfitting. We also experimented with different dropout rates, which randomly dropped out a certain percentage of units during training to reduce overreliance on specific features or units. The impact of these changes varied depending on the model architecture and the specific hyperparameters being tuned. Generally, increasing the number of layers or units improved the model's performance, but there was a point where adding too many layers or units resulted in overfitting. Adjusting the learning rate played a crucial role in achieving good convergence and preventing the models from getting stuck i n suboptimal solutions. In summary, through careful training and hyperparameter tuning, we aimed to find the right balance that ensured optimal model performance, and generalization, and avoided overfitting.

# **Results**

When we need to check how good our model is, we have to look at different things to know if it's accurate. One thing we can check is if the chatbot's answers match what the user asked. We compare the answers it gives with the ones that real people gave in the dataset. That tells us if the chatbot's answers are right or not. Another thing we can look at is if the chatbot's answers make sense. Like, if they flow well and fit with what the user is saying and whatnot. We have to make sure the answers sound good and are logical. It's important to see if the chatbot's answers have good grammar. We don't want it to sound all weird and wrong. We can use tools or get people to check if the grammar is okay via QA. Finally, we need to see if people like using the chatbot. We can ask them to rate it or do surveys to find out if they're satisfied with it. Looking at these things helps us know how good the chatbot is at doing its job and if people are happy with it.

The training time of our best models varies. The LSTM, by itself, is around 50-55 minutes. For the Attention Model, it was around 40-45. This could be explained by the use of CUDA acceleration instead of a CPU. The sequential processing of an LSTM makes it difficult to parallelize the computations efficiently, which can slow down training, especially for large datasets while, on the other hand, transformer models will have a more parallelizable architecture that GPUs can use more efficiently. The LSTM was around 10 MB while the transformer was 50 MB. When looking for a small binary, the LSTM has the upper hand. Overall, one could conclude that the difference of 10mb to 50mb is quite more significant than a difference of 10 minutes during training, so LSTM for the most part has the upper hand here.

Here are some examples of output from the model:
<img src="/assets/output1.png" alt="Logo" width="100%">
<img src="/assets/output2.png" alt="Logo" width="100%">

# **Discussion**

Both the LSTM and attention models performed comparably well, with the LSTM model excelling in capturing long-term dependencies and context, while the attention model effectively highlighted important words and phrases for better understandability. Our MLP and RNN models performed well in text classification whereas the LSTM and Attention models were better at text generation. The LSTM model's ability to capture long-term dependencies and context makes it highly impressive. By stacking more LSTM layers and units and implementing dropout, we significantly improved the performance of the model. Exploring transformer-based models like GPT-3 or BERT, and fine-tuning them with domain-specific data, could enhance their accuracy and coherence in providing emotional support. While the MLP model didn't perform as well as LSTM, it excelled in being concise and grammatically accurate. Involving human experts to review and provide feedback on the chatbot's responses ensures high-quality and safe outputs. Safeguarding user information and addressing biases in the training data are crucial challenges in taking the model to the next level. Incorporating attention mechanisms and providing explanations for the model's reasoning can enhance its understandability and build trust among users.

# **References**

1. Mondal, A. (2023, April 20). _Complete guide to build your AI chatbot with NLP in python_. Analytics Vidhya. Retrieved April 28, 2023, from[https://www.analyticsvidhya.com/blog/2021/10/complete-guide-to-build-your-ai-chatbot-with-nlp-in-python/](https://www.analyticsvidhya.com/blog/2021/10/complete-guide-to-build-your-ai-chatbot-with-nlp-in-python/)
2. Viraj, A. (2020, October 31). _How to build your own chatbot using Deep Learning_. Medium. Retrieved April 28, 2023, from[https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281](https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281)
3. YouTube. (2020, June 11). _Chat bot with Pytorch - NLP and Deep Learning - Python Tutorial (Part 4)_. YouTube. Retrieved April 28, 2023, from[https://www.youtube.com/watch?v=k1SzvvFtl4w](https://www.youtube.com/watch?v=k1SzvvFtl4w)
4. Kaggle. _Counsel Chat Dataset_. Kaggle. Retrieved May 2nd, 2023, from [https://www.kaggle.com/datasets/ssp1411/counsel-chat](https://www.kaggle.com/datasets/ssp1411/counsel-chat).
