<!-- PROJECT LOGO -->

<div align="center">
    <h1 style="font-weight: 900" align="center">ChatAIML2023</h3>
    <img src="/assets/readme-header.png" alt="Logo" width="100%">
</div>

---
## Demo
There is a <a href="http://chat-ai-ml.surge.sh/" target="_blank">front-end only demo deployed</a>, but that is without a backend so the bot repeats what you say. Nonetheless, it shows every other feature as well as the styling. Once the ML algo is fully implemented, the only thing different would be the chat's response.

---
# Updating the Model with Google Colab Jupyter Notebook

The model project gets edited in a <a href="https://colab.research.google.com/drive/1RchSMU53kyKMPEHizHPCaqskX5H404zb">Google Colab Notebook that can be found here</a>.

1. Access the Colab Notebook <a href="https://colab.research.google.com/drive/1RchSMU53kyKMPEHizHPCaqskX5H404zb">here</a>.
2. Up top go to 'File' -> 'Save a Copy in Github'. This will open the 'Copy to Github' Modal
3. Edit the fields so that it looks likes this (Note: Feel free to change the commit message to a useful summary of the change):
    <div align="center"> <img src="/assets/colab-save-to-github-screenshot.png" alt="colab-save-to-github-screenshot" width="100%" /> </div>
4. Hit 'Ok', and a git commit will directly push to this repo. 

Note: After doing this. it will only update the remote repo. To view the change locally, you need to push. The project would also have to be redeployed everytime an ML change is pushed (Perhaps a CI/CD that automatically redeploys when a push is made to the 'development' branch would simplifiy this).

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
