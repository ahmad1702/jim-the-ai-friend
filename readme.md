<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="/assets/readme-header.png" alt="Logo" height="80">
    <h1 style="font-weight: 900" align="center">ChatAIML2023</h3>
</div>

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
```
