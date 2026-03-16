# ConVRsation (backend)
Backend for the **ConVRsation**, a VR-based learning application that integrates AI speech models to enable natural spoken dialogue with NPCs inside immersive environments.
The project explores how real-time voice interaction can support language learning in everyday scenarios such as a bakery or a friend's visit.
The corresponding Unity environments used to interact with this backend can be downloaded from the ConVRsation Unity Desktop repository and the ConVRsation Unity Quest 3 repository.

# How to Start the Application
1. Install all required Python packages.
```bash
pip install -r requirements.txt
```

2. Create a .env file by copying .env.example and provide the required environment variables.

3. Start the FastAPI server using Uvicorn. After starting, the backend will be available at `http://localhost:8000`. You can verify that the server ist running by opening `http://localhost:8000/health`.
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```