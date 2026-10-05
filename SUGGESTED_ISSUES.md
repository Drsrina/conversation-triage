# Suggested Issues for Conversation Triage

Based on an analysis of the current repository, here are several suggested issues that could be opened to improve the project:

## 1. Bug / Error Handling
**Title**: `FileNotFoundError` during prediction if models aren't trained
**Description**:
In `src/models/predict.py`, the `TriageClassifier` attempts to load models directly from the `models/` directory using `joblib.load()` during initialization. If a user starts the API without having run the training script first, the app will crash on startup or when predicting.
**Suggested Fix**:
Add a try-except block to handle `FileNotFoundError` and either raise a clearer custom error ("Models not found. Please run train.py first") or instantiate empty/dummy models with a warning.

## 2. Testing / CI
**Title**: Add Unit Tests and Pytest Integration
**Description**:
Currently, the repository lacks tests. We should add a `tests/` directory and create unit tests for the data preprocessing (`src/data/preprocess.py`), model training, and API endpoints.
**Suggested Fix**:
- Create `tests/test_preprocess.py` to verify `limpar_texto`, `stemming`, and `split_tags`.
- Create `tests/test_api.py` using `TestClient` from FastAPI to test `/` and `/predict`.
- Add test instructions to `README.md`.

## 3. Data Processing / NLTK Management
**Title**: Slow API Startup due to NLTK Downloads
**Description**:
In `src/data/preprocess.py`, there is a block of `try-except` statements downloading NLTK resources (`punkt`, `punkt_tab`, `rslp`). Since these are executed when the module is imported, it can slow down API startup time and fail if there's no internet access during startup (e.g., in a constrained container environment).
**Suggested Fix**:
Move NLTK downloads to a separate script (e.g., `scripts/download_nltk_data.py`) and run this script during the Docker build process (`RUN python scripts/download_nltk_data.py` in the `Dockerfile`), removing the runtime downloads.

## 4. Architecture / Model Handling
**Title**: Implement a Model Registry or Cloud Storage loading
**Description**:
The `models/` directory is gitignored, and the API relies on local files in that directory. For a production deployment on Google Cloud Run (as mentioned in the README), it's not ideal to bake models into the Docker image or train them before building locally.
**Suggested Fix**:
Update `TriageClassifier` to load models from Google Cloud Storage (GCS) on startup or use a proper Model Registry (like MLflow) if they don't exist locally. The `requirements.txt` already includes `google-cloud-storage`.

## 5. Security / Input Validation
**Title**: Add API Rate Limiting and Input Size Validation
**Description**:
The `/predict` endpoint accepts any string length in `PredictRequest`. A massive string could cause a Denial of Service (DoS) due to CPU-intensive text preprocessing (regex, NLTK stemming) and feature extraction.
**Suggested Fix**:
- Use Pydantic's `Field(max_length=...)` to limit the input text size in `PredictRequest`.
- Add a rate limiter (e.g., `slowapi`) to prevent API abuse.

## 6. Features / Machine Learning
**Title**: Add the DistilBERT (M2) Model Implementation
**Description**:
The `README.md` mentions "DistilBERT (M2)" as part of the stack, but the current code (`src/models/train.py`) only implements the M1 baseline (TF-IDF + Logistic Regression).
**Suggested Fix**:
Add a new script `src/models/train_bert.py` and modify `predict.py` to support loading and predicting with the DistilBERT model. Include logic to switch between M1 and M2 via an environment variable or API request parameter.

## 7. Refactoring / Code Quality
**Title**: Use Dependency Injection for `TriageClassifier` in FastAPI
**Description**:
In `src/api/main.py`, the `_classifier = TriageClassifier()` is instantiated as a global variable. This makes testing harder as we can't easily mock the classifier.
**Suggested Fix**:
Use FastAPI's `Depends` to inject the classifier instance into the `/predict` route. This aligns with FastAPI best practices and makes unit testing much easier.
