# Setting up MLFlow server

NOTES: Running `> mlflow ui` automatically starts a local mlflow tracking server

Here's how to set up a **remote MLflow tracking server** using **S3 for artifact storage** and **PostgreSQL for tracking metadata** — ideal for production or team environments, and easily extendable to AWS deployment later.

It is also possible to set up an MLflow server that stores artifacts into file storage [`./mlruns`] or into a local object store [MinIO], and metadata into a local PostgreSQL DB.

---

## ✅ High-Level Architecture

```text
Your Code (train.py)
    │
    ├── logs metrics, parameters → PostgreSQL (tracking DB)
    └── uploads models/artifacts → S3 bucket
```

---

## ✅ Step-by-Step Setup

### 1. 📦 Requirements

Install dependencies:

```bash
pip install mlflow[extras] psycopg2-binary boto3
```

---

### 2. 📦 Set up Remote PostgreSQL DB

You can use:

* AWS RDS PostgreSQL
* Local PostgreSQL instance
* Dockerized PostgreSQL (for dev/test)

> For quick local testing, run:

```bash
docker run --name mlflow-postgres -p 5432:5432 -e POSTGRES_PASSWORD=mlflow -e POSTGRES_USER=mlflow -e POSTGRES_DB=mlflowdb -d postgres
```

---

### 3. 🪣 Set up S3 Bucket

* Create a new S3 bucket (e.g. `mlflow-artifacts-wh0am1`)
* Make sure your AWS credentials are configured locally (`~/.aws/credentials`)

---

### 4. ⚙️ Set Environment Variables

Set these in your shell or in `.env`:

```bash
export MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@localhost:5432/mlflowdb
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com  # for AWS, optional for other object storage
```

To use a **custom S3-compatible** (e.g., MinIO, Wasabi), adjust `MLFLOW_S3_ENDPOINT_URL`.

---

### 5. 🚀 Start MLflow Server

```bash
mlflow server \
  --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflowdb \
  --default-artifact-root s3://my-mlflow-artifacts/ \
  --host 0.0.0.0 \
  --port 5000
```

You now have:

* Web UI at `http://localhost:5000`
* Artifacts stored in S3
* Tracking data in PostgreSQL

---

### 6. 🔁 Modify Your `train.py`

Add this at the top before `mlflow.set_experiment(...)`:

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
```

> Or configure it globally with `export MLFLOW_TRACKING_URI=...` in your `.bashrc`

---

## 🧪 Test It

```bash
dvc repro
mlflow ui --backend-store-uri postgresql://...  # optional, if you want a second viewer
```

---

## ✅ Optional: Deploy on AWS EC2 or ECS

Once verified locally, you can:

* Deploy PostgreSQL (RDS)
* Keep artifacts on S3
* Run MLflow server on EC2
* Use a domain + SSL via Nginx/Cloudflare

---

Here's a simple `docker-compose.yml` setup that spins up PostgreSQL and a MinIO server (S3-compatible object storage) locally, plus instructions to run MLflow server connected to them — perfect for quick local dev with a remote-style MLflow tracking server.

---

### docker-compose.yml

```yaml
version: "3.8"

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflowdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  minio:
    image: minio/minio
    command: server /data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
    volumes:
      - miniodata:/data

volumes:
  pgdata:
  miniodata:
```

---

### Step-by-step to run

1. Start containers:

    ```bash
    docker-compose up -d
    ```

2. Set environment variables in your shell:

    ```bash
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    ```

3. Start MLflow server pointing to local PostgreSQL and MinIO artifact storage:

    ```bash
    mlflow server \
    --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflowdb \
    --default-artifact-root s3://mlflow-artifacts/ \
    --host 0.0.0.0 \
    --port 5000
    ```

4. Create the S3 bucket in MinIO (use [MinIO Console](http://localhost:9000) or `mc` CLI):

    ```bash
    # Install MinIO client (mc) if needed: https://min.io/docs/minio/linux/reference/mc.html
    mc alias set local http://localhost:9000 minioadmin minioadmin
    mc mb local/mlflow-artifacts
    ```

---

### Usage notes

* Your MLflow artifacts will be stored in `mlflow-artifacts` bucket on MinIO.
* PostgreSQL stores tracking metadata.
* Access MLflow UI at [http://localhost:5000](http://localhost:5000)
* Access MinIO UI at [http://localhost:9000](http://localhost:9000)

---

Awesome! Here’s how to update your `src/train.py` to use this local MLflow remote tracking setup with PostgreSQL + MinIO, assuming the environment variables are set as before.

---

### Updated `src/train.py` with remote MLflow tracking

This is a minimal `src/train.py` example demostrating the use to important mlflow interfaces - `set_tracking_uri`, `start_run`, `log_metric`, `log_artifact`

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000") # Set tracking URI if needed

with mlflow.start_run():
    mlflow.log_metric("my_metric", 123)
    with open("my_artifact.txt", "w") as f:
        f.write("This is a test artifact.")
    mlflow.log_artifact("my_artifact.txt")
```

... and this is an sklearn training example with remote MLflow training, demonstrating some other important interfaces - `set_experiment`, `log_param`, `log_metric`, `log_model`

```python
import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# MLflow tracking URI from env, fallback to localhost
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("IrisClassifier")

PROCESSED_DATA_PATH = "data/processed/iris_processed.csv"
MODEL_OUTPUT_PATH = "models/model.pt"

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=200)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 200)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        print(f"Validation Accuracy: {acc:.4f}")

        # Save model locally for DVC
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        with open(MODEL_OUTPUT_PATH, "wb") as f:
            pickle.dump(model, f)

        print(f"Saved trained model to {MODEL_OUTPUT_PATH}")

        # Log model to MLflow remote server
        mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    main()
```

---

### How to run

1. Make sure docker-compose services are running:

    ```bash
    docker-compose up -d
    ```

2. Set env vars in the terminal session where you run `train.py`:

    ```bash
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    ```

3. Run your training:

    ```bash
    python src/train.py
    ```

4. Visit the MLflow UI at [http://localhost:5000](http://localhost:5000) to see your logged runs and model artifacts.

---
