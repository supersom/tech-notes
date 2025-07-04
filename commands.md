# Common ML/DL Training Commands

## PyTorch Distributed Training

### Single GPU training

torchrun --nproc-per-node=1 src/train.py

### Multi-GPU training (e.g. 4 GPUs)

torchrun --nproc-per-node=4 src/train.py

### Specify master port

torchrun --nproc-per-node=2 --master_port=29500 src/train.py

## FastAPI

### Running an endpoint server with auto-reload

uvicorn fastapi_app.main:app --reload

### alternative port

uvicorn fastapi_app.main:app --reload --port 8001

## Docker Commands

### Build image

docker build -t myimage .

### Run container

docker run -it myimage

### Run with GPU

docker run --gpus all -it myimage

### Run with env var set and port mapping

docker run -e PYTHONPATH="/app/src:$PYTHONPATH" -p 8000:8000 myimage

### Run with mounted folder

There are bind-mounts and volumes

* bind-mounts: any folder in the host FS can be mapped to a container folder.
* volumes: Create a volume first. The volume is created in /var/lib/docker/volumes. Now mount that volume and it behaves like bind-mounts. Preferred for backups.

#### bind-mounts

Mount "~/code/mlops-iris" to "/app" and tracks changes
`docker run -v ~/code/mlops-iris:/app`

#### volumes

Create volume iris. Created in /var/lib/docker/volumes
`docker volume create iris`

`docker volume ls` # should show "iris"

Mounts volume "iris" to "/app" in container, dumps data from /app to iris and tracks changes
`docker run -v iris:/app myimage`

### Stop container

docker stop container_id

## Git Commands

### Create new branch

git checkout -b feature_branch

### Stage changes

git add .

### Commit changes

git commit -m "commit message"

### Push to remote

git push origin branch_name

## Kubernetes Commands

### Get pods

kubectl get pods

### Create deployment

kubectl create deployment myapp --image=myimage

### Scale deployment

kubectl scale deployment myapp --replicas=3

### Delete deployment

kubectl delete deployment myapp

## AWS CLI Commands

### List S3 buckets

aws s3 ls

### Copy to S3

aws s3 cp file.txt s3://{bucket-name}/

### Create EC2 instance

aws ec2 run-instances --image-id ami-id --instance-type t2.micro

### List instances

aws ec2 describe-instances

## Conda Commands

### Create environment

conda create -n myenv python=3.8

### Activate environment

conda activate myenv

### Install package

conda install package_name

### List environments

conda env list

## Curl Commands

### Basic GET request

curl https://api.example.com/endpoint

### POST request with data

curl -X POST -d "param1=value1&param2=value2" https://api.example.com/endpoint

### POST JSON data

curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' https://api.example.com/endpoint

### Download file

curl -O https://example.com/file.zip

### Follow redirects

curl -L https://example.com/redirecting-url

### Include headers in output

curl -i https://api.example.com/endpoint

### Use authentication

curl -u username:password https://api.example.com/endpoint

### Custom HTTP method

curl -X PUT -d "data" https://api.example.com/endpoint

### Save output to file

curl -o output.html https://example.com

### Send custom headers

curl -H "Authorization: Bearer token123" https://api.example.com/endpoint

## Snap

### find packages

snap find minio-client

## Linux

### find - search / for {aws_completer}

find / -name {aws_completer}

### Looking for env vars containing {STR}

printenv | grep {STR}

### Turn off/on bash history - prevent sensitive info typed into terminal from leaking into plain text

bash +o history # off
bash -o history # on

## Tools to investigate

tmux, nohup - so that process doesn't terminate when SSH closed
supervisord, systemd - services in production

## Python packages

python-dotenv
