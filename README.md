# Practical 6: Dockerize the ML Prediction Application

Extends Practical 5 (Continuous Delivery) with a `Dockerfile` and `.dockerignore`.

## Run locally (without Docker)
```
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest test_ml_pipeline.py -v
python -m unittest test_app.py -v
```

## Build and run with Docker
Run from this folder (needs `student_result_model.pkl` and `metrics.json`, created by `train_model.py`):
```
docker build -t student-result-ml:1.0 .
docker run -d --name student-result-ml-container -p 5000:5000 student-result-ml:1.0
docker ps
docker logs student-result-ml-container
```
Health check: http://localhost:5000/

## Test the API (PowerShell)
```
$body = @{ attendance = 90; internal_marks = 85; assignment_marks = 88; previous_score = 80 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
```
Expected: `PASS / 1`. Low sample (55, 30, 40, 35) -> `FAIL / 0`.

## Cleanup
```
docker stop student-result-ml-container
docker rm student-result-ml-container
```
