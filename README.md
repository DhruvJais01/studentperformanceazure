## End to End Machine Learning Project

# studentperformanceazure

## Run from terminal:

docker build -t testdockermlops.azurecr.io/studentperform1:latest .

docker login testdockermlops.azurecr.io

docker push testdockermlops.azurecr.io/studentperform1:latest
