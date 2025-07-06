# Dockerfile for building AWS Lambda deployment package for lambda_github_pull_request.py
FROM public.ecr.aws/lambda/python:3.13

# Set workdir
WORKDIR /var/task

# Copy requirements and install to lambda_package
COPY lambda_requirements.txt ./
RUN pip3 install -r lambda_requirements.txt -t lambda_package

# Copy lambda function code
COPY lambda_github_pull_request.py lambda_package/

CMD ["/bin/bash"]
