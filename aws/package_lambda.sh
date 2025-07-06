#!/bin/bash
set -e

# Ensure output directory exists
mkdir -p output

# Build the Docker image
docker build -t lambda-builder .

# Remove any previous lambda_package
rm -rf output/lambda_package

# Run the container and copy out the lambda_package directory to output/
docker run --rm --entrypoint /bin/bash -v "$(pwd)/output":/output lambda-builder -c "cp -r /var/task/lambda_package /output/"

# Zip the package on the host using Python (cross-platform)
python3 -c "import shutil; shutil.make_archive('output/lambda_deploy', 'zip', 'output/lambda_package')"

echo "Build complete. output/lambda_deploy.zip is ready."