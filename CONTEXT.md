# MMA Predictive Modeling - AI Assistant Context

This document provides comprehensive context for AI coding assistants working with the MMA Predictive Modeling project. It covers architecture, implementation details, conventions, and important considerations for development.

## Project Overview

**Purpose**: Machine learning system that predicts Mixed Martial Arts (MMA) fight outcomes using multiple algorithms and automated cloud infrastructure.

**Technology Stack**:

- **Languages**: Python 3.12
- **ML Frameworks**: PyTorch, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Cloud**: AWS (SageMaker, Lambda, S3, EventBridge, ECS)

## Architecture

### Core Components

1. **Machine Learning Pipeline** (`code/`):

   - **9 Algorithms**: KNN, Naive Bayes, Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting, FCNN (PyTorch), Transformer (PyTorch)
   - **Data Processing**: Feature engineering, normalization, preprocessing
   - **Training**: Automated model training with hyperparameter optimization
   - **Evaluation**: Comprehensive metrics, learning curves, performance visualization
   - **Inference**: Multi-model predictions with confidence scoring

2. **AWS Infrastructure** (`aws/`):

   - **Training Pipeline**: Weekly automated retraining (Sundays)
   - **Inference Pipeline**: Weekly prediction generation (Fridays)
   - **Lambda Functions**: Event-driven job orchestration
   - **SageMaker**: Model training and inference execution
   - **S3**: Data storage, model persistence, results archival

## AWS Infrastructure Details

### Lambda Functions

**Training Trigger** (`lambda_training_job.py`):

- Triggered by S3 upload to `data/` or `experiments/`
- Supports dual-mode: main vs experimental training
- Configures SageMaker with ml.g4dn.xlarge instances
- Environment-based configuration management

**Inference Trigger** (`lambda_inference_job.py`):

- Triggered by `data/upcoming_fights.json` upload
- Creates SageMaker training job for inference
- Passes S3 bucket configuration as hyperparameters

**API Endpoints**:

- `lambda_api_predictions.py`: Serves latest predictions
- `lambda_api_past_predictions.py`: Historical prediction archive
- `lambda_api_training_plots.py`: Learning curve visualization with optimized S3 URLs
- `lambda_api_model_leaderboard.py`: Pre-calculated model performance statistics (NEW)

### IAM Permissions

**Lambda Execution Roles**:

Each Lambda function requires specific IAM permissions. See `.env` file for AWS account details and resource configurations.

**Required Policies**:
- **S3 Access**: `s3:GetObject`, `s3:ListBucket` for bucket and objects
- **Parameter Store Access**: `ssm:GetParameter`, `ssm:GetParameters` for secrets, plus `kms:Decrypt` with `kms:ViaService` condition for SecureString parameters
- **CloudWatch Logs**: `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`

**Common Permission Issues**:
- `AccessDeniedException` for Parameter Store: Add `ssm:GetParameter` and `kms:Decrypt` permissions
- S3 access errors: Verify bucket policy and IAM role permissions
- KMS decrypt errors: Ensure Lambda role has `kms:Decrypt` with `kms:ViaService` condition for SSM

### SageMaker Configuration

- **Instance Type**: ml.g4dn.xlarge (GPU-enabled for PyTorch)
- **Container**: PyTorch 1.13.1 training image
- **Storage**: 30GB volume size
- **Timeout**: Training (6 hours), Inference (1 hour)
- **Input**: S3 data paths, hyperparameters via environment

### Automated Workflows

**Weekly Training** (Sundays):

1. ECS data scraper updates fight data
2. S3 upload triggers Lambda
3. SageMaker retrains all models
4. Results uploaded to S3
5. GitHub PR created with new results

**Weekly Inference** (Fridays):

1. ECS scraper gets upcoming fights
2. S3 upload triggers inference Lambda
3. SageMaker generates predictions
4. Results published to website

**Prediction Archival & Performance Updates**:

1. New predictions archived automatically
2. Model leaderboard recalculated when new events added
3. Cached statistics refreshed for optimal performance

## Performance Optimization Architecture

### Backend Optimizations

**Model Leaderboard API** (`lambda_api_model_leaderboard.py`):

- **Purpose**: Pre-calculates model performance statistics server-side
- **Caching**: S3-based caching with 24-hour TTL
- **Endpoints**: GET `/model-leaderboard`, POST `/model-leaderboard/refresh`

**Training Plot Optimization** (`lambda_api_training_plots.py`):

- **Direct S3 URLs**: Returns pre-signed URLs when possible
- **Fallback Strategy**: Base64 encoding for non-public content
- **Bandwidth Reduction**: Eliminates large plot transfers through API

**Prediction Archiver Integration** (`lambda_prediction_archiver.py`):

- **Automatic Refresh**: Triggers leaderboard recalculation on new data
- **Event-Driven**: Ensures statistics stay current with new predictions

### Frontend Integration

**API Caching Service**: Comprehensive caching layer for all MMA API calls

- **Dual Caching**: Memory + sessionStorage for optimal performance
- **TTL Management**: Configurable cache lifetimes by endpoint type
- **Cache Strategies**: 5 minutes (predictions) to 24 hours (training data)
- **Error Handling**: Graceful fallbacks and retry mechanisms

## Data Schema

### Historical Fight Data (`fight_events.csv`)

```
Columns: EventName, EventDate, Fighter1_Name, Fighter1_Height, Fighter1_Reach,
         Fighter1_Stance, Fighter2_Name, Fighter2_Height, Fighter2_Reach,
         Fighter2_Stance, Winner, Method, Round, Time, WeightClass,
         Fighter1_ControlTime, Fighter2_ControlTime, [strike statistics], etc.
```

### Upcoming Fights (`upcoming_fights.json`)

```json
{
  "EventName": "UFC XXX",
  "EventDate": "YYYY-MM-DD",
  "Fights": [
    {
      "Fighter1Name": "Name",
      "Fighter1Url": "ufcstats.com URL",
      "Fighter2Name": "Name",
      "Fighter2Url": "ufcstats.com URL",
      "WeightClass": "Division"
    }
  ]
}
```

### Prediction Output

```json
{
  "fight_id": "fighter1_vs_fighter2",
  "predicted_winner": "Fighter Name",
  "confidence": 0.XX,
  "model_predictions": {
    "Random Forest": "Fighter1",
    "SVM": "Fighter2",
    // ... all 9 models
  },
  "model_agreement": 0.XX
}
```

## Important Considerations

### Model Training

- **Device Management**: Always check CUDA availability and move tensors properly
- **Hyperparameter Tuning**: All parameters centralized in `config.py`
- **Data Leakage**: Careful train/test splits, no future data in features
- **Normalization**: Critical for performance - 10% accuracy drop without it
- **Validation**: Cross-validation and learning curves for robust evaluation

### Feature Engineering

- **Missing Data**: Proper handling with defaults and validation
- **Categorical Encoding**: Consistent one-hot encoding across train/inference
- **Temporal Features**: Age calculation, time since last fight
- **Fighter URLs**: Used for historical stat lookup, handle missing gracefully

### AWS Integration

- **Environment Variables**: AWS credentials stored in `.env` (not in git). Critical for Lambda configuration.
- **S3 Paths**: Consistent naming conventions for data/model storage
- **Error Handling**: Robust exception handling for cloud operations
- **Permissions**: Proper IAM roles for SageMaker execution

### Performance Optimization

- **Batch Processing**: Efficient data loading for training
- **GPU Utilization**: Proper PyTorch device management
- **Memory Management**: Careful handling of large datasets
- **Parallel Processing**: Multi-core usage where applicable
- **API Optimization**: Backend pre-calculation of expensive operations
- **Caching Strategy**: Multi-layer caching for API responses (memory + sessionStorage)
- **Direct S3 Access**: Optimized image delivery bypassing API bottlenecks

## Experimental Framework

- **Documentation**: Structured format in `experiments/experiments.md` with results, findings, and reproducibility guidelines

## Common Development Tasks

### Adding New Models

1. Implement in `models.py` (PyTorch or sklearn)
2. Add hyperparameters to `config.py`
3. Update `initialize_models()` function
4. Test training in `training.py`
5. Verify evaluation in `evaluation.py`

### Modifying Features

1. Update feature engineering in `data.py`
2. Ensure consistent preprocessing for inference
3. Update feature lists and categorical handling
4. Test with sample data

### Performance Optimization Development

1. **New API Endpoints**: Add to appropriate lambda function
2. **Caching Strategy**: Consider data freshness requirements and update frequency
3. **Frontend Integration**: Update MMAApiService.ts in payaam.dev project
4. **Cache Invalidation**: Implement proper cache refresh triggers

### AWS Deployment

1. Update Lambda functions if needed
2. Test locally before deployment
3. Verify environment variables
4. Monitor CloudWatch logs

#### Lambda Function Deployment

**Method**: Direct zip deployment with pip dependencies

**Process**:
1. Install dependencies with `pip install -r lambda_requirements.txt -t lambda-package/ --platform manylinux2014_x86_64 --only-binary=:all:` (Linux-compatible for Lambda runtime)
2. Copy Lambda function to package directory
3. Zip package and deploy via `aws lambda update-function-code`

**Critical Notes**:
- Must use `--platform manylinux2014_x86_64` flag for Linux-compatible binaries (Lambda runs on Amazon Linux)
- Function file and dependencies at root level of deployment package
- Use pinned versions in `lambda_requirements.txt`: `PyGithub==1.59.1`, `cryptography==3.4.8`
- Function naming: `mpm-[function-purpose]` (e.g., `mpm-github-pull-request`, `mpm-model-leaderboard`)

### Debugging

- Check device placement for PyTorch models
- Verify S3 paths and permissions
- Validate data shapes and types
- Use print statements for pipeline monitoring
- Check hyperparameter consistency
- **Performance Issues**: Monitor API call counts and cache hit rates
- **Lambda Cold Starts**: Consider provisioned concurrency for frequently accessed endpoints
- **Cache Debugging**: Use browser DevTools to inspect cache states and TTL values

## Testing

### Unit Tests (`tests/`)

- Data processing function validation (height, reach, strike parsing)
- Feature engineering verification
- Run with: `python -m pytest tests/`
