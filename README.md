# AI Model Training & Testing Dashboard

A web-based dashboard for downloading Wikipedia articles, training language models, and generating text.

## Features

- Download Wikipedia articles with customizable parameters
- Train language models on the downloaded data
- Test the trained model with text generation
- Real-time progress tracking and statistics
- Multi-threaded article downloading
- Configurable storage backend (local or Google Cloud Storage)
- Beautiful and responsive web interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- Google Cloud Storage account (for production use)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Set `TEST=true` for local storage or `TEST=false` for Google Cloud Storage
   - If using GCS, set `GOOGLE_BUCKET_DATA` to your data bucket name
   - Set up Google Cloud credentials if using GCS (see Configuration section)

## Configuration

### Local Storage
Set `TEST=true` in your `.env` file to store downloaded articles locally in the `data/wikipedia` directory.

### Google Cloud Storage
For production use with Google Cloud Storage:
1. Set `TEST=false` in your `.env` file
2. Set `GOOGLE_BUCKET_DATA` to your GCS bucket name for storing downloaded data
3. Set up authentication:
   - Option 1: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key file
   - Option 2: Use Google Cloud SDK authentication (`gcloud auth application-default login`)

## Project Structure

```
.
├── core/
│   ├── download_data.py    # Wikipedia article downloader
│   ├── prepare_data.py     # Data preparation utilities
│   ├── train.py           # Model training logic
│   └── model.pth          # Trained model (created after training)
├── templates/
│   ├── base.html          # Base template
│   ├── download.html      # Download page template
│   ├── train.html         # Training page template
│   └── test.html          # Testing page template
├── main.py                # FastAPI application
├── requirements.txt       # Project dependencies
└── .env                  # Environment configuration
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:8080
```

3. Using the Dashboard:
   - Click "Download Data" to fetch Wikipedia articles
   - Use the "Train" page to train the model on downloaded data
   - Test the model using the "Test" page for text generation

## API Endpoints

- `GET /`: Home page
- `GET /download`: Download articles page
- `GET /train`: Training page
- `GET /test`: Testing page
- `POST /api/download`: Start article download
- `POST /api/train`: Start model training
- `POST /api/predict`: Generate text using the trained model

## Data Storage

- Test Mode (`TEST=true`):
  - Articles are stored in the `data/wikipedia` directory in JSON format
  - The trained model is saved as `core/model.pth`

- Production Mode (`TEST=false`):
  - Articles are stored in the specified Google Cloud Storage bucket
  - Model files are still stored locally as `core/model.pth`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.