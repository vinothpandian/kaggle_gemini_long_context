# Gemini Long Context Competition Demo

This repository demonstrates using Google's Gemini model with long context to solve a Kaggle competition prediction task. The project showcases how to leverage Gemini's extended context window capabilities to analyze datasets and train binary classifiers.

## Overview

The demo uses Gemini to:
- Analyze a depression prediction dataset 
- Provide insights about the data
- Help train and evaluate machine learning models
- Make recommendations for model selection and optimization

## Requirements

- Python 3.11+
- uv package manager

## Setup

1. Install dependencies using uv:

    ```sh
    uv venv
    source .venv/bin/activate
    uv sync
    ```

2. Set up environment variables:
- Create a `.env` file
- Add your Google API key:
    ```
    GOOGLE_API_KEY=your_key_here
    ```

## Usage

Check `notebook.ipynb`to see the demonstration.

## Project Structure

```
.
├── data/               # Data files
├── experiment/         # Jupyter notebooks of multiple experiments before creating the final notebook
├── .env               # Environment variables
├── pyproject.toml     # Project configuration
└── README.md          # This file
```

## License

This project is for demonstration purposes as part of the Kaggle competition.