# SME Valuation Machine Learning Project

## 1. Project Overview
This project trains a regression model using a Random Forest algorithm to predict the `valuation_egp` of Egyptian SMEs based on their financial and operational features. The trained model is exported to ONNX format for seamless integration into production backends (such as C# .NET).

## 2. Folder Structure
```text
project_root/
├── data/
│   └── sme_valuation_mixed_dataset.csv
├── models/
│   ├── trained_model.pkl
│   └── sme_valuation_model.onnx
├── src/
│   ├── config.py         # Configuration settings
│   ├── preprocess.py     # Data loading and preprocessing pipeline
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Model evaluation script
│   ├── export_onnx.py    # ONNX export script
│   └── inference.py      # ONNX Runtime inference demonstration
├── tests/
│   └── test_model.py     # Unit tests for the trained model
├── notebooks/
│   └── exploration.ipynb # Jupyter notebook for EDA
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 3. How to create the virtual environment
To isolate project dependencies, create a Python virtual environment:
```bash
python -m venv venv
```

To activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

## 4. How to install dependencies
Once the virtual environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

## 5. How to train the model
Run the training script, which will load the data, apply preprocessing, train the random forest model, and save the binary model (`trained_model.pkl`) to the `models/` folder:
```bash
python src/train.py
```

## 6. How to evaluate it
To evaluate the trained model on a test set (computing R², MAE, RMSE), run:
```bash
python src/evaluate.py
```

## 7. How to export ONNX
After training, convert the model pipeline into an ONNX format to be deployed to the backend:
```bash
python src/export_onnx.py
```

## 8. How to run inference
You can test the exported ONNX model by running the inference demonstration script, which feeds sample data through the ONNX runtime:
```bash
python src/inference.py
```

You can also run the full test suite using `pytest` or `unittest`:
```bash
python -m unittest tests/test_model.py
```
