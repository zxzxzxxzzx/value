# 🏠 HDB Valuation Calculator (LITE)

![Python](https://img.shields.io/badge/python-3.5%2B-blue?logo=python)
![Dataset](https://img.shields.io/badge/dataset-37k%2B%20records-green?logo=databricks)
![Model Accuracy](https://img.shields.io/badge/accuracy-~71%25-orange?logo=github)
![License](https://img.shields.io/badge/license-Educational-lightgrey)
![Status](https://img.shields.io/badge/status-Stable-brightgreen?logo=github)

---

## 🔥 Description
The **HDB Valuation Calculator (LITE)** is a lightweight version of the original **HDB Valuation Calculator (HDBVC)**. It builds on existing resale datasets to provide accurate estimations of a flat’s resale value based on key factors.

This tool serves as a decision-making assistant to help homeowners and buyers understand fair market pricing. In today’s market, making price estimations without data can be misleading, especially for newer homeowners.

By centralizing and analyzing publicly available resale transaction data, this calculator offers a **transparent, data-driven solution** — empowering users with insights into how features such as location, flat type, and floor area influence pricing.

---

## 🔨 Backend
The backend uses **statistical analysis and exploratory data analysis (EDA)** to construct a predictive model for resale prices.

- Initially, 9 features were extracted:
  - Date
  - Town
  - Flat Type
  - Block
  - Street Name
  - Storey Range
  - Floor Area
  - Flat Model
  - Lease Details

- After testing dependencies, the following **primary features** were selected:
  - **Town**
  - **Flat Type**
  - **Storey Range**
  - **Floor Area**
  - **Flat Model**
  - **Remaining Lease**

The final model is a **4th-degree polynomial regression** trained on historical resale transactions.

---

## ✨ Features
- 📈 **Price Prediction** – Predict resale values with polynomial regression trained on HDB resale data.
- 🛠 **Data Processing** – Automatic cleaning, transformation, and feature engineering.
- 💻 **CLI Interface** – Lightweight command-line interface designed for Python IDLE (Python 3.5+).
- 📊 **Visualization** – Generate charts for market trends, flat type analysis, and pricing vs. floor area.
- 📂 **Sample Dataset** – 37,153 transaction records included for testing.

---

## 🚀 System Requirements

### Python Version
- **Python 3.5 or higher**
- Tested for compatibility with **Python IDLE Executor**

### Required Libraries
The following libraries are required (auto-installed in most environments):
- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`
- `colorama`

If a library fails to install automatically, install manually:
```bash
pip install <module>
```

---

## ⚡ How to Run on IDLE Executor

### Step 1: Start the Application
1. Open **Python IDLE**.
2. Navigate to the project folder.
3. Run the application:
   ```bash
   python main.py
   ```
   OR simply click **Run** inside IDLE on `main.py`.

### Step 2: Initial Setup (Automatic)
The app will automatically:
1. Load the full dataset (`sample_data.csv`).
2. Clean & preprocess the data.
3. Train the polynomial regression model.
4. Display a training summary with accuracy metrics.

**Example Training Summary**:
```
==================================================
              MODEL TRAINING SUMMARY
==================================================
Training Samples:      29,408
Original Features:     6
Training R²:           0.6993
Testing R²:            0.7059
Accuracy %:            70.59%
==================================================
```

**Main Menu After Setup**:
```
==================================================
                MAIN MENU
==================================================
1. Calculate HDB Price
2. View Results History
3. Exit
==================================================
```

---

## 📂 Project Structure
```
HDBVC (LITE)
├── main.py                   # Main entry point
├── cli_interface.py          # CLI Engine
├── hdb_polynomial_model.py   # Machine Learning Model
├── data_processor.py         # Data Cleaning Engine
├── visualizer.py             # Visualization Engine
├── sample_data.csv           # Dataset (37,153 records)
└── /graphs/                  # Generated visualization files
```

---

## 🖥 IDLE Executor Specific Features
- **Console Clearing** – Compatible with Windows (`cls`), Unix (`clear`), and IDLE handling.
- **Colored Output** – Uses `colorama` (fallback to plain text if unavailable).
- **Input Validation** – Better error handling for invalid inputs.

---

## 🛠 Troubleshooting

**1. Import Errors**
```
ModuleNotFoundError: No module named 'pandas'
```
✅ Install missing library:
```bash
pip install pandas
```

**2. Data File Missing**
```
FileNotFoundError: sample_data.csv
```
✅ Ensure `sample_data.csv` exists in the root directory.

**3. Colors Not Displaying**
- Some IDLE versions may not support `colorama`.
✅ Program falls back to plain text automatically.

**4. Slow Performance**
- First run takes ~10–15s due to training (dataset has 37,153 records).
- Predictions afterward are instant (<1s).

---

## 📊 Performance Notes
- **Training Time**: ~10–15 seconds
- **Prediction Time**: <1 second
- **Memory Usage**: ~50–100MB
- **Model Accuracy**: ~71% (R² score)

---

## 📈 Advanced Usage

### Viewing Charts
Generated charts are saved to `/graphs/`:
- `prediction_summary_[timestamp].png` – Feature contribution analysis
- `price_comparison_[timestamp].png` – Predicted vs. actual prices
- `market_analysis_[timestamp].png` – Market trend analysis

### Session Management
- Predictions saved per session (with timestamps).
- History viewable from main menu.
- Resets on app restart.

---

## 🧠 Technical Details
- **Algorithm**: Polynomial Regression (4th degree)
- **Features Used**: 6 original features → expanded to **209 polynomial features**
- **Validation**: Hold-out cross-validation (80% train, 20% test)
- **Metrics**: R², Accuracy %

📌 A full notebook with a deconstruction of the polynomial model will be released in **Version 1.1**.

---

## ⭐ Credits
- ML References:
  - [W3Schools](https://www.w3schools.com/python/python_ml_getting_started.asp)
  - [GeeksforGeeks](https://www.geeksforgeeks.org/)
  - [Data36 Polynomial Regression](https://data36.com/polynomial-regression-python-scikit-learn/)

- Tools Used:
  - [Replit](https://replit.com/)
  - [VS Code](https://code.visualstudio.com/)
  - [Jupyter](https://jupyter.org/)
  - [Google Colab](https://colab.google/)

- Dataset Source: [data.gov.sg](https://data.gov.sg/)
- Debugging Assistance: [Replit AI](https://replit.com/ai) & [ChatGPT](https://chatgpt.com/) (_minor usage_)
- Final Release: [GitHub](https://github.com/)

---

## 👨‍💻 Author
Developed by **Souritra Samanta**
📧 Email: souritrasamanta@gmail.com
🏫 Commonwealth Secondary School

---

## ⚠️ Disclaimer
This project is created for **educational purposes** and **local use**.
For production use, further validation and security improvements are required.

🔥 **PS**: Hopefully this version doesn’t crash.
📓 **PS2**: Testing notebook for the model will be released in **v1.1**.
