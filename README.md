# Farmer Income Prediction Dashboard

Interactive Streamlit dashboard for the IBM SkillsBuild Data Analytics with AI internship project.

## Features
- Dataset overview and income distribution
- Interactive exploratory analysis and correlation heatmap
- Train and compare Linear Regression, Decision Tree and Random Forest
- Predict annual income from user-entered features
- Download model metrics as CSV

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `lte_train.csv` in the app sidebar. The app expects a total-income target column such as `Target_Variable/Total Income`.

## Deploy publicly
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository you control.
2. Open https://share.streamlit.io/
3. Sign in with GitHub and choose **Create app**.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy. Upload the CSV through the dashboard after it opens.

Do not commit private farmer-level data to a public repository. The dashboard expects the dataset to be uploaded at runtime.
