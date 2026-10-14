# Used Car Price Prediction

## Objective
Estimate used-car listing prices in US dollars.

## Workflow
EDA, data cleaning, feature engineering, encoding and scaling.
Compare Ridge, Random Forest and Gradient Boosting with a median baseline.
Apply five-fold cross-validation and hyperparameter tuning.
Evaluate the selected model using test MAE, RMSE and R².

## Leakage Prevention
Fit preprocessing inside model pipelines.
Keep test data separate from model selection.

## Run the Application
python -m pip install -r requirements.txt
python -m streamlit run app.py

## Limitations
Predictions estimate historical listing prices.
Rare vehicles and extreme prices may have larger errors.
