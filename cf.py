import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Function to calculate metrics
def calculate_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return accuracy, precision, recall, f1

# Function to get user inputs
def get_model_data(model_name, suffix):
    st.subheader(f"Enter data for {model_name} ({suffix})")
    TN = st.number_input(f'True Negatives for {model_name} ({suffix})', min_value=0, value=0, key=f'TN_{model_name}_{suffix}')
    TP = st.number_input(f'True Positives for {model_name} ({suffix})', min_value=0, value=0, key=f'TP_{model_name}_{suffix}')
    FN = st.number_input(f'False Negatives for {model_name} ({suffix})', min_value=0, value=0, key=f'FN_{model_name}_{suffix}')
    FP = st.number_input(f'False Positives for {model_name} ({suffix})', min_value=0, value=0, key=f'FP_{model_name}_{suffix}')
    return TN, TP, FN, FP

# Function to create a downloadable Excel file with color coding
def create_download_link(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    df.to_excel(writer, index=False, sheet_name='Model Comparison')

    workbook = writer.book
    worksheet = writer.sheets['Model Comparison']

    # Define a format for the best and worst scores
    highlight_best_format = workbook.add_format({'bg_color': 'lightgreen'})
    highlight_worst_format = workbook.add_format({'bg_color': 'lightcoral'})

    # Apply color coding for best and worst scores
    for col_num, col_name in enumerate(df.columns[1:], 1):  # Skip the 'Metric' column
        col_values = df[col_name]
        max_row = col_values.idxmax() + 1  # +1 to account for header row
        min_row = col_values.idxmin() + 1  # +1 to account for header row
        worksheet.conditional_format(1, col_num, len(df), col_num, 
                                     {'type': 'cell', 'criteria': '==', 'value': col_values.max(), 'format': highlight_best_format})
        worksheet.conditional_format(1, col_num, len(df), col_num, 
                                     {'type': 'cell', 'criteria': '==', 'value': col_values.min(), 'format': highlight_worst_format})

    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Main Streamlit app
def main():
    st.title("Confusion Matrix and Model Evaluation")

    num_models = st.number_input("How many models?", min_value=1, value=1, key='num_models')

    model_names = []
    for i in range(num_models):
        model_name = st.text_input(f"Enter name for model {i+1}", f"Model{i+1}", key=f'model_name_{i+1}')
        model_names.append(model_name)

    results = []

    for model_name in model_names:
        st.subheader(f"Train Data for {model_name}")
        train_TN, train_TP, train_FN, train_FP = get_model_data(model_name, "Train")
        
        st.subheader(f"Test Data for {model_name}")
        test_TN, test_TP, test_FN, test_FP = get_model_data(model_name, "Test")

        if any(v == 0 for v in [train_TN, train_TP, train_FN, train_FP, test_TN, test_TP, test_FN, test_FP]):
            results.append([model_name, "Train", "Information not provided", "Information not provided", "Information not provided", "Information not provided"])
            results.append([model_name, "Test", "Information not provided", "Information not provided", "Information not provided", "Information not provided"])
        else:
            train_acc, train_prec, train_rec, train_f1 = calculate_metrics(train_TP, train_TN, train_FP, train_FN)
            test_acc, test_prec, test_rec, test_f1 = calculate_metrics(test_TP, test_TN, test_FP, test_FN)
            
            results.append([model_name, "Train", train_prec, train_acc, train_rec, train_f1])
            results.append([model_name, "Test", test_prec, test_acc, test_rec, test_f1])

            # Check for overfitting
            for train_metric, test_metric, metric_name in zip([train_acc, train_prec, train_rec, train_f1], [test_acc, test_prec, test_rec, test_f1], ["Accuracy", "Precision", "Recall", "F1 Score"]):
                if abs(test_metric - train_metric) / train_metric > 0.1:
                    st.warning(f"Overfitting detected in {model_name} for {metric_name}")

    # Create DataFrame
    df = pd.DataFrame(results, columns=["Model", "Type", "Precision", "Accuracy", "Recall", "F1 Score"])

    # Reshape DataFrame for better readability
    df_reshaped = df.pivot_table(index=['Model', 'Type'], values=['Precision', 'Accuracy', 'Recall', 'F1 Score']).T.reset_index()
    
    # Rename columns for better readability
    df_reshaped.columns = ['Metric'] + [f'{col[0]} {col[1]}' for col in df_reshaped.columns[1:]]

    # Highlight best and worst scores
    def highlight_best_worst(s):
        is_best = s == s.max()
        is_worst = s == s.min()
        return ['background-color: lightgreen' if v else 'background-color: lightcoral' if w else '' for v, w in zip(is_best, is_worst)]

    st.subheader("Model Comparison")
    st.dataframe(df_reshaped.style.apply(highlight_best_worst, subset=df_reshaped.columns[1:], axis=1))

    # Provide option to download the results as an Excel file
    if st.button("Download Results as Excel"):
        excel_data = create_download_link(df_reshaped)
        st.download_button(label="Download Excel file", data=excel_data, file_name="model_evaluation_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Recommend the best model based on F1 Score
    best_model = df[df['Type'] == 'Test'].loc[df["F1 Score"].idxmax()]["Model"]
    st.subheader(f"Recommended Model: {best_model}")

if __name__ == "__main__":
    main()
