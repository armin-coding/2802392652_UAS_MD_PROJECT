import re
import joblib
import numpy as np
import pandas as pd

class ModelInference:
    
    # load model dan preprocessor yg udah dilatih ama
    # lakuin prediksi pada data baru
    
    def __init__(self, model_path='models/best_model.pkl', preprocessor_path='models/preprocessor.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            print("[INFO] Model dan Preprocessor berhasil dimuat.")
        except FileNotFoundError as e:
            print(f"[ERROR] Artefak tidak ditemukan. Pastikan pipeline.py sudah dijalankan: {e}")

            

    def _clean_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        

        target_drops = [
            'unnamed: 0', 'id', 'customer_id', 'name', 
            'ssn', 'month', 'type_of_loan', 'backlogs',
            'total_emi_per_month', 'monthly_inhand_salary', 
            'credit_utilization_ratio', 'payment_of_min_amount', 
            'num_credit_card', 'num_credit_inquiries', 
            'delay_from_due_date', 'monthly_balance', 
            'interest_rate', 'num_bank_accounts'
        ]
        for col in df.columns:
            if col.lower() in target_drops:
                df = df.drop(columns=[col])

        # bersihin karakter non numerik
        str_num_cols = [
            'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
            'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly'
        ]
        for col in str_num_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        #parsing durasi kredit
        if 'Credit_History_Age' in df.columns:
            def parse_credit_age(s):
                if pd.isna(s): return np.nan
                years = re.search(r'(\d+)\s+Year', str(s))
                months = re.search(r'(\d+)\s+Month', str(s))
                y = int(years.group(1)) if years else 0
                m = int(months.group(1)) if months else 0
                return y * 12 + m
            df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_credit_age)



        # imputasi untuk inferencing
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in num_cols:
            df[col] = df[col].fillna(0)
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

        return df





    def predict(self, input_data: dict) -> str:
        """
        input dictionary yg dapet dari streamlit
        membersihkan data, mentransformasikannya, dan mengembalikan hasil prediksi.
        """
        # DICTIONARY JADI DATAFRAME
        df_input = pd.DataFrame([input_data])
        
        # bersihkan data
        df_clean = self._clean_inference_data(df_input)
        
        # transformasi dgn menggunakan ColumnTransformer yang sudah di-fit
        X_trans = self.preprocessor.transform(df_clean)
        
        # prediksi
        prediction = self.model.predict(X_trans)
        
        return prediction[0]





#pengujian lokal
if __name__ == "__main__":
    inference = ModelInference()
    
    # contoh data test case yang merepresentasikan nasabah
    dummy_input = {
        "Age": "28",
        "Annual_Income": "34000.50",
        "Num_of_Loan": "2",
        "Num_of_Delayed_Payment": "1",
        "Changed_Credit_Limit": "5.5",
        "Outstanding_Debt": "120.00",
        "Amount_invested_monthly": "50.00",
        "Credit_History_Age": "5 Years and 2 Months",
        "Occupation": "Engineer",
        "Credit_Mix": "Good",
        "Payment_Behaviour": "High_spent_Medium_value_payments"
    }
    
    hasil = inference.predict(dummy_input)
    print(f"\n[HASIL] Prediksi Kelas Credit Score: {hasil}")
