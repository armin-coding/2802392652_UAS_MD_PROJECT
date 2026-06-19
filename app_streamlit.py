import streamlit as st
import pandas as pd
from inferencing import ModelInference




# INISIALISASI BACKEND (INFERENCING) YANG BAKAL LOAD BESTMODEL DAN PREPROCESSING
# file best_model.pkl dan preprocessor.pkl hrs udah ada di folder models/

inference_system = ModelInference()




st.set_page_config(page_title="Credit Scoring System", layout="centered")
st.title("AI Credit Scoring Deployment")
st.write("Aplikasi ini digunakan untuk menilai performa dan kelas kredit nasabah secara real-time.")
st.markdown("-")

st.subheader("Form Data Nasabah Baru")



# form input UI kita
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia Nasabah (Age)", min_value=18, max_value=100, value=30, step=1)
    annual_income = st.number_input("Pendapatan Tahunan (Annual Income dalam USD)", min_value=0.0, value=50000.0, step=500.0)
    num_of_loan = st.number_input("Jumlah Pinjaman Aktif (Num of Loan)", min_value=0, max_value=20, value=2, step=1)
    num_of_delayed = st.number_input("Jumlah Pembayaran Terlambat (Num of Delayed Payment)", min_value=0, max_value=50, value=1, step=1)
    changed_limit = st.number_input("Perubahan Batas Kredit (Changed Credit Limit)", min_value=0.0, value=10.0, step=0.5)

with col2:
    outstanding_debt = st.number_input("Sisa Hutang (Outstanding Debt dalam USD)", min_value=0.0, value=1000.0, step=100.0)
    amount_invested = st.number_input("Investasi Bulanan (Amount Invested Monthly)", min_value=0.0, value=200.0, step=10.0)
    
    # input umur riwayat kredit (nanti diparsing oleh fungsi regex di inferencing.py)
    credit_history_year = st.slider("Riwayat Kredit (Tahun)", min_value=0, max_value=50, value=5)
    credit_history_month = st.slider("Riwayat Kredit (Bulan)", min_value=0, max_value=11, value=6)
    credit_history_age_str = f"{credit_history_year} Years and {credit_history_month} Months"
    
    # input Kategori
    occupation = st.selectbox("Pekerjaan (Occupation)", [
        "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
        "Lawyer", "Media_Manager", "Doctor", "Accountant", "Musician"
    ])
    credit_mix = st.selectbox("Bauran Kredit (Credit Mix)", ["Good", "Standard", "Bad"])
    payment_behaviour = st.selectbox("Perilaku Pembayaran", [
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments"
    ])

st.markdown("---")



# proses prediksi saat tombol diklik
if st.button("Evaluasi Skor Kredit", type="primary"):
    # Menyusun dictionary input yang strukturnya sama dengan dummy_input di inferencing.py
    input_data = {
        "Age": age,
        "Annual_Income": annual_income,
        "Num_of_Loan": num_of_loan,
        "Num_of_Delayed_Payment": num_of_delayed,
        "Changed_Credit_Limit": changed_limit,
        "Outstanding_Debt": outstanding_debt,
        "Amount_invested_monthly": amount_invested,
        "Credit_History_Age": credit_history_age_str,
        "Occupation": occupation,
        "Credit_Mix": credit_mix,
        "Payment_Behaviour": payment_behaviour
    }
    
    with st.spinner("Mengevaluasi data nasabah..."):
        try:
            # memanggil fungsi predict dari inferencing.py
            hasil_prediksi = inference_system.predict(input_data)
            
            # menampilkan hasil dengan warna indikator
            st.success("### Proses Evaluasi Selesai!")
            
            if str(hasil_prediksi).lower() == 'good':
                st.balloons()
                st.markdown(f"Hasil Penilaian Skor Kredit: **{hasil_prediksi.upper()}** (Performa Sangat Baik)")
            elif str(hasil_prediksi).lower() == 'standard':
                st.markdown(f"Hasil Penilaian Skor Kredit: **{hasil_prediksi.upper()}** (Performa Cukup/Normal)")
            else:
                st.markdown(f"Hasil Penilaian Skor Kredit: **{hasil_prediksi.upper()}** (Performa Berisiko Tinggi)")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
