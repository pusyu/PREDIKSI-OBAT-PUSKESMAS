import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from docx import Document
from docx.shared import Inches
import os

# Pastikan folder untuk upload ada
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

# Fungsi preprocessing: penggabungan dan pembersihan data
def preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip', sep=None, engine='python')
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
    diagnosa_columns = ['Diagnosa 1']
    for col in diagnosa_columns:
        data[col] = data[col].fillna('')
    data['All_Diagnoses'] = data[diagnosa_columns].apply(lambda x: ' | '.join(x), axis=1)
    data['All_Diagnoses'] = data['All_Diagnoses'].str.replace(' \| ', ' | ')
    data.drop_duplicates(subset=['No. eRM', 'Tanggal'], keep='first', inplace=True)
    data_selected = data[['Tanggal', 'All_Diagnoses']].copy()
    data_selected.rename(columns={'All_Diagnoses': 'Diagnosa 1'}, inplace=True)
    data_selected['Tanggal'] = pd.to_datetime(data_selected['Tanggal'], errors='coerce').dt.date
    data_cleaned = data_selected.dropna(subset=['Tanggal', 'Diagnosa 1'])
    data_cleaned = data_cleaned[data_cleaned['Diagnosa 1'].str.strip() != '']
    data_aggregated = data_cleaned.groupby(['Tanggal', 'Diagnosa 1']).size().reset_index(name='Total Kasus')
    output_file_clean = os.path.join(UPLOAD_FOLDER, 'data_aggregated.csv')
    data_aggregated.to_csv(output_file_clean, index=False)
    return output_file_clean

# Fungsi untuk analisis dan pembuatan dokumen
def generate_report(file_path):
    data = pd.read_csv(file_path, parse_dates=['Tanggal'])
    diagnosa_counts = data.groupby('Diagnosa 1')['Total Kasus'].sum().reset_index()
    top_10_diagnoses = diagnosa_counts.sort_values(by='Total Kasus', ascending=False).head(10)

    output_file = "Top10_Diagnoses_Analysis_Weekly.docx"
    doc = Document()
    doc.add_heading("Analisis 10 Diagnosis Teratas (Resampling Mingguan)", level=1)

    for diagnosis_terpilih in top_10_diagnoses['Diagnosa 1']:
        doc.add_heading(f"Diagnosis: {diagnosis_terpilih}", level=2)
        data_diagnosis = data[data['Diagnosa 1'] == diagnosis_terpilih]
        data_diagnosis.set_index('Tanggal', inplace=True)
        weekly_data = data_diagnosis['Total Kasus'].resample('W').sum().fillna(0)
        doc.add_heading("Data Mingguan", level=3)
        doc.add_paragraph(weekly_data.to_string())

        adf_test = adfuller(weekly_data)
        adf_result = {
            "Test Statistic": adf_test[0],
            "p-Value": adf_test[1],
            "Lags Used": adf_test[2],
            "Number of Observations Used": adf_test[3]
        }
        doc.add_heading("Hasil Uji ADF", level=3)
        for key, value in adf_result.items():
            doc.add_paragraph(f"{key}: {value}")

        if adf_test[1] > 0.05:
            weekly_data_diff = weekly_data.diff().dropna()
            doc.add_paragraph("Data tidak stationer. Differencing dilakukan.")
        else:
            weekly_data_diff = weekly_data

        try:
            decomposition = seasonal_decompose(weekly_data, model='additive', period=26)
            decomposition.plot()
            seasonal_plot = f"{diagnosis_terpilih[:20]}_seasonal_weekly.png"
            plt.savefig(seasonal_plot)
            plt.close()
            doc.add_heading("Grafik Dekomposisi Musiman", level=3)
            doc.add_picture(seasonal_plot, width=Inches(6))
            os.remove(seasonal_plot)
        except Exception as e:
            doc.add_paragraph("Dekomposisi musiman gagal dilakukan.")
            doc.add_paragraph(f"Error: {e}")

        plt.figure(figsize=(12, 6))
        plot_acf(weekly_data_diff, ax=plt.subplot(121), title='ACF')
        plot_pacf(weekly_data_diff, ax=plt.subplot(122), title='PACF')
        plt.suptitle(f"ACF dan PACF untuk {diagnosis_terpilih} (Mingguan)", fontsize=16)
        plt.tight_layout()
        acf_pacf_plot = f"{diagnosis_terpilih[:20]}_acf_pacf_weekly.png"
        plt.savefig(acf_pacf_plot)
        plt.close()
        doc.add_heading("Plot ACF dan PACF", level=3)
        doc.add_picture(acf_pacf_plot, width=Inches(6))
        os.remove(acf_pacf_plot)

        try:
            model = ARIMA(weekly_data, order=(1, 1, 1))
            model_fit = model.fit()
            mse = ((model_fit.resid) ** 2).mean()
            doc.add_heading("Model ARIMA", level=3)
            doc.add_paragraph(f"Mean Squared Error (MSE): {mse}")
            forecast = model_fit.forecast(steps=12)
            doc.add_heading("Prediksi 12 Minggu ke Depan", level=3)
            doc.add_paragraph(forecast.to_string())
        except Exception as e:
            doc.add_paragraph(f"Model ARIMA gagal untuk {diagnosis_terpilih}. Error: {e}")

    doc.save(output_file)
    return output_file

# Streamlit app
st.markdown("""
    <style>
        body {
            background-color: #f4f7fc;
            color: #000;
            font-family: 'Arial', sans-serif;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #4CAF50;
            margin-bottom: 30px;
        }
        h2 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Prediksi Kebutuhan Obat di Puskesmas Berakit")
st.write("Upload file CSV untuk menganalisis data kebutuhan obat.")

uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])

if uploaded_file is not None:
    # Simpan file yang diupload
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess data
    st.write("Memproses data...")
    with st.spinner('Sedang memproses data...'):
        cleaned_file_path = preprocess_data(file_path)

    # Analisis dan hasilkan laporan
    st.write("Menjalankan analisis...")
    with st.spinner('Sedang menjalankan analisis...'):
        output_file = generate_report(cleaned_file_path)

    # Menampilkan tautan unduh
    st.success("Analisis selesai!")
    st.download_button(
        label="Unduh Hasil Analisis",
        data=open(output_file, 'rb'),
        file_name=output_file,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
