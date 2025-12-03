import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import re

# Styling CSS
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background: linear-gradient(to bottom, #eef2f3, #8e9eab);
}
header {
    background-color: #3C792F;  /* Header hijau */
    color: white;
    text-align: center;
    padding: 10px 0;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header Aplikasi (tetap di semua halaman)
st.markdown("<header><h2>Aplikasi Monitoring Pengelolaan Sampah</h2></header>", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    # Tampilkan logo di atas navigasi (file: logo.png di root project)
    try:
        st.image('logo.png', width=200)
    except Exception:
        # Jika logo tidak ditemukan, tampilkan teks sederhana
        st.markdown("**PEDULI SAMPAH**")

    selected = option_menu(
        menu_title="Navigasi",
        options=["Home", "Input Data", "Preprocessing", "Clustering", "Visualisasi", "Tentang Kami"],
        icons=["house", "upload", "gear", "bar-chart", "map", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "#3C792F", "font-size": "20px"},  # Ikon hijau
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#ddd"},
            "nav-link-selected": {"background-color": "#3C792F", "color": "white"},  # Link terpilih hijau
        },
    )

# Session data
if "data" not in st.session_state:
    st.session_state["data"] = None
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

# === Functions ===
def home():
    st.header("Selamat Datang")
    st.write("""
    Aplikasi ini membantu memantau dan menganalisis data sampah per daerah.  
    Anda dapat:
    - Mengunggah dataset sampah atau input manual.
    - Membersihkan dan menormalisasi data.
    - Mengelompokkan wilayah berdasarkan tingkat timbulan sampah menggunakan K-Means.
    - Memvisualisasikan hasil clustering dan mengekspor laporan.
    """)

# --- Fungsi untuk Menghasilkan Data Simulasi ---
def generate_simulated_data(num_rows):
    """
    Menghasilkan DataFrame simulasi dengan kolom-kolom standar:
    Tahun, Provinsi, Kabupaten/Kota, Volume_Sampah, Sampah_Terkelola.
    """
    years = np.random.randint(2010, 2025, size=num_rows)
    
    provinces = ['Jawa Barat', 'DKI Jakarta', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara', 'Banten', 'Bali', 'Riau']
    cities = ['Bandung', 'Jakarta Selatan', 'Semarang', 'Surabaya', 'Medan', 'Tangerang', 'Denpasar', 'Pekanbaru']
    
    provinsi = np.random.choice(provinces, size=num_rows)
    kabupaten = np.random.choice(cities, size=num_rows)
    
    volume_sampah = np.round(np.random.uniform(100.0, 1000.0, size=num_rows), 2)
    
    terkelola_ratio = np.random.uniform(0.5, 0.8, size=num_rows)
    sampah_terkelola = np.round(volume_sampah * terkelola_ratio, 2)
    
    data = pd.DataFrame({
        "Tahun": years,
        "Provinsi": provinsi,
        "Kabupaten/Kota": kabupaten,
        "Volume_Sampah": volume_sampah,
        "Sampah_Terkelola": sampah_terkelola
    })
    
    return data

# --- Fungsi Utama Input Data yang Dimodifikasi (Hanya 2 Opsi) ---
def input_data():
    st.header("Input Data Sampah")
    
    # Tampilkan dataset saat ini
    if st.session_state.get('raw_data') is not None:
        with st.expander("Dataset Saat Ini (Mentah)", expanded=True):
            data_shape = st.session_state['raw_data'].shape
            st.write(f"Dataset berisi **{data_shape[0]} baris** dan **{data_shape[1]} kolom**.")
            st.dataframe(st.session_state['raw_data'])

    st.markdown("---")
    
    # Pilihan sekarang hanya memiliki 2 opsi
    choice = st.radio(
        "Pilih cara mendapatkan data:", 
        ["Unggah Dataset", "Hasilkan Data Simulasi"]
    )
    
    # --- OPSI 1: Unggah Dataset ---
    if choice == "Unggah Dataset":
        st.subheader("📁 Unggah Dataset")
        uploaded_file = st.file_uploader("Unggah dataset CSV atau Excel", type=["csv","xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith("csv"):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file, header=1) 

                # Bersihkan nama kolom
                data.columns = data.columns.astype(str).str.strip()
                data.columns = data.columns.str.replace(r"[^a-zA-Z0-9_ ]", "", regex=True) 
                data.columns = data.columns.str.replace(" ", "_")
                
                # Simpan dataset mentah
                st.session_state['raw_data'] = data

                # Tampilkan hasil
                st.subheader("✅ Dataset Berhasil Diunggah")
                st.write(f"Dataset berisi **{data.shape[0]} baris** dan **{data.shape[1]} kolom**.")
                st.dataframe(data)
            except Exception as e:
                st.error(f"File yang diunggah tidak valid atau formatnya salah. Error: {e}")

    # --- OPSI 2: Hasilkan Data Simulasi ---
    elif choice == "Hasilkan Data Simulasi":
        st.subheader("🔴 Hasilkan Data Simulasi")
        
        # Streamlit Slider untuk jumlah data
        num_data = st.slider(
            "Jumlah data yang dihasilkan:",
            min_value=10,
            max_value=100,
            value=50
        )
        
        st.info(f"Dataset simulasi akan berisi **{num_data} baris** dan **5 kolom**.")
        
        if st.button("Generate dan Gunakan Data"):
            simulated_df = generate_simulated_data(num_data)
            
            # Simpan DataFrame ke session_state
            st.session_state['raw_data'] = simulated_df
            
            # Tampilkan hasil
            st.success(f"Berhasil menghasilkan **{num_data}** baris data simulasi!")
            st.dataframe(st.session_state['raw_data'])

def preprocessing():
    st.header("Preprocessing Data")
    if st.session_state.get('raw_data') is None:
        st.warning("Harap unggah atau input data terlebih dahulu (dataset mentah).")
        return

    data = st.session_state['raw_data'].drop_duplicates()

    # Normalisasi nama kolom (sama seperti saat unggah file) dan konsolidasi
    # agar tidak ada duplikat kolom seperti 'Kabupaten/Kota' dan 'KabupatenKota'.
    orig_cols = list(data.columns)
    map_orig_to_new = {}
    for orig in orig_cols:
        nc = orig.strip()
        nc = re.sub(r"[^\w\s]", "", nc)
        nc = nc.replace(" ", "_")
        map_orig_to_new[orig] = nc

    # Untuk setiap target nama baru, gabungkan kolom yang peta ke nama tersebut
    new_names = set(map_orig_to_new.values())
    for new_name in new_names:
        originals = [o for o, n in map_orig_to_new.items() if n == new_name]
        if len(originals) == 1:
            # cukup ganti nama kolom
            data.rename(columns={originals[0]: new_name}, inplace=True)
        else:
            # gabungkan dengan mengambil nilai non-null pertama dari kiri ke kanan
            data[new_name] = data[originals].bfill(axis=1).iloc[:, 0]
            # drop semua kolom asli kecuali jika ada yang sudah bernama new_name
            drop_cols = [o for o in originals if o != new_name]
            # Jika tidak ada kolom dengan nama new_name di originals, drop semua originals
            if new_name not in originals:
                data.drop(columns=originals, inplace=True)
            else:
                data.drop(columns=drop_cols, inplace=True)

    # Deteksi kolom numerik (konsolidasi jika ada beberapa variasi nama)
    volume_col = [c for c in data.columns if re.search(r'(?i)Timbulan|Volume', c)]
    terkelola_col = [c for c in data.columns if re.search(r'(?i)Terkelola', c)]

    # Konsolidasi kolom volume menjadi satu kolom 'Volume_Sampah'
    if volume_col:
        if len(volume_col) > 1:
            data["Volume_Sampah"] = data[volume_col].bfill(axis=1).iloc[:, 0]
        else:
            data["Volume_Sampah"] = data[volume_col[0]]
        # Hapus kolom sumber (kecuali jika namanya sama dengan target)
        for c in list(volume_col):
            if c != "Volume_Sampah":
                data.drop(columns=[c], inplace=True, errors='ignore')

    # Konsolidasi kolom terkelola menjadi satu kolom 'Sampah_Terkelola'
    if terkelola_col:
        if len(terkelola_col) > 1:
            data["Sampah_Terkelola"] = data[terkelola_col].bfill(axis=1).iloc[:, 0]
        else:
            data["Sampah_Terkelola"] = data[terkelola_col[0]]
        # Hapus kolom sumber (kecuali jika namanya sama dengan target)
        for c in list(terkelola_col):
            if c != "Sampah_Terkelola":
                data.drop(columns=[c], inplace=True, errors='ignore')

    # Fill missing values
    for col in ["Volume_Sampah", "Sampah_Terkelola"]:
        if col in data.columns:
            data[col].fillna(data[col].mean(), inplace=True)

    # Tampilkan dataset dengan judul
    st.subheader("SIPSN - Dataset Setelah Preprocessing")
    st.dataframe(data)
    st.session_state['data'] = data
    st.success("Preprocessing selesai!")

def clustering():
    st.header("Clustering Sampah (K-Means)")
    if st.session_state['data'] is None:
        st.warning("Harap unggah atau input data terlebih dahulu.")
        return

    data = st.session_state['data']
    numeric_cols = ["Volume_Sampah", "Sampah_Terkelola"]

    for col in numeric_cols:
        if col not in data.columns:
            st.error(f"Kolom '{col}' tidak ditemukan! Silakan preprocessing terlebih dahulu.")
            return

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])

    # Elbow Method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        distortions.append(kmeans.inertia_)

    st.subheader("Menentukan jumlah cluster (Elbow Method)")
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bo-')
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("Distorsi (Inertia)")
    ax.set_title("Metode Elbow")
    st.pyplot(fig)

    num_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)

    centroids = kmeans.cluster_centers_[:,0]
    sorted_idx = np.argsort(centroids)
    labels_map = {}
    # Prepare label names safely for variable number of clusters (2..10)
    descriptive_labels = [
        "Sangat Rendah", "Rendah", "Agak Rendah", "Sedikit Rendah",
        "Sedang", "Agak Tinggi", "Tinggi", "Sangat Tinggi", "Ekstrem", "Sangat Ekstrem"
    ]
    if num_clusters <= 3:
        label_names = ["Rendah", "Sedang", "Tinggi"][:num_clusters]
    else:
        if num_clusters <= len(descriptive_labels):
            label_names = descriptive_labels[:num_clusters]
        else:
            label_names = [f"Cluster {i+1}" for i in range(num_clusters)]

    for i, idx in enumerate(sorted_idx):
        # Map cluster index (from kmeans) to a descriptive label based on centroid order
        labels_map[idx] = label_names[i]
    data['Label_Cluster'] = data['Cluster'].map(labels_map)

    st.subheader("Hasil Cluster")
    st.dataframe(data)

    # Export CSV/Excel
    st.subheader("Unduh Hasil Cluster")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh CSV", data=csv, file_name='hasil_cluster.csv', mime='text/csv')

    towrite = BytesIO()
    data.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button("Unduh Excel", data=towrite, file_name="hasil_cluster.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.session_state['data'] = data

def visualisasi():
    st.header("Visualisasi Data Sampah")
    if st.session_state['data'] is None:
        st.warning("Harap unggah atau input data terlebih dahulu.")
        return

    data = st.session_state['data']
    if 'Cluster' not in data.columns:
        st.warning("Clustering belum dilakukan.")
        return

    # Simpan semua plot untuk di-export ke PDF multi-halaman
    figs = []

    # Plot 1: Scatter Plot
    st.subheader("Volume Sampah vs Sampah Terkelola (Hasil Clustering)")
    fig1, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(data['Label_Cluster'].unique()))
    for i, label in enumerate(sorted(data['Label_Cluster'].unique())):
        cluster_data = data[data['Label_Cluster'] == label]
        ax.scatter(cluster_data['Volume_Sampah'], 
                  cluster_data['Sampah_Terkelola'],
                  label=label,
                  s=100,
                  alpha=0.7,
                  color=colors[i])
    ax.set_xlabel("Volume Sampah (ton)")
    ax.set_ylabel("Sampah Terkelola (ton)")
    ax.set_title("Scatter Plot Clustering: Volume Sampah vs Sampah Terkelola")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig1)
    figs.append(fig1)

    # Plot 2: Boxplot Volume Sampah
    st.subheader("Distribusi Volume Sampah per Cluster")
    fig2, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Label_Cluster', y='Volume_Sampah', data=data, ax=ax, palette="Set2")
    ax.set_ylabel("Volume Sampah (ton)")
    ax.set_xlabel("Cluster")
    st.pyplot(fig2)
    figs.append(fig2)

    # Plot 3: Boxplot Sampah Terkelola
    st.subheader("Distribusi Sampah Terkelola per Cluster")
    fig3, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Label_Cluster', y='Sampah_Terkelola', data=data, ax=ax, palette="Set3")
    ax.set_ylabel("Sampah Terkelola (ton)")
    ax.set_xlabel("Cluster")
    st.pyplot(fig3)
    figs.append(fig3)

    # Plot 4: Pie chart distribusi label cluster
    st.subheader("Perbandingan Distribusi Sampah Secara Keseluruhan (Pie Chart)")
    counts = data['Label_Cluster'].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values
    fig4, ax = plt.subplots(figsize=(4,4))
    colors = sns.color_palette('Set2', len(labels))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.pyplot(fig4)
    figs.append(fig4)

    # Unduh semua visualisasi sebagai satu PDF multi-halaman
    st.subheader("Unduh Semua Visualisasi")
    # Buat buffer PDF sekarang dan tampilkan satu tombol unduh langsung
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
    pdf_buffer.seek(0)
    st.download_button(
        label="📄 Print/Unduh Semua Grafik (PDF)",
        data=pdf_buffer,
        file_name="visualisasi_clustering.pdf",
        mime="application/pdf"
    )

def about_us():
    st.header("Tentang Kami")
    st.write("""
    **Pengembang**: Kelompok 2 TB Akuisisi Data Kelas B  

    **Anggota Kelompok**:  
    1. 2311522036 Rizka Putri Ananda  
    2. 2311522040 Ririn Fauzia Rahma  
    3. 2311523008 Riandi Arista Muhammad  

    **Tujuan**:  
    Membantu dalam menganalisis dan monitoring penumpukan sampah di tiap daerah menggunakan algoritma **K-Means Clustering**.  

    **Teknologi yang digunakan**:  
    Streamlit, Pandas, Scikit-learn, Plotly, dan Seaborn.
    """)

# === Menu Navigation ===
if selected == "Home":
    home()
elif selected == "Input Data":
    input_data()
elif selected == "Preprocessing":
    preprocessing()
elif selected == "Clustering":
    clustering()
elif selected == "Visualisasi":
    visualisasi()
elif selected == "Tentang Kami":
    about_us()
