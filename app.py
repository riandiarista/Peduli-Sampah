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
import textwrap

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


def generate_insights_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan insight dalam bentuk DataFrame dua kolom: 'Insight' dan 'Detail'.
    Cocok untuk ditampilkan sebagai tabel dan diekspor ke Excel/PDF.
    """
    rows = []
    total = len(df)
    rows.append(("Total baris setelah clustering", f"{total}"))

    # Statistik dasar
    if 'Volume_Sampah' in df.columns:
        avg_vol = df['Volume_Sampah'].mean()
        med_vol = df['Volume_Sampah'].median()
        rows.append(("Rata-rata Volume Sampah", f"{avg_vol:.2f}"))
        rows.append(("Median Volume Sampah", f"{med_vol:.2f}"))

    if 'Sampah_Terkelola' in df.columns and 'Volume_Sampah' in df.columns:
        avg_terkelola = df['Sampah_Terkelola'].mean()
        rows.append(("Rata-rata Sampah Terkelola", f"{avg_terkelola:.2f}"))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = (df['Sampah_Terkelola'] / df['Volume_Sampah']).replace([np.inf, -np.inf], np.nan).fillna(0)
        avg_ratio = ratios.mean()
        rows.append(("Rata-rata proporsi sampah terkelola", f"{avg_ratio*100:.1f}%"))

    # Cluster summary
    if 'Label_Cluster' in df.columns:
        counts = df['Label_Cluster'].value_counts()
        top = counts.idxmax()
        top_pct = counts.max() / total * 100 if total > 0 else 0
        rows.append(("Cluster terbanyak", f"{top} ({top_pct:.1f}%)"))

        grp_vol = df.groupby('Label_Cluster')['Volume_Sampah'].mean()
        high_vol_cluster = grp_vol.idxmax()
        low_vol_cluster = grp_vol.idxmin()
        rows.append(("Cluster rata-rata Volume tertinggi", f"{high_vol_cluster} ({grp_vol.max():.2f})"))
        rows.append(("Cluster rata-rata Volume terendah", f"{low_vol_cluster} ({grp_vol.min():.2f})"))

        grp_ratio = (df.assign(_ratio=(df['Sampah_Terkelola'] / df['Volume_Sampah']).replace([np.inf, -np.inf], np.nan).fillna(0))
                     .groupby('Label_Cluster')['_ratio'].mean())
        worst_managed = grp_ratio.idxmin()
        rows.append(("Cluster proporsi terkelola terendah", f"{worst_managed} ({grp_ratio.min()*100:.1f}%)"))

    # Tren sederhana jika tersedia kolom Tahun
    if 'Tahun' in df.columns:
        try:
            yearly = df.groupby('Tahun')['Volume_Sampah'].mean().sort_index()
            if len(yearly) >= 2:
                coeffs = np.polyfit(yearly.index.astype(int), yearly.values, 1)
                slope = coeffs[0]
                trend = 'meningkat' if slope > 0 else ('menurun' if slope < 0 else 'stabil')
                rows.append(("Tren rata-rata Volume per Tahun", f"{trend} (slope={slope:.3f})"))
        except Exception:
            pass

    # (Rekomendasi dibuat terpisah oleh fungsi generate_recommendations_list)

    insight_df = pd.DataFrame(rows, columns=["Insight", "Detail"])
    return insight_df


def generate_recommendations_list(df: pd.DataFrame) -> list:
    """
    Menghasilkan daftar rekomendasi (list of strings) berdasarkan DataFrame.
    Rekomendasi ditampilkan terpisah dari tabel insight.
    """
    recs = []
    try:
        # rekomendasi umum
        recs.append("Fokus pada cluster dengan proporsi sampah terkelola rendah untuk intervensi.")
        recs.append("Periksa area dalam cluster dengan volume tinggi untuk peningkatan kapasitas pengelolaan.")

        # rekomendasi tambahan berdasarkan per-cluster ratio
        if 'Label_Cluster' in df.columns and 'Volume_Sampah' in df.columns and 'Sampah_Terkelola' in df.columns:
            grp = (df.assign(_ratio=(df['Sampah_Terkelola'] / df['Volume_Sampah']).replace([np.inf, -np.inf], np.nan).fillna(0))
                   .groupby('Label_Cluster')['_ratio'].mean())
            low = grp.nsmallest(2).index.tolist()
            if low:
                recs.append(f"Pertimbangkan intervensi prioritas pada cluster: {', '.join(map(str, low))} karena proporsi terkelola rendah.")
    except Exception:
        pass
    return recs

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

    # Salin dataset mentah untuk diolah (tetap simpan raw di session)
    raw = st.session_state['raw_data']
    st.subheader("1) Dataset Mentah (Preview)")
    with st.expander("Tampilkan dataset mentah", expanded=False):
        st.write(f"Ukuran: {raw.shape[0]} baris x {raw.shape[1]} kolom")
        st.dataframe(raw.head(10))

    # Langkah 1: Hapus duplikat
    st.subheader("2) Hapus Duplikat")
    before = raw.shape[0]
    data = raw.drop_duplicates().copy()
    after = data.shape[0]
    dropped = before - after
    st.write(f"Baris sebelum: **{before}**, setelah drop_duplicates: **{after}**. (Terhapus: **{dropped}**) ")

    # Langkah 2: Normalisasi nama kolom dan tampilkan mapping
    st.subheader("3) Normalisasi Nama Kolom")
    orig_cols = list(data.columns)
    map_orig_to_new = {}
    for orig in orig_cols:
        nc = orig.strip()
        nc = re.sub(r"[^\w\s]", "", nc)
        nc = nc.replace(" ", "_")
        map_orig_to_new[orig] = nc

    mapping_df = pd.DataFrame(list(map_orig_to_new.items()), columns=["Original", "Normalized"])
    st.write("Mapping nama kolom (Original -> Normalized):")
    st.dataframe(mapping_df)

    # Tampilkan kemungkinan collision (beberapa original -> same normalized)
    collisions = mapping_df.groupby('Normalized')['Original'].agg(list)
    collisions = collisions[collisions.apply(lambda x: len(x) > 1)]
    if not collisions.empty:
        st.warning("Terdapat beberapa nama kolom yang dikonsolidasikan (collision). Mereka akan digabungkan.")
        st.write(collisions.to_frame(name='Originals'))
    else:
        st.info("Tidak ada collision nama kolom yang terdeteksi.")

    # Langkah 3: Konsolidasi kolom yang map ke nama yang sama
    st.subheader("4) Konsolidasi Kolom (Rename / Merge)")
    new_names = set(map_orig_to_new.values())
    merged_info = []
    dropped_columns_total = []
    for new_name in new_names:
        originals = [o for o, n in map_orig_to_new.items() if n == new_name]
        if len(originals) == 1:
            # cukup ganti nama kolom jika berbeda
            if originals[0] != new_name:
                data.rename(columns={originals[0]: new_name}, inplace=True)
                merged_info.append((new_name, "renamed", originals))
        else:
            # gabungkan dengan nilai non-null pertama
            before_cols = data.columns.tolist()
            data[new_name] = data[originals].bfill(axis=1).iloc[:, 0]
            # drop semua kolom asli kecuali jika salah satu sudah bernama new_name
            if new_name not in originals:
                data.drop(columns=originals, inplace=True, errors='ignore')
                dropped_columns_total.extend(originals)
            else:
                drop_cols = [o for o in originals if o != new_name]
                data.drop(columns=drop_cols, inplace=True, errors='ignore')
                dropped_columns_total.extend(drop_cols)
            merged_info.append((new_name, 'merged', originals))

    if merged_info:
        st.write("Ringkasan tindakan rename/merge:")
        st.dataframe(pd.DataFrame(merged_info, columns=['Target', 'Action', 'Sources']))
    if dropped_columns_total:
        st.write("Kolom yang dihapus selama konsolidasi:")
        st.write(dropped_columns_total)

    # Langkah 4: Deteksi kolom numerik penting
    st.subheader("5) Deteksi Kolom Numerik (Volume & Terkelola)")
    volume_col = [c for c in data.columns if re.search(r'(?i)Timbulan|Volume', c)]
    terkelola_col = [c for c in data.columns if re.search(r'(?i)Terkelola', c)]
    st.write(f"Kolom yang terdeteksi untuk Volume: {volume_col}")
    st.write(f"Kolom yang terdeteksi untuk Terkelola: {terkelola_col}")

    # Langkah 5: Konsolidasi kolom numerik menjadi kolom standar
    st.subheader("6) Konsolidasi Kolom Numerik: Volume_Sampah & Sampah_Terkelola")
    dropped_numeric = []
    if volume_col:
        if len(volume_col) > 1:
            data['Volume_Sampah'] = data[volume_col].bfill(axis=1).iloc[:, 0]
        else:
            data['Volume_Sampah'] = data[volume_col[0]]
        for c in list(volume_col):
            if c != 'Volume_Sampah':
                try:
                    data.drop(columns=[c], inplace=True)
                    dropped_numeric.append(c)
                except Exception:
                    pass

    if terkelola_col:
        if len(terkelola_col) > 1:
            data['Sampah_Terkelola'] = data[terkelola_col].bfill(axis=1).iloc[:, 0]
        else:
            data['Sampah_Terkelola'] = data[terkelola_col[0]]
        for c in list(terkelola_col):
            if c != 'Sampah_Terkelola':
                try:
                    data.drop(columns=[c], inplace=True)
                    dropped_numeric.append(c)
                except Exception:
                    pass

    if dropped_numeric:
        st.write("Kolom numerik sumber yang dihapus:")
        st.write(dropped_numeric)

    # Langkah 6: Fill missing values untuk fitur numerik
    st.subheader("7) Penanganan Missing Values")
    for col in ['Volume_Sampah', 'Sampah_Terkelola']:
        if col in data.columns:
            null_before = data[col].isna().sum()
            mean_val = data[col].mean()
            data[col].fillna(mean_val, inplace=True)
            null_after = data[col].isna().sum()
            st.write(f"Kolom `{col}`: null sebelum = {null_before}, mean = {mean_val:.4f}, null sesudah = {null_after}")

    # Langkah akhir: Tampilkan dataset hasil preprocessing
    st.subheader("8) Dataset Setelah Preprocessing")
    st.write(f"Ukuran akhir: {data.shape[0]} baris x {data.shape[1]} kolom")
    st.dataframe(data.head(20))

    # Simpan hasil preprocessing ke session
    st.session_state['data'] = data
    st.success("Preprocessing selesai dan hasil disimpan di `st.session_state['data']`.")

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
    try:
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Hasil_Cluster', index=False)
            try:
                insight_df = generate_insights_df(data)
                insight_df.to_excel(writer, sheet_name='Insight', index=False)
                # write recommendations sheet as well
                recs = generate_recommendations_list(data)
                rec_df = pd.DataFrame({'Rekomendasi': recs})
                rec_df.to_excel(writer, sheet_name='Rekomendasi', index=False)
            except Exception:
                # jika gagal membuat sheet insight jangan gagalkan seluruh proses
                pass
        towrite.seek(0)
        st.download_button("Unduh Excel", data=towrite, file_name="hasil_cluster.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        # fallback sederhana bila terjadi error saat menulis multi-sheet
        towrite = BytesIO()
        data.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button("Unduh Excel (fallback)", data=towrite, file_name="hasil_cluster.xlsx",
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

    # Tampilkan Insight dalam bentuk tabel (DataFrame)
    try:
        insight_df = generate_insights_df(data)
        with st.expander("🔎 Insight (Kesimpulan) dari Visualisasi & Tabel", expanded=True):
            st.table(insight_df)

            # Tombol unduh untuk insight sebagai teks
            st.download_button(
                label="Unduh Insight (TXT)",
                data=insight_df.to_csv(index=False, sep=';'),
                file_name="insight_summary.txt",
                mime="text/plain"
            )

            # Tombol unduh insight sebagai Excel (Insight + Rekomendasi pada sheet terpisah)
            excel_buf = BytesIO()
            try:
                recs = generate_recommendations_list(data)
                rec_df = pd.DataFrame({'Rekomendasi': recs})
                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                    insight_df.to_excel(writer, sheet_name='Insight', index=False)
                    rec_df.to_excel(writer, sheet_name='Rekomendasi', index=False)
                excel_buf.seek(0)
                st.download_button(
                    label="Unduh Insight (Excel)",
                    data=excel_buf,
                    file_name="insight_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception:
                pass

            # Tampilkan rekomendasi di UI (di luar tabel)
            st.markdown("**Rekomendasi:**")
            for r in recs:
                st.markdown(f"- {r}")
    except Exception as e:
        st.warning(f"Gagal membuat insight otomatis: {e}")

    # Unduh semua visualisasi sebagai satu PDF multi-halaman
    st.subheader("Unduh Semua Visualisasi")
    # Buat buffer PDF sekarang dan tampilkan satu tombol unduh langsung
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_buffer = BytesIO()

    # Sertakan insight sebagai halaman terakhir PDF (tabel)
    try:
        insight_df_for_pdf = generate_insights_df(data)
        fig_insight = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax_ins = fig_insight.add_subplot(111)
        ax_ins.axis('off')
        # buat tabel matplotlib
        tbl = ax_ins.table(cellText=insight_df_for_pdf.values, colLabels=insight_df_for_pdf.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        figs.append(fig_insight)
        # tambahkan halaman rekomendasi sebagai teks (terpisah)
        try:
            recs_for_pdf = generate_recommendations_list(data)
            fig_rec = plt.figure(figsize=(8.27, 11.69))
            fig_rec.clf()
            fig_rec.text(0.02, 0.98, 'Rekomendasi:', va='top', fontsize=12, weight='bold')
            y = 0.94
            for r in recs_for_pdf:
                fig_rec.text(0.02, y, f"- {r}", va='top', fontsize=10)
                y -= 0.04
            plt.axis('off')
            figs.append(fig_rec)
        except Exception:
            pass
    except Exception:
        # Jika gagal membuat halaman insight, lanjutkan tanpa insight
        pass

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
