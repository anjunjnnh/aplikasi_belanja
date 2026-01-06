import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Pencari Kode Belanja", page_icon="🔍")

st.title("🔍 Asisten Kode Belanja AI")
st.write("Masukkan nama barang/jasa, AI akan mencarikan kode rekening yang sesuai.")

# 2. Setup Koneksi (Mengambil rahasia dari Streamlit Secrets)
# Nanti kita atur ini di dashboard Streamlit, jangan tulis key di sini
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_connection()

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# 3. Interface Pengguna
query = st.text_input("Mau belanja apa?", placeholder="Contoh: Kertas HVS dan tinta printer")

if query:
    with st.spinner('Mencari kode yang cocok...'):
        # Encode query user ke vector
        query_vector = model.encode(query).tolist()

        # Panggil fungsi RPC di Supabase yang sudah kita buat di Langkah 1
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.3, # Sesuaikan sensitivitas
                "match_count": 5
            }
        ).execute()

        # Tampilkan Hasil
        st.subheader("Hasil Rekomendasi:")
        if response.data:
            for item in response.data:
                score = round(item['similarity'] * 100, 1)
                st.success(f"**{item['kode_rekening']}** - {item['nama_rekening']}")
                st.caption(f"Kecocokan: {score}%")
        else:
            st.warning("Tidak ditemukan kode yang relevan.")