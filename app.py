import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pyproj import Transformer
import re
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ==================== KONFIGŪRACIJA ====================
st.set_page_config(page_title="StopGuard AI - Hack4Vilnius", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    iframe {width: 100% !important;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 1. DUOMENŲ UŽKROVIMAS ====================

def normalize_text(text):
    """Teksto normalizavimas lietuviškiems simboliams"""
    if not isinstance(text, str): 
        return ""
    replacements = {
        'ą': 'a', 'č': 'c', 'ę': 'e', 'ė': 'e', 'į': 'i', 'š': 's', 'ų': 'u', 'ū': 'u', 'ž': 'z',
        'Ą': 'A', 'Č': 'C', 'Ę': 'E', 'Ė': 'E', 'Į': 'I', 'Š': 'S', 'Ų': 'U', 'Ū': 'U', 'Ž': 'Z'
    }
    text = text.lower()
    for lt, en in replacements.items():
        text = text.replace(lt, en)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@st.cache_resource
def get_transformer():
    return Transformer.from_crs("epsg:3346", "epsg:4326", always_xy=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('stoteles.csv')
        transformer = get_transformer()
        lon, lat = transformer.transform(df['X'].values, df['Y'].values)
        df['lon'] = lon
        df['lat'] = lat
        df = df.fillna("nėra")
        df['norm_name'] = df['pavadinimas'].apply(normalize_text)
        df['uid'] = df.index 
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_nlp_model():
    """NLP Komponentas #1: Sentence Transformers modelis semantinei paieškai"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ==================== 2. NLP KOMPONENTAS #1: SEMANTINĖ PAIEŠKA ====================

def semantic_search(query, df, model):
    """
    Semantinė paieška naudojant Sentence Transformers
    Pagrindimas: Pasirinkttas multilingual modelis, nes lietuviški pavadinimai
    Alternatyvos: TF-IDF (paprastesnis), BERT (didesnis, lėtesnis)
    """
    query_norm = normalize_text(query)
    stopwords = ['ar', 'stoteleje', 'stotele', 'yra', 'kur', 'stogas', 'suolas', 'svieslente']
    query_words = query_norm.split()
    valid_keywords = [w for w in query_words if len(w) > 3 and w not in stopwords]
    
    # Tekstinė paieška (greita)
    if valid_keywords:
        best_keyword = max(valid_keywords, key=len)
        mask = df['norm_name'].str.contains(best_keyword, regex=False)
        candidates = df[mask]
        if not candidates.empty:
            return candidates
    
    # AI semantinė paieška
    stop_embeddings = model.encode(df['pavadinimas'].unique(), convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    search_results = util.semantic_search(query_embedding, stop_embeddings, top_k=5)
    
    found_names = []
    for res in search_results[0]:
        if res['score'] > 0.5:
            found_names.append(df['pavadinimas'].unique()[res['corpus_id']])
    
    if found_names:
        return df[df['pavadinimas'].isin(found_names)]
    
    return pd.DataFrame()

# ==================== 3. NLP KOMPONENTAS #2: SENTIMENTŲ ANALIZĖ ====================

def analyze_sentiment(text):
    """
    Sentimentų analizė infrastruktūros aprašymams
    Pagrindimas: Rule-based sistemą, nes duomenys struktūrizuoti
    Alternatyvos: BERT sentiment (per daug kompleksinis šiems duomenims)
    """
    if not isinstance(text, str):
        return 0
    
    text = text.lower()
    
    # Teigiami signalai
    positive_keywords = ['yra', 'veikia', 'tvarkinga', 'gera', 'projektuojama']
    negative_keywords = ['nėra', 'dingęs', 'sugadinta', 'neveikia', 'bloga']
    
    pos_score = sum(1 for word in positive_keywords if word in text)
    neg_score = sum(1 for word in negative_keywords if word in text)
    
    # Grąžiname sentiment balą [-1, 1]
    total = pos_score + neg_score
    if total == 0:
        return 0
    
    return (pos_score - neg_score) / total

@st.cache_data
def calculate_sentiment_scores(_df):
    """Apskaičiuojame sentiment scores visiems laukams"""
    df = _df.copy()
    
    # Analizuojame kiekvieną infrastruktūros elementą
    df['sentiment_paviljonas'] = df['paviljonas'].apply(analyze_sentiment)
    df['sentiment_suolas'] = df['suolas'].apply(analyze_sentiment)
    df['sentiment_svieslente'] = df['svieslente'].apply(analyze_sentiment)
    df['sentiment_transporto_balsas'] = df['transporto_balsas'].apply(analyze_sentiment)
    
    # Bendras sentiment balas
    df['sentiment_total'] = (
        df['sentiment_paviljonas'] + 
        df['sentiment_suolas'] + 
        df['sentiment_svieslente'] + 
        df['sentiment_transporto_balsas']
    ) / 4
    
    return df

# ==================== 4. ML MODULIS: ANOMALIJŲ DETEKTORIUS ====================

def calculate_quality_score(row):
    """Infrastruktūros kokybės balas"""
    score = 50 
    if str(row['paviljonas']).lower() == 'yra': score += 20
    if str(row['suolas']).lower() == 'yra': score += 15
    if 'veikia' in str(row['transporto_balsas']).lower(): score += 15
    svieslente = str(row['svieslente']).lower()
    if 'nėra' in svieslente: score -= 5
    if 'dingęs' in svieslente: score -= 25
    if 'projektuojama' in svieslente: score += 5
    return max(0, min(100, score))

@st.cache_data
def detect_anomalies(_df):
    """
    Anomalijų detektorius naudojant Isolation Forest
    Pagrindimas: Unsupervised learning metodas, nes neturime labeled duomenų
    Alternatyvos: One-Class SVM (lėtesnis), Autoencoder (per kompleksinis)
    """
    df = _df.copy()
    
    # Paruošiame features
    df['has_paviljonas'] = (df['paviljonas'].str.lower() == 'yra').astype(int)
    df['has_suolas'] = (df['suolas'].str.lower() == 'yra').astype(int)
    df['has_svieslente'] = df['svieslente'].str.contains('veikia', case=False).astype(int)
    df['has_transporto_balsas'] = df['transporto_balsas'].str.contains('veikia', case=False).astype(int)
    
    # Features matrica
    features = df[['has_paviljonas', 'has_suolas', 'has_svieslente', 
                    'has_transporto_balsas', 'Kokybe']].values
    
    # Normalizuojame
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Isolation Forest modelis
    iso_forest = IsolationForest(
        contamination=0.1,  # 10% anomalijų
        random_state=42,
        n_estimators=100
    )
    
    df['anomaly'] = iso_forest.fit_predict(features_scaled)
    df['anomaly_score'] = iso_forest.score_samples(features_scaled)
    
    # -1 = anomalija, 1 = normali
    df['is_anomaly'] = df['anomaly'] == -1
    
    return df

# ==================== 5. ML MODULIS: CLUSTERING ====================

@st.cache_data
def cluster_stops(_df):
    """
    K-Means clustering stotelių grupavimui pagal kokybę
    Tikslas: Identifikuoti probleminių stotelių zonas
    """
    df = _df.copy()
    
    # Features clustering'ui
    features = df[['lat', 'lon', 'Kokybe']].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means su 4 klasteriais
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    
    return df, kmeans

# ==================== 6. CALLBACK FUNKCIJOS ====================

def select_stop_callback(stop_data):
    st.session_state.current_view_stop = stop_data
    confirm_msg = f"Rodau stotelę: **{stop_data['pavadinimas']}** ({stop_data['kryptis']})"
    st.session_state.messages.append({"role": "assistant", "content": confirm_msg})

# ==================== 7. PAGRINDINĖ PROGRAMA ====================

def main():
    # Antraštė
    st.title("🤖 StopGuard AI - Vilniaus Viešojo Transporto Analizė")
    st.caption("Hack4Vilnius 2024 | AI-powered infrastruktūros vertinimas")
    
    # Užkrauname duomenis
    df = load_data()
    if df.empty:
        st.error("❌ Įkelkite stoteles.csv failą!")
        return
    
    model = load_nlp_model()
    
    # Skaičiuojame visus metrikas
    df['Kokybe'] = df.apply(calculate_quality_score, axis=1)
    df = calculate_sentiment_scores(df)
    df = detect_anomalies(df)
    df, kmeans_model = cluster_stops(df)
    
    # Sidebar: Analizės įrankiai
    with st.sidebar:
        st.header("📊 Analizės Įrankiai")
        
        analysis_mode = st.radio(
            "Pasirinkite režimą:",
            ["💬 Chat Asistentas", "🔍 Anomalijų Analizė", "📈 Statistika", "🗺️ Clustering Žemėlapis"]
        )
        
        st.divider()
        
        # Statistika sidebar'e
        st.metric("Iš viso stotelių", len(df))
        st.metric("Vidutinė kokybė", f"{df['Kokybe'].mean():.1f}/100")
        st.metric("Anomalijų", df['is_anomaly'].sum())
        
        avg_sentiment = df['sentiment_total'].mean()
        sentiment_emoji = "😊" if avg_sentiment > 0 else "😐" if avg_sentiment == 0 else "😟"
        st.metric("Vidutinis sentiment", f"{avg_sentiment:.2f} {sentiment_emoji}")
    
    # ==================== REŽIMAI ====================
    
    if analysis_mode == "💬 Chat Asistentas":
        render_chat_mode(df, model)
    
    elif analysis_mode == "🔍 Anomalijų Analizė":
        render_anomaly_mode(df)
    
    elif analysis_mode == "📈 Statistika":
        render_statistics_mode(df)
    
    elif analysis_mode == "🗺️ Clustering Žemėlapis":
        render_clustering_mode(df)

# ==================== CHAT REŽIMAS ====================

def render_chat_mode(df, model):
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Labas! Aš esu AI asistentas. Parašykite stotelės pavadinimą ir analizuosiu jos infrastruktūrą."
        }]
    
    if "current_view_stop" not in st.session_state:
        st.session_state.current_view_stop = None
    
    col_chat, col_visuals = st.columns([1, 1.3])
    
    with col_chat:
        st.header("💬 Pokalbis")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Rašykite čia..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            candidates = semantic_search(prompt, df, model)
            
            if candidates.empty:
                response = "Atsiprašau, neradau nieko panašaus. Galite patikslinti?"
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                if len(candidates) > 1:
                    response = f"Radau {len(candidates)} stoteles. Pasirinkite kryptį:"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.direction_candidates = candidates
                else:
                    stop = candidates.iloc[0]
                    st.session_state.current_view_stop = stop
                    response = f"Rodau stotelę: **{stop['pavadinimas']}**."
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()
        
        if "direction_candidates" in st.session_state and st.session_state.direction_candidates is not None:
            st.markdown("---")
            st.markdown("**Pasirinkite kryptį:**")
            
            for _, stop in st.session_state.direction_candidates.iterrows():
                stop_dict = stop.to_dict()
                
                if st.button(
                    f"👉 {stop['kryptis']}", 
                    key=f"chat_btn_{stop['uid']}",
                    on_click=select_stop_callback,
                    args=(stop_dict,)
                ):
                    st.session_state.direction_candidates = None
                    st.rerun()
    
    with col_visuals:
        stop = st.session_state.current_view_stop
        
        if stop is not None:
            render_stop_details(stop)
        else:
            render_welcome_screen(df)

# ==================== ANOMALIJŲ REŽIMAS ====================

def render_anomaly_mode(df):
    st.header("🔍 Anomalijų Detektorius")
    st.markdown("""
    **Metodas:** Isolation Forest (unsupervised learning)  
    **Tikslas:** Rasti stoteles su neįprastomis infrastruktūros kombinacijomis
    """)
    
    anomalies = df[df['is_anomaly'] == True].sort_values('anomaly_score')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Rastos {len(anomalies)} anomalijos")
        
        for idx, row in anomalies.head(20).iterrows():
            with st.expander(f"🚨 {row['pavadinimas']} ({row['kryptis']})"):
                st.write(f"**Anomaly Score:** {row['anomaly_score']:.3f}")
                st.write(f"**Kokybė:** {row['Kokybe']}/100")
                
                col_a, col_b = st.columns(2)
                col_a.write(f"Paviljonas: {row['paviljonas']}")
                col_a.write(f"Suolas: {row['suolas']}")
                col_b.write(f"Švieslentė: {row['svieslente']}")
                col_b.write(f"Transporto balsas: {row['transporto_balsas']}")
                
                st.caption("**Kodėl anomalija?** Nestandartinė infrastruktūros kombinacija")
    
    with col2:
        st.subheader("📊 Pasiskirstymas")
        
        fig = px.histogram(
            df, 
            x='anomaly_score',
            color='is_anomaly',
            title='Anomaly Score pasiskirstymas',
            labels={'anomaly_score': 'Anomaly Score', 'is_anomaly': 'Anomalija'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Žemėlapis su anomalijomis
        st.subheader("🗺️ Anomalijų žemėlapis")
        m = folium.Map(location=[54.6872, 25.2797], zoom_start=12, tiles="CartoDB positron")
        
        for _, row in anomalies.iterrows():
            folium.CircleMarker(
                [row['lat'], row['lon']], 
                radius=6, 
                color="red", 
                fill=True,
                popup=f"{row['pavadinimas']}<br>Score: {row['anomaly_score']:.3f}"
            ).add_to(m)
        
        st_folium(m, width=None, height=400)

# ==================== STATISTIKOS REŽIMAS ====================

def render_statistics_mode(df):
    st.header("📈 Išsami Statistika")
    
    tab1, tab2, tab3 = st.tabs(["📊 Kokybė", "💭 Sentimentai", "🏗️ Infrastruktūra"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x='Kokybe',
                nbins=20,
                title='Kokybės balų pasiskirstymas',
                labels={'Kokybe': 'Kokybės balas', 'count': 'Stotelių skaičius'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            quality_categories = pd.cut(df['Kokybe'], bins=[0, 50, 75, 100], labels=['Žema', 'Vidutinė', 'Aukšta'])
            fig = px.pie(
                values=quality_categories.value_counts().values,
                names=quality_categories.value_counts().index,
                title='Kokybės kategorijos'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sentimentų Analizė")
        
        fig = px.box(
            df,
            y=['sentiment_paviljonas', 'sentiment_suolas', 'sentiment_svieslente', 'sentiment_transporto_balsas'],
            title='Sentiment balai pagal infrastruktūros elementus'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Teigiamų sentimentų", len(df[df['sentiment_total'] > 0]))
        with col2:
            st.metric("Neigiamų sentimentų", len(df[df['sentiment_total'] < 0]))
    
    with tab3:
        st.subheader("Infrastruktūros Aprėptis")
        
        infra_stats = {
            'Paviljonas': (df['paviljonas'].str.lower() == 'yra').sum(),
            'Suolas': (df['suolas'].str.lower() == 'yra').sum(),
            'Švieslentė': df['svieslente'].str.contains('veikia', case=False).sum(),
            'Transporto balsas': df['transporto_balsas'].str.contains('veikia', case=False).sum()
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(infra_stats.keys()),
                y=list(infra_stats.values()),
                text=[f"{v}/{len(df)}" for v in infra_stats.values()],
                textposition='auto'
            )
        ])
        fig.update_layout(title='Infrastruktūros elementų aprėptis', yaxis_title='Stotelių skaičius')
        st.plotly_chart(fig, use_container_width=True)

# ==================== CLUSTERING REŽIMAS ====================

def render_clustering_mode(df):
    st.header("🗺️ Clustering Analizė")
    st.markdown("""
    **Metodas:** K-Means Clustering  
    **Tikslas:** Grupuoti stoteles pagal kokybę ir geografinę vietą
    """)
    
    # Cluster statistika
    col1, col2, col3, col4 = st.columns(4)
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        avg_quality = cluster_data['Kokybe'].mean()
        
        with [col1, col2, col3, col4][i]:
            st.metric(
                f"Cluster {i}",
                f"{len(cluster_data)} stotelių",
                f"Kokybė: {avg_quality:.1f}"
            )
    
    # Žemėlapis su cluster'iais
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='cluster',
        size='Kokybe',
        hover_name='pavadinimas',
        hover_data={'Kokybe': True, 'cluster': True},
        title='Stotelių clustering pagal kokybę ir vietą',
        zoom=11,
        height=600
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=54.6872, lon=25.2797))
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== STOTELĖS DETALĖS ====================

def render_stop_details(stop):
    if isinstance(stop, dict):
        stop_name = stop['pavadinimas']
        stop_kryptis = stop['kryptis']
        stop_gatve = stop['gatve']
        stop_lat = stop['lat']
        stop_lon = stop['lon']
        stop_kokybe = stop['Kokybe']
        stop_paviljonas = stop['paviljonas']
        stop_suolas = stop['suolas']
        stop_svieslente = stop['svieslente']
        stop_sentiment = stop.get('sentiment_total', 0)
        stop_anomaly = stop.get('is_anomaly', False)
    else:
        stop_name = stop['pavadinimas']
        stop_kryptis = stop['kryptis']
        stop_gatve = stop['gatve']
        stop_lat = stop['lat']
        stop_lon = stop['lon']
        stop_kokybe = stop['Kokybe']
        stop_paviljonas = stop['paviljonas']
        stop_suolas = stop['suolas']
        stop_svieslente = stop['svieslente']
        stop_sentiment = stop.get('sentiment_total', 0)
        stop_anomaly = stop.get('is_anomaly', False)
    
    st.subheader(f"📍 {stop_name}")
    st.caption(f"Kryptis: {stop_kryptis} | Gatvė: {stop_gatve}")
    
    # AI Analizė viršuje
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "green" if stop_kokybe >= 50 else "red"
        st.markdown(f"### :{color}[{stop_kokybe}/100]")
        st.caption("Kokybės balas")
    
    with col2:
        sentiment_emoji = "😊" if stop_sentiment > 0.2 else "😐" if stop_sentiment > -0.2 else "😟"
        st.markdown(f"### {sentiment_emoji} {stop_sentiment:.2f}")
        st.caption("Sentiment balas")
    
    with col3:
        anomaly_status = "🚨 Anomalija" if stop_anomaly else "✅ Normali"
        st.markdown(f"### {anomaly_status}")
        st.caption("ML analizė")
    
    # Infrastruktūros detalės
    with st.container(border=True):
        st.markdown("**🏗️ Infrastruktūros detalės:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"• **Paviljonas:** {stop_paviljonas}")
            st.markdown(f"• **Suolas:** {stop_suolas}")
        
        with col2:
            st.markdown(f"• **Švieslentė:** {stop_svieslente}")
            st.markdown(f"• **Transporto balsas:** {stop.get('transporto_balsas', 'nėra')}")
    
    st.divider()
    
    # Vizualizacijos
    tab1, tab2 = st.tabs(["📸 Street View", "🗺️ Žemėlapis"])
    
    with tab1:
        street_view_url = f"https://maps.google.com/maps?q=&layer=c&cbll={stop_lat},{stop_lon}&cbp=11,0,0,0,0&output=svembed"
        components.iframe(street_view_url, height=400, scrolling=False)
    
    with tab2:
        m = folium.Map(location=[stop_lat, stop_lon], zoom_start=18, tiles="CartoDB positron")
        folium.Marker(
            [stop_lat, stop_lon],
            popup=stop_name,
            icon=folium.Icon(color="green" if stop_kokybe >= 50 else "red", icon="bus", prefix="fa")
        ).add_to(m)
        st_folium(m, width=None, height=350)

def render_welcome_screen(df):
    st.markdown("### 👋 Sveiki atvykę į StopGuard AI!")
    st.markdown("""
    **AI-powered viešojo transporto infrastruktūros analizė**
    
    🤖 **Naudojami AI/ML metodai:**
    1. **Sentence Transformers** - semantinė paieška (NLP #1)
    2. **Sentimentų analizė** - infrastruktūros būklės vertinimas (NLP #2)
    3. **Isolation Forest** - anomalijų detektorius (ML)
    4. **K-Means Clustering** - geografinė analizė (ML)
    
    📊 **Sistemos galimybės:**
    - Natūralios kalbos paieška
    - Automatinis kokybės vertinimas
    - Anomalijų aptikimas
    - Statistinė analizė
    
    *Pradėkite pokalbį kairėje arba pasirinkite analizės režimą sidebar'e!*
    """)
    
    # Probleminių stotelių žemėlapis
    st.markdown("##### 🚨 Top 50 Probleminių Stotelių")
    m = folium.Map(location=[54.6872, 25.2797], zoom_start=12, tiles="CartoDB positron")
    bad_stops = df[df['Kokybe'] < 50].head(50)
    
    for _, row in bad_stops.iterrows():
        folium.CircleMarker(
            [row['lat'], row['lon']], 
            radius=4, 
            color="red", 
            fill=True,
            popup=f"{row['pavadinimas']}: {row['Kokybe']}/100"
        ).add_to(m)
    
    st_folium(m, width=None, height=400)

if __name__ == "__main__":
    main()