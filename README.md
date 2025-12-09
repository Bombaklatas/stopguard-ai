# 🤖 StopGuard AI - Vilniaus Viešojo Transporto Analizė

**Hack4Vilnius 2024** | AI-powered infrastruktūros vertinimo sistema

## 🎯 Projekto tikslas
Sukurti AI asistentas Vilniaus viešojo transporto stotelių infrastruktūros analizei, naudojant NLP ir ML metodus.

## 🤖 Naudojami AI/ML metodai

### NLP Komponentai:
1. **Sentence Transformers** (paraphrase-multilingual-MiniLM-L12-v2)
   - Semantinė paieška lietuviškais tekstais
   - Alternatyvos: TF-IDF (per paprastas), BERT (per lėtas)
   
2. **Sentimentų Analizė** (Rule-based)
   - Infrastruktūros būklės automatinis vertinimas
   - Keyword detection su pozityvių/negatyvių žodžių svoriais

### ML Moduliai:
3. **Isolation Forest** (Anomaly Detection)
   - Neįprastų infrastruktūros kombinacijų aptikimas
   - Unsupervised learning - nereikia labeled duomenų
   
4. **K-Means Clustering**
   - Geografinė stotelių segmentacija pagal kokybę
   - 4 cluster'iai probleminių zonų identifikavimui

## 📊 Funkcionalumas
- 💬 **Chat Asistentas** - NLP-powered interaktyvus dialogas
- 🔍 **Anomalijų Detektorius** - ML anomalijų aptikimas
- 📈 **Statistinė Analizė** - išsami duomenų vizualizacija
- 🗺️ **Clustering Žemėlapis** - geografinė analizė

## 🚀 Kaip paleisti

```bash
# 1. Klonuokite repozitoriją
git clone https://github.com/jusu-username/stopguard-ai.git
cd stopguard-ai

# 2. Įdiekite priklausomybes
pip install -r requirements.txt

# 3. Paleiskite aplikaciją
streamlit run app.py
```

## 📦 Technologijos
- **Frontend:** Streamlit
- **NLP:** Sentence Transformers, Custom Sentiment Analysis
- **ML:** Scikit-learn (Isolation Forest, K-Means)
- **Vizualizacija:** Plotly, Folium, PyDeck
- **Duomenys:** Hack4Vilnius Open Data

## 📋 Projekto struktūra
```
stopguard-ai/
├── app.py              # Pagrindinė aplikacija
├── stoteles.csv        # Duomenų failas
├── requirements.txt    # Python priklausomybės
└── README.md          # Dokumentacija
```

## 🎓 Egzamino reikalavimai
✅ Veikiantis web sprendimas (Streamlit)  
✅ 2 NLP komponentai (Semantic Search + Sentiment Analysis)  
✅ 2 ML moduliai (Isolation Forest + K-Means)  
✅ Hack4Vilnius duomenys  
✅ Interaktyvi vizualizacija  

## 👨‍💻 Autorius - Tomas Jagminas
Egzamino darbas - 2024

## 📄 Licencija
MIT License
# stopguard-ai
