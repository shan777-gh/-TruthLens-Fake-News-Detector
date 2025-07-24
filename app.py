import streamlit as st
import torch
import time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import tldextract
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import feedparser
import urllib.parse

# Load models
model = DistilBertForSequenceClassification.from_pretrained("/kaggle/input/saved_model/scikitlearn/default/1") #Enter your model path
tokenizer = DistilBertTokenizerFast.from_pretrained("/kaggle/input/saved_model/scikitlearn/default/1") #Enter your model path
model_sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

trusted_headlines = [
    "Iran denounces Israeli airstrikes following nuclear tension",
    "Supreme Court rules in major free speech case",
    "US economy grows despite inflation pressure",
    "Floods displace thousands in Pakistan after record rains",
    "Journalists awarded Nobel Prize for disinformation fight"
]
trusted_embeddings = model_sbert.encode(trusted_headlines, convert_to_tensor=True)

source_scores = {
    "cnn.com": {"score": 95, "label": "Trusted"},
    "nytimes.com": {"score": 90, "label": "Trusted"},
    "bbc.com": {"score": 92, "label": "Trusted"},
    "foxnews.com": {"score": 70, "label": "Biased"},
    "infowars.com": {"score": 10, "label": "Unreliable"},
    "theonion.com": {"score": 5, "label": "Satire"},
    "reuters.com": {"score": 96, "label": "Trusted"},
    "aljazeera.com": {"score": 91, "label": "Trusted"},
    "guardian.co.uk": {"score": 90, "label": "Trusted"}
}

def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title + "\\n\\n" + article.text
    except:
        return ""

def check_source_credibility(url):
    domain = extract_domain(url)
    info = source_scores.get(domain)
    if info:
        return domain, info['score'], info['label']
    else:
        return domain, None, "Unknown ⚠️"

def predict_fake_news(text):
    if len(text.split()) > 300:
        text = " ".join(text.split()[:300])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = float(probs[0][pred])
    label = "Real ✅" if pred == 1 else "Possibly Fake"
    return label, round(confidence * 100, 2)

def rank_against_trusted(article_text):
    query_embedding = model_sbert.encode(article_text, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, trusted_embeddings)[0]
    best_score = float(similarity_scores.max())
    best_match_idx = int(similarity_scores.argmax())
    best_headline = trusted_headlines[best_match_idx]
    return best_headline, best_score

def verify_with_google_news(text, threshold=0.65):
    query = urllib.parse.quote(" ".join(text.split()[:10]))
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return [], 0
    headlines = [entry.title for entry in feed.entries[:10]]
    urls = [entry.link for entry in feed.entries[:10]]
    embeddings = model_sbert.encode(headlines, convert_to_tensor=True)
    query_embedding = model_sbert.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    matched_headlines = [(headlines[i], urls[i], float(similarities[i]))
                         for i in range(len(similarities)) if similarities[i] > threshold]
    return matched_headlines, len(matched_headlines)

# Streamlit app
st.set_page_config(page_title="TruthLens", layout="wide")
st.title("📰 TruthLens: Fake News Detection + Source Trust Ranking")

input_type = st.radio("Input Type", ["Text", "URL"])
text = ""
domain = ""

if input_type == "Text":
    text = st.text_area("Paste news article text here:", height=200)
elif input_type == "URL":
    url = st.text_input("Enter news article URL")
    if url:
        domain, score, label = check_source_credibility(url)
        st.write(f"🔗 Source: `{domain}` → **{label}** (Score: {score if score else 'N/A'})")
        text = extract_text_from_url(url)
        if not text:
            st.warning("⚠️ Could not fetch article content. Please try another URL.")

if st.button("Analyze") and text.strip():
    with st.spinner("🔍 Analyzing... please wait"):
        time.sleep(1)
        label, confidence = predict_fake_news(text)
        st.markdown(f"### 🧠 AI Verdict: `{label}` (Confidence: {confidence}%)")

        # Phase 3: Similarity Check with Known Trusted Headlines
        match, score = rank_against_trusted(text)
        st.markdown("---")
        st.markdown("### 🔁 Similarity Check with Trusted Headlines")
        st.info(f"**Closest match:** *{match}*")
        st.markdown(f"🧮 Similarity Score: **{score:.2f}**")
        if score > 0.7:
            st.success("✅ Highly similar to a known trusted article.")
        elif score > 0.5:
            st.warning("⚠️ Possibly related to a trusted story.")
        else:
            st.error("❌ No strong similarity found with trusted sources.")

        # Phase 4.5: Google News Cross-verification
        st.markdown("### 🌐 Verified by Multiple Trusted Sources (Google News)")
        try:
            matches, count = verify_with_google_news(text)
            if count >= 2:
                st.success(f"✅ Verified by {count} trusted headlines:")
            elif count == 1:
                st.warning(f"⚠️ Only 1 strong match found — partial support.")
            else:
                st.error("❌ No strong support from trusted sources.")

            for i, (title, link, score) in enumerate(matches):
                st.markdown(f"{i+1}. [{title}]({link}) — 🧮 Score: `{score:.2f}`")
        except Exception as e:
            st.error(f"Google News check failed: {e}")

        # Final verdict override logic
        st.markdown("---")
        st.markdown("### 🧾 Final Verdict")
        if count >= 2:
            final = "✅ Verified Real (Cross-source matched)"
        elif label == "Real ✅":
            final = "✅ Trusted Content"
        else:
            final = "⚠️ Needs Review"

        st.success(f"**{final}**")
