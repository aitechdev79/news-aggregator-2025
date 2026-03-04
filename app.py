import os
import re
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from io import BytesIO
from dateutil import parser
from pypdf import PdfReader
from bs4 import BeautifulSoup
import trafilatura
from langchain_openai import ChatOpenAI

# Load API keys
#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#SERPAPI_KEY = os.getenv("SERPAPI_KEY")

import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# Page config
st.set_page_config(page_title="News Aggregation Assistant (SerpAPI)", page_icon="📰", layout="wide")

# Initialize GPT model
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

llm = get_llm()

# === SERPAPI SEARCH FUNCTION ===
def serpapi_search(query, engine="google_news", lang="en", max_results=10):
    params = {"engine": engine, "q": query, "api_key": SERPAPI_KEY, "num": max_results}
    if engine in ["google", "google_news"]:
        params["hl"] = lang
    if engine == "bing_news":
        params["cc"] = "vn" if lang == "vi" else "us"

    resp = requests.get("https://serpapi.com/search", params=params)
    data = resp.json()
    results = []

    if engine == "google_news" and "news_results" in data:
        for item in data["news_results"][:max_results]:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "content": item.get("snippet", ""),
                "published_date": item.get("date", "Unknown"),
                "source": item.get("source", {}).get("name", "Unknown"),
                "thumbnail": item.get("thumbnail") or item.get("image")
            })
    elif engine == "google" and "organic_results" in data:
        for item in data["organic_results"][:max_results]:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "content": item.get("snippet", ""),
                "published_date": "Unknown",
                "source": item.get("source", "Google Search"),
                "thumbnail": item.get("thumbnail") or item.get("image")
            })
    elif engine == "bing_news" and "news_results" in data:
        for item in data["news_results"][:max_results]:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "content": item.get("snippet", ""),
                "published_date": item.get("date", "Unknown"),
                "source": item.get("source", {}).get("name", "Unknown"),
                "thumbnail": item.get("thumbnail") or item.get("image")
            })
    return results

# === DATE NORMALIZATION ===
def normalize_date(date_str, url=None):
    if not date_str:
        return "Unknown"
    try:
        if "ago" in date_str:
            num = int(date_str.split()[0])
            if "hour" in date_str:
                dt = datetime.utcnow() - timedelta(hours=num)
            elif "day" in date_str:
                dt = datetime.utcnow() - timedelta(days=num)
            else:
                dt = datetime.utcnow()
            return dt.strftime("%Y-%m-%d")
        dt = parser.parse(date_str, fuzzy=True)
        if dt.year < 2024 and url and "2025" in url:
            return "2025-" + dt.strftime("%m-%d")
        return dt.strftime("%Y-%m-%d")
    except:
        if url:
            m = re.search(r"(20\d{2})", url)
            if m:
                return m.group(1) + "-01-01"
        return "Unknown"

def filter_by_date(articles, days):
    if days == 0:
        return articles, []
    cutoff = datetime.utcnow() - timedelta(days=days)
    fresh, old = [], []
    for a in articles:
        norm_date = normalize_date(a.get("published_date"), a.get("url"))
        a["normalized_date"] = norm_date
        try:
            if norm_date != "Unknown":
                dt = parser.parse(norm_date)
                if dt >= cutoff:
                    fresh.append(a)
                else:
                    old.append(a)
            else:
                old.append(a)
        except:
            old.append(a)
    return fresh, old

# === PROMPTS ===
DETAILED_SUMMARY_TEMPLATE = """
You are a professional news summarizer.

Task: Provide a more thorough summary of the news article.

Article:
- Title: {title}
- Snippet: {snippet}
- URL: {url}

Requirements:
- Write 3 coherent paragraphs (each 3-5 sentences).
- Use a neutral, factual, professional tone.
- At the end, include: [Read full article]({url})
"""

# === HIGHLIGHT FUNCTION SAFE ===
def highlight_text_safe(text, keywords, style="bold"):
    if not keywords:
        return text
    urls = re.findall(r"\[.*?\]\((.*?)\)", text)
    placeholders = {}
    for i, url in enumerate(urls):
        key = f"__URL_PLACEHOLDER_{i}__"
        placeholders[key] = url
        text = text.replace(url, key)
    for kw in keywords:
        if kw.strip():
            if style == "bold":
                text = re.sub(rf"(?i)({re.escape(kw.strip())})", r"**\1**", text)
            elif style == "highlight":
                text = re.sub(rf"(?i)({re.escape(kw.strip())})", r"<mark>\1</mark>", text)
    for key, url in placeholders.items():
        text = text.replace(key, url)
    return text

# === CACHED DETAILED SUMMARY ===
def get_detailed_summary(article, keywords, style="bold"):
    key = f"detailed_{article['url']}"
    if key in st.session_state:
        return st.session_state[key]
    else:
        detailed = DETAILED_SUMMARY_TEMPLATE.format(
            title=article["title"],
            snippet=article["content"],
            url=article["url"]
        )
        resp = llm.invoke(detailed)
        text = highlight_text_safe(resp.content, keywords, style)
        st.session_state[key] = text
        return text

# === CONTENT EXTRACTION + GENERIC SUMMARY ===
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join([p for p in pages if p])
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def extract_text_from_url(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        extracted = trafilatura.extract(downloaded) if downloaded else None
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:
        pass

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join([p for p in paragraphs if p])
        return text
    except Exception as e:
        st.error(f"Failed to extract website text: {e}")
        return ""

def summarize_text(text, style="professional", max_len=6000, keywords=None, highlight_style="bold"):
    if not text or not text.strip():
        return "No content to summarize."

    clipped_text = text[:max_len]
    prompt = f"""
You are a professional summarizer.

Style: {style}
Task:
- Summarize the content below in a neutral, factual tone.
- Focus on key points, outcomes, and notable numbers if present.
- Keep it concise and readable.

Content:
{clipped_text}
"""
    resp = llm.invoke(prompt)
    return highlight_text_safe(resp.content, keywords or [], highlight_style)

# === SIDEBAR ===
with st.sidebar:
    st.header("Configure Your News Feed")
    timeframe_options = ["the last 24 hours","the last 3 days","the last 7 days","the last 14 days","the last 30 days"]
    timeframe = st.selectbox("Time Frame", timeframe_options, index=2)
    default_topic = "sustainable agriculture"
    topic = st.text_input("Topic", value=default_topic)
    geo_options = ["Vietnam, ASEAN, and Global","Vietnam only","ASEAN only","Global only"]
    geo_scope = st.selectbox("Geographical Scope", geo_options, index=0)
    search_engine = st.selectbox("Search Engine", ["google_news", "google", "bing_news"], index=0)
    language = st.selectbox("Language", ["en","vi","fr","ja","zh"], index=1)
    summary_sentences = st.slider("Number of sentences per summary", 2, 8, 4, 1)
    max_results = st.selectbox("Number of search results", [10, 20, 30, 50], index=0)
    max_articles = st.slider("Max articles in report", 5, 20, 10, 1)
    highlight_keywords = st.text_input("Highlight keywords (comma-separated)", value="Vietnam, Climate, Agriculture")
    highlight_style = st.radio("Highlight Style", ["bold","highlight"], index=0)
    summary_style = st.selectbox("Summary Style", ["professional", "bullet", "executive"], index=0)
    summary_max_len = st.slider("Summary max input chars", 1000, 20000, 6000, 500)
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    website_url = st.text_input("Website URL")
    enable_fallback = st.checkbox("Enable Fallback Mode", value=True)
    fallback_days = None
    if enable_fallback:
        fallback_days = st.slider("Fallback lookback (days)", 30, 365, 90, 30)
    summarize_source_button = st.button("Summarize Uploaded Source")
    generate_button = st.button("Generate News Report", type="primary")

# === MAIN ===
if summarize_source_button:
    st.header("Source Summary")
    keywords = [k.strip() for k in highlight_keywords.split(",")]
    source_text_blocks = []

    if uploaded_pdf is not None:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        if pdf_text:
            source_text_blocks.append(pdf_text)

    if website_url.strip():
        url_text = extract_text_from_url(website_url.strip())
        if url_text:
            source_text_blocks.append(url_text)

    if source_text_blocks:
        combined_source_text = "\n\n".join(source_text_blocks)
        source_summary = summarize_text(
            combined_source_text,
            style=summary_style,
            max_len=summary_max_len,
            keywords=keywords,
            highlight_style=highlight_style
        )
        if highlight_style == "highlight":
            st.markdown(source_summary, unsafe_allow_html=True)
        else:
            st.markdown(source_summary)
    else:
        st.warning("Please upload a PDF or enter a website URL to summarize.")

if generate_button:
    st.header(f"Top News on {topic}")
    st.caption(f"Time frame: {timeframe} | Engine: {search_engine} | Language: {language}")

    with st.spinner("Searching and compiling news..."):
        try:
            old_articles = []
            query = f"{topic} {geo_scope} news"
            articles = serpapi_search(query, engine=search_engine, lang=language, max_results=max_results)
            days_filter = {"the last 24 hours":1,"the last 3 days":3,"the last 7 days":7,"the last 14 days":14,"the last 30 days":30}.get(timeframe,0)
            fresh, _ = filter_by_date(articles, days_filter)

            keywords = [k.strip() for k in highlight_keywords.split(",")]

            if fresh:
                for i, a in enumerate(fresh[:max_articles], start=1):
                    st.subheader(f"{i}. {a['title']}")
                    st.write(f"**Date:** {a.get('normalized_date','Unknown')} | **Source:** {a.get('source','Unknown')}")
                    if a.get("thumbnail"):
                        st.image(a["thumbnail"], width=400)
                    st.write(highlight_text_safe(a["content"], keywords, highlight_style))
                    st.markdown(f"🔗 [Read more]({a['url']})")

                    with st.expander("Read more (detailed summary)"):
                        st.write(get_detailed_summary(a, keywords, highlight_style))
            else:
                st.warning("No relevant news found.")

            if enable_fallback and fallback_days:
                st.markdown("## Closest Relevant Articles (Older than requested timeframe)")
                old_articles = serpapi_search(query, engine=search_engine, lang=language, max_results=max_results)
                fresh_urls = {a['url'] for a in fresh}
                old_articles = [a for a in old_articles if a['url'] not in fresh_urls]
                if old_articles:
                    for i, a in enumerate(old_articles[:max_articles], start=1):
                        st.subheader(f"{i}. {a['title']}")
                        st.write(f"**Date:** {normalize_date(a.get('published_date'), a.get('url'))} | **Source:** {a.get('source','Unknown')}")
                        if a.get("thumbnail"):
                            st.image(a["thumbnail"], width=400)
                        st.write(highlight_text_safe(a["content"], keywords, highlight_style))
                        st.markdown(f"🔗 [Read more]({a['url']})")

                        with st.expander("Read more (detailed summary)"):
                            st.write(get_detailed_summary(a, keywords, highlight_style))

            # CSV / Excel export
            if fresh or (enable_fallback and old_articles):
                all_articles = fresh[:max_articles] + (old_articles[:max_articles] if enable_fallback else [])
                for a in all_articles:
                    a["normalized_date"] = normalize_date(a.get("published_date"), a.get("url"))
                df = pd.DataFrame(all_articles)
                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), f"news_{topic.replace(' ','_')}.csv", "text/csv")
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False, engine="openpyxl")
                st.download_button("Download Excel", excel_buffer.getvalue(), f"news_{topic.replace(' ','_')}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error: {str(e)}")

