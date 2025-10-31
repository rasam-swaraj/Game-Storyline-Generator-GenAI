import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint

import re

st.set_page_config(page_title="🎮 AI Game Storyline Generator", layout="wide")
st.title("🎮 AI Game Storyline Generator")
st.caption("Create characters, quests, and narratives — choose your model (OpenAI, Google, or HuggingFace)")

story_prompt = PromptTemplate.from_template("""
You are an expert AI game story writer and narrative designer.
Generate a rich, original video game storyline focused on immersive characters, interconnected quests, and an engaging overarching narrative.

User inputs:
- Genre: {genre}
- Theme/Setting: {setting}
- Tone: {tone}
- Story Length: {length}
- Target Audience: {audience}

Your response must include:

# 🎮 Game Title
## 🎬 Narrative Overview
## 🌍 Game World / Setting
## 🧙 Main Characters
## 🎯 Quests & Missions
## ⚔️ Gameplay & Story Hooks
## 🏁 Endings & Player Goals
## ✨ Tagline
""")

MODEL_OPTIONS = {
    "OpenAI — gpt-3.5-turbo": "openai",
    "Google — gemini-2.5-flash": "google",
    "HuggingFace — google/flan-t5-small": "huggingface",
}

model_choice_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=1)
model_choice = MODEL_OPTIONS[model_choice_label]

col1, col2 = st.columns(2)
with col1:
    genre = st.text_input("Game Genre", "Fantasy")
    tone = st.selectbox("Tone / Mood", ["Adventurous", "Dark", "Humorous", "Epic", "Mysterious", "Lighthearted"])
    length = st.selectbox("Story Length", ["Short", "Medium", "Long"])
with col2:
    setting = st.text_input("Theme / Setting", "A medieval kingdom")
    audience = st.selectbox("Target Audience", ["Kids", "Teen", "Adult", "All Ages"])

generate_btn = st.button("Generate Storyline")

def create_llm_for_provider(provider_key: str):
    provider_key = provider_key.lower()
    if provider_key == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Missing GOOGLE_API_KEY in .env")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    elif provider_key == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

    elif provider_key == "huggingface":
        hf_token = os.getenv("HF_API_KEY")
        if not hf_token:
            raise RuntimeError("Missing HF_API_KEY in .env")
        return HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-small",
            huggingfacehub_api_token=hf_token
        )
    else:
        raise RuntimeError("Unsupported provider selected.")

def is_hf_not_found_error(exc: Exception) -> bool:
    
    msg = str(exc).lower()
    hf_indicators = [
        "404", "not found", "model not found", "no such model", "could not find model",
        "hfhub", "huggingface", "hfhub", "hfhubhttperror"
    ]
    return any(ind in msg for ind in hf_indicators)

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    name = re.sub(r'\s+', '_', name)
    if not name:
        return "AI_Game_Storyline"
    return name[:120]
                    
def extract_title_from_story(text: str, fallback_genre: str = "AI_Game_Storyline") -> str:
    if not text:
        return fallback_genre
    lines = [ln.strip() for ln in text.splitlines()]

    for i, ln in enumerate(lines):
        if "game title" in ln.lower() or "🎮 game title" in ln.lower():
            for j in range(i+1, min(i+6, len(lines))):
                if lines[j]:
                    return sanitize_filename(lines[j])

    for ln in lines:
        if ln and not ln.startswith("#"):
            return sanitize_filename(ln[:120])

    if fallback_genre:
        return sanitize_filename(fallback_genre)
    return "AI_Game_Storyline"

story_text = None

if generate_btn:
    llm_to_use = model_choice

    with st.spinner("Generating storyline — please wait..."):
        try:
            llm = create_llm_for_provider(llm_to_use)
            story_chain = story_prompt | llm
            resp = story_chain.invoke({
                "genre": genre,
                "setting": setting,
                "tone": tone,
                "length": length.lower(),
                "audience": audience
            })
            story_text = getattr(resp, "content", str(resp))
            if not story_text.strip():
                st.error("Model returned empty output. Try different inputs or another model.")
            else:
                st.markdown("---")
                st.markdown(story_text)

        except Exception as e:
            if "insufficient_quota" in str(e).lower() and llm_to_use == "openai":
                st.warning("OpenAI quota exceeded — falling back to Google Gemini...")
                st.info(str(e))
                try:
                    llm = create_llm_for_provider("google")
                    story_chain = story_prompt | llm
                    resp = story_chain.invoke({
                        "genre": genre,
                        "setting": setting,
                        "tone": tone,
                        "length": length.lower(),
                        "audience": audience
                    })
                    story_text = getattr(resp, "content", str(resp))
                    st.markdown("---")
                    st.markdown(story_text)
                except Exception as ge:
                    st.error(f"Error during fallback to Google: {ge}")
                    with st.expander("Show error details"):
                        st.exception(ge)

            elif llm_to_use == "huggingface" and is_hf_not_found_error(e):
                try:
                    llm = create_llm_for_provider("google")
                    story_chain = story_prompt | llm
                    resp = story_chain.invoke({
                        "genre": genre,
                        "setting": setting,
                        "tone": tone,
                        "length": length.lower(),
                        "audience": audience
                    })
                    story_text = getattr(resp, "content", str(resp))
                    st.markdown("---")
                    st.markdown(story_text)
                except Exception as ge:
                    st.error(f"Error during fallback to Google: {ge}")
                    with st.expander("Show error details"):
                        st.exception(ge)
            else:
                st.error(f"Error generating storyline: {e}")
                with st.expander("Show error details"):
                    st.exception(e)
                    
if story_text and story_text.strip():
    file_title = extract_title_from_story(story_text, fallback_genre=genre or "AI_Game_Storyline")
    filename = f"{file_title}.txt"

    st.download_button(
        label="Download Storyline as .txt",
        data=story_text,
        file_name=filename,
        mime="text/plain"
    )

st.markdown("---")
st.caption("© 2025 Guru Nanak Institute of Management Studies, Mumbai. All rights reserved")
