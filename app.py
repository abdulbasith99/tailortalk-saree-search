"""
TailorTalk — AI-powered Saree Visual Search Agent
===================================================
An AI agent that chats naturally, understands when a similarity search is
being asked, takes an image (upload or link), searches a vector index,
and returns visually similar sarees with confidence scores.

Technical Stack:
  - Vector DB: ChromaDB (cosine distance)
  - Embeddings: CLIP ViT-B/16 (sentence-transformers) with multi-view fusion
  - Agent Framework: LangChain function-calling with OpenAI GPT-4o-mini
  - Frontend: Streamlit chat interface

Usage:
    streamlit run app.py
"""

import io
import os
from pathlib import Path

import chromadb
import requests
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "saree_images"
EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-16"  # Higher-capacity CLIP
DEFAULT_TOP_K = 4
IMAGE_WIDTH = 220

# ---------------------------------------------------------------------------
# Custom CSS — premium enterprise styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* ---------- Hide Streamlit branding & chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.25rem;
    }

    .greeting-card {
        background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
        border: 1px solid #fbcfe8;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.02rem;
        color: #4a044e;
        line-height: 1.6;
    }
    .greeting-card strong {
        color: #be185d;
    }

    [data-testid="stFileUploader"] {
        border-radius: 14px;
        border: 2px dashed #e5e7eb;
        background: #fafafa;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #ec4899;
    }

    .match-label {
        text-align: center;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    .match-score {
        text-align: center;
        font-size: 0.85rem;
        color: #be185d;
        font-weight: 600;
    }

    .results-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.5rem 0 0.25rem 0;
    }
    .results-subheading {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding-top: 1.5rem;
        margin-top: 2.5rem;
        border-top: 1px solid #f3f4f6;
        letter-spacing: 0.3px;
    }
    .footer strong {
        color: #6b7280;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #f3f4f6;
    }
    .sidebar-brand {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
    .how-it-works {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.7;
    }
    .how-it-works .step {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 0.55rem;
    }
    .how-it-works .step:last-child { margin-bottom: 0; }
    .how-it-works .step-num {
        flex-shrink: 0;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #fce7f3;
        color: #be185d;
        font-size: 0.72rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 2px;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        overflow: hidden;
    }
    .stChatInput > div {
        border-radius: 12px;
    }

    /* Result card styling */
    .result-card {
        border: 1px solid #f3f4f6;
        border-radius: 12px;
        padding: 0.5rem;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.15s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
"""


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> SentenceTransformer:
    """Load the CLIP embedding model (cached)."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_chroma_collection():
    """Get the ChromaDB collection (cached)."""
    if not CHROMA_DB_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image & embedding helpers
# ---------------------------------------------------------------------------
def _extract_multi_view_embedding(model, image: Image.Image):
    """
    Extract a richer embedding by fusing multiple views of the saree image:
    full, center-crop, top-border, and bottom-border. This captures
    border/pallu work, motifs, fabric, and overall colour composition.
    """
    img = image.convert("RGB")
    width, height = img.size

    center_crop = img.crop((int(width * 0.15), int(height * 0.2),
                            int(width * 0.85), int(height * 0.8)))
    top_strip = img.crop((0, 0, width, int(height * 0.35)))
    bottom_strip = img.crop((0, int(height * 0.65), width, height))

    views = [img, center_crop, top_strip, bottom_strip]
    embeddings = [model.encode(v.convert("RGB"), convert_to_numpy=True) for v in views]

    # Normalise each view embedding, then average for fusion
    import numpy as np
    normed = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
    combined = sum(normed) / len(normed)
    combined = combined / (np.linalg.norm(combined) + 1e-8)
    return combined


def extract_embedding(model, image: Image.Image) -> list[float]:
    """Extract fused embedding from a PIL image."""
    combined = _extract_multi_view_embedding(model, image)
    return combined.tolist()


def search_similar(model, collection, query_image: Image.Image, top_k: int = DEFAULT_TOP_K):
    """Query ChromaDB for the top-k most similar sarees."""
    query_embedding = extract_embedding(model, query_image)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, 10),
    )
    return results


def load_image_from_url(url: str) -> Image.Image | None:
    """Download an image from a URL and return it as a PIL Image."""
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200 and len(resp.content) > 1000:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Agent tool — the callable tool the LLM uses
# ---------------------------------------------------------------------------
def _format_results(results, top_k: int) -> str:
    """Format ChromaDB results into a readable string for the LLM."""
    if not results or not results.get("ids"):
        return "No matches found."

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    lines = []
    for i, (img_id, dist, meta) in enumerate(zip(ids, distances, metadatas)):
        similarity = 1 - dist  # cosine distance → similarity
        name = meta.get("name", img_id)
        sku = meta.get("sku", "")
        filepath = meta.get("filepath", "")
        url = meta.get("url", "")
        lines.append(
            f"Match {i + 1}: {name} | similarity={similarity:.4f} "
            f"({similarity * 100:.1f}%) | filepath={filepath}{f' | sku={sku}' if sku else ''}"
        )
        if url:
            lines[-1] += f" | url={url}"

    return "\n".join(lines)


def display_results(results, heading="Similar Sarees Found"):
    """Display search results in a clean horizontal layout."""
    if not results or not results.get("ids"):
        st.warning("No results found.")
        return

    ids = results["ids"][0]
    if len(ids) == 0:
        st.warning("No matches found.")
        return

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    similarities = [1 - d for d in distances]

    st.markdown(f'<div class="results-heading">{heading}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="results-subheading">Top {len(ids)} matches ranked by visual similarity.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(ids))
    for i, (img_id, sim, meta) in enumerate(zip(ids, similarities, metadatas)):
        with cols[i]:
            url = meta.get("url", "")
            filepath = meta.get("filepath", "")
            name = meta.get("name", img_id)
            confidence_pct = sim * 100

            result_img = None
            if url:
                result_img = load_image_from_url(url)
            if result_img is None and filepath and Path(filepath).exists():
                # Fallback to local file if URL fails or is missing
                result_img = Image.open(filepath)

            if result_img is not None:
                st.image(result_img, width=IMAGE_WIDTH)
            else:
                st.warning("Image unavailable")

            st.markdown(
                f'<div class="match-label">Match {i + 1}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="match-score">{confidence_pct:.1f}% match</div>',
                unsafe_allow_html=True,
            )
            if name and name != img_id:
                st.caption(name[:60])

    st.markdown("---")


def visual_similarity_search(image_input: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Find visually similar sarees to the given image.

    Args:
        image_input: A local file path or a URL to an image of a saree.
        top_k: Number of similar sarees to return (default 4, max 10).

    Returns:
        A formatted string listing the closest matching sarees with their
        similarity scores and file paths.
    """
    model = load_model()
    collection = get_chroma_collection()

    if collection is None:
        return "ERROR: No indexed data found. Please run `python index_data.py` first."

    # Load the query image
    if image_input.startswith(("http://", "https://")):
        query_image = load_image_from_url(image_input)
        if query_image is None:
            return f"ERROR: Could not download image from URL: {image_input}"
    else:
        path = Path(image_input)
        if not path.exists():
            return f"ERROR: Image file not found: {image_input}"
        try:
            query_image = Image.open(path).convert("RGB")
        except Exception as e:
            return f"ERROR: Could not open image: {e}"

    # Perform search
    results = search_similar(model, collection, query_image, top_k=top_k)
    return _format_results(results, top_k)


# ---------------------------------------------------------------------------
# LangChain Agent setup (uses OpenAI function calling)
# ---------------------------------------------------------------------------
def get_agent():
    """Create a LangChain agent that can call the visual similarity search tool."""
    from langchain.agents import create_tool_calling_agent
    from langchain.agents import AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    api_key = os.environ.get("OPENAI_API_KEY", st.session_state.get("api_key", ""))
    if not api_key:
        return None

    @tool
    def search_similar_sarees(image_input: str, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Search for visually similar sarees from the catalogue.

        Args:
            image_input: Path to a local image file OR a URL to an image of a saree.
            top_k: Number of results to return (1-10, default 4).

        Returns:
            Formatted list of matching sarees with similarity scores.
        """
        return visual_similarity_search(image_input, top_k)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    # System prompt guides the agent to chat naturally while using tools correctly
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are TailorTalk, a friendly and helpful AI assistant for saree visual search.
Your job is to chat naturally with users and help them find visually similar sarees.

When a user wants to find similar sarees (by uploading an image or providing an image URL),
use the `search_similar_sarees` tool. Understand the user's intent naturally —
they might say things like "find similar", "show me similar sarees", "what goes with this",
"match this", etc.

Rules:
1. If the user provides an image (upload or URL) and asks for similar sarees, call the tool.
2. If the user is just chatting, respond conversationally without calling tools.
3. If the user asks for similar sarees but hasn't provided an image, ask them to
   upload an image or provide an image URL.
4. When the tool returns results, present them nicely — mention similarity percentages
   and confirm you've found matches.
5. If the user asks for a URL, note that they can provide any direct image URL.""")
        ,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, [search_similar_sarees], prompt)
    return AgentExecutor(agent=agent, tools=[search_similar_sarees])


# ---------------------------------------------------------------------------
# Rule-based fallback (no API key required) — understands simple patterns
# ---------------------------------------------------------------------------
def rule_based_response(user_input: str, uploaded_image_path: str | None = None) -> str:
    """Simple intent detection when no OpenAI API key is configured."""
    text = user_input.lower()
    search_intents = ["similar", "saree", "match", "find", "looks like", "search", "show me"]

    # Check if they're providing a URL
    has_url = "http://" in text or "https://" in text
    # Check for image upload
    has_image = uploaded_image_path is not None

    if has_image:
        return "PROCESS_IMAGE_SEARCH"
    elif has_url:
        return "PROCESS_URL_SEARCH"
    elif any(word in text for word in search_intents):
        return "ASK_FOR_IMAGE"
    else:
        return ("Hello! I'm TailorTalk, your saree visual search assistant. 👋\n\n"
                "Upload an image of a saree (using the uploader above) or paste an image URL, "
                "and I'll find visually similar sarees from our collection.")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TailorTalk — AI Saree Visual Search",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🧵 TailorTalk</div>', unsafe_allow_html=True)

    # OpenAI API key input (optional — falls back to rule-based if empty)
    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="Add for AI agent chat. Without it, TailorTalk uses built-in intent detection.",
    )
    if api_key:
        st.session_state["api_key"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown(
        """
        <div class="how-it-works">
            <div class="step">
                <span class="step-num">1</span>
                <span>Upload a saree image <b>or</b> paste an image URL.</span>
            </div>
            <div class="step">
                <span class="step-num">2</span>
                <span>Chat naturally — ask for "similar sarees".</span>
            </div>
            <div class="step">
                <span class="step-num">3</span>
                <span>Our AI agent searches 1000+ sarees and returns the closest visual matches.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats
    collection = get_chroma_collection()
    if collection is not None:
        count = collection.count()
        st.markdown(f"**📊 Collection:** {count:,} sarees indexed")
    else:
        st.markdown("**📊 Collection:** Not indexed — run `python index_data.py`")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🧵 TailorTalk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered Saree Visual Search Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="greeting-card">
        <strong>Hello! I'm TailorTalk assistant.</strong> 👋<br>
        Upload a saree image or paste an image URL, then ask me to find visually similar
        sarees from our catalogue of <strong>1,000+ designs</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# Image input: upload or URL
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload a saree image...",
        type=["png", "jpg", "jpeg", "webp"],
        key="uploader",
        label_visibility="collapsed",
    )

with col2:
    image_url = st.text_input(
        "Or paste an image URL...",
        placeholder="https://example.com/saree.jpg",
        label_visibility="collapsed",
    )

# Store the uploaded image to a temp file for the agent
query_image_path = None
if uploaded_file is not None:
    # Save upload to temp
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    query_image_path = temp_dir / "query_image.png"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(query_image_path, "PNG")

    # Show the query
    st.markdown('<div class="results-heading">Your Query</div>', unsafe_allow_html=True)
    st.image(img, width=200)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="results-heading">💬 Chat with TailorTalk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="results-subheading">Ask for similar sarees, or just chat.</div>',
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": 'Hello! 👋 Upload a saree image or paste a URL, '
                                         'then tell me "find similar sarees" and I\'ll match it!'}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = get_agent() if os.environ.get("OPENAI_API_KEY") else None

            if agent is not None:
                # Use the LangChain agent
                try:
                    response = agent.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.messages[:-1],
                    })
                    reply = response.get("output", "I couldn't process that.")
                except Exception as e:
                    reply = f"Sorry, I hit an error: {e}"
            else:
                # Rule-based fallback
                reply = rule_based_response(user_input, str(query_image_path) if query_image_path else None)

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # If the rule-based handler wants a search, process it directly
    if not os.environ.get("OPENAI_API_KEY"):
        if reply == "PROCESS_IMAGE_SEARCH" and query_image_path:
            with st.spinner("Searching the catalogue..."):
                model = load_model()
                collection = get_chroma_collection()
                if collection is None:
                    st.info("⚠️ No indexed data found. Run `python index_data.py` first.")
                else:
                    results = search_similar(model, collection, Image.open(query_image_path))
                    display_results(results, "✨ Similar Sarees Found")
        elif reply == "PROCESS_URL_SEARCH" and image_url:
            with st.spinner("Searching the catalogue..."):
                query_img = load_image_from_url(image_url)
                if query_img is None:
                    st.warning("Could not download image from that URL. Try a different one.")
                else:
                    model = load_model()
                    collection = get_chroma_collection()
                    if collection is not None:
                        results = search_similar(model, collection, query_img)
                        display_results(results, "✨ Similar Sarees Found")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <strong>TailorTalk</strong> — AI-powered Saree Visual Search Agent &nbsp;|&nbsp; © 2026
    </div>
    """,
    unsafe_allow_html=True,
)