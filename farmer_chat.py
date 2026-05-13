"""
Farmer-friendly chat UI for the Amharic RAG (Streamlit).

Run from project root (``.env`` for Groq/Gemini keys, or local Ollama):
  cd /home/lenovo/Desktop/RAG && . .venv/bin/activate && streamlit run farmer_chat.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from llm_providers import effective_llm_backend, load_dotenv_if_present
from query import default_top_k, rag_runtime_env, sanitize_chat_answer, stream_rag_answer

load_dotenv_if_present()

ROOT = Path(__file__).resolve().parent
DB = ROOT / "chroma_amharic"

st.set_page_config(
    page_title="የግብርና ረዳት",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { max-width: 760px; padding-top: 1.2rem; }
    [data-testid="stChatMessage"] { font-size: 1.05rem; line-height: 1.55; }
    .stCaption { font-size: 0.95rem; opacity: 0.9; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        content = msg.get("content") or ""
        if msg["role"] == "assistant" and (
            content.startswith("[Ollama]")
            or content.startswith("[groq]")
            or content.startswith("[gemini]")
            or content.startswith("[openai]")
        ):
            st.error(content)
        else:
            st.markdown(content)
        if msg.get("sources"):
            with st.expander("📎 ምንጮች (ለማረጋገጫ ይክፈቱ)"):
                for i, s in enumerate(msg["sources"], 1):
                    src = s.get("source", "?")
                    kind = s.get("kind", "")
                    page = s.get("page") or "—"
                    st.markdown(f"**{i}.** `{src}` · {kind} · ገጽ {page}")
                    prev = (s.get("preview") or "").strip()
                    if prev:
                        st.caption(prev)


with st.sidebar:
    st.header("⚙️ ማስተካከያ")
    st.caption(
        f"LLM: **{effective_llm_backend()}** · `RAG_LLM_BACKEND` (auto፡ Groq→Gemini፤ ከ429/503 በኋላ **Ollama** ይሞክራል ከተጫነ። "
        "`RAG_LOCAL_FIRST=1` ከሆነ በመጀመሪያ **Ollama**። Groq 429 → Gemini → (ስህተት) Ollama።"
    )
    quality = st.toggle(
        "ጥራት ቅድሚያ (የበለጠ ዝርዝር መልስ፣ ቀርጥፍ ይሆናል)",
        value=False,
        help="ከፍተኛ ጥራት ለመጠቀም ከበስተኛው ይመርጡ።",
    )
    fast = not quality
    top_k = st.slider(
        "የማመሳከሪያ ቁጥር",
        min_value=2,
        max_value=8,
        value=default_top_k(fast),
        help="በብዛት ትክክለኛ መረጃ ለማምጣት ቁጥሩን ይጨምሩ፤ ለፍጥነት ይቀንሱ።",
    )
    st.divider()
    tools_on = st.toggle(
        "ድር / መሳሪያ (RAG_TOOLS)",
        value=False,
        help="ከበስተኛው በሚከፈትበት ጊዜ RAG_WEB_MODE ይተገበራል። !web … እና !weather … ሁልጊዜ ይሰራሉ።",
    )
    web_mode = st.selectbox(
        "RAG_WEB_MODE",
        ("off", "auto", "always", "if_kb_sparse"),
        index=0,
        help="auto: RAG_WEB_AUTO ወይም regex፤ if_kb_sparse: በChroma ርቀት/ቁጥር።",
    )
    st.divider()
    st.markdown(
        "**ማስታወሻ**\n\n"
        "• ዋናው መረጃ ከመመሪያ እና ከጥያቄ-መልስ ቤት ነው።\n\n"
        "• መልሶች **ምክር** እና **የወደፊት ግምት** ሊያካትቱ ይችላሉ፤ ከመመሪያው ሲደገፉ ብቻ በጥንቃቄ ይሰጣሉ።\n\n"
        "• `!web ፍለጋ` ወይም `!weather Addis Ababa` ለውጫዊ እውቀት።\n\n"
        "• ለመርዝ፣ ለዕጣዕጅ፣ ለእንስሳ ቀውስ ወይም ለሕጋዊ/ገንዘብ ውሳኔ ከአካባቢ ማራዝሚያ ወይም ባለሙያ ያማካኙ።\n\n"
        "• **HTTP 429** የየኔት ወሰን/ኮታ ነው (428 አይደለም)። ታሪክ + መመሪያ ቶከን ያቃጣል — `RAG_HOSTED_CHAT_ROUNDS` ነባሪ **3** ዙር ብቻ ወደ API ይላካል፤ ለረጅም ውይይት ቁጥሩን ይጨምሩ (`0` = ሁሉንም)።\n\n"
        "• ከAPI ውጭ፡ `RAG_LLM_BACKEND=ollama` ወይም `RAG_LOCAL_FIRST=1`። Groq/Gemini ከወሰኑ በኋላ **Ollama** አውቶማቲክ ይሞክራል (`RAG_HOSTED_FALLBACK_OLLAMA=1`)።\n\n"
        "• ለአካባቢ **Ollama** `qwen2.5:3b` / `qwen3:4b-instruct` ይጫኑ።"
    )
    if st.button("ውይይት አጽዳ", type="secondary"):
        st.session_state.messages = []
        st.rerun()

st.title("🌾 የግብርና ረዳት")
st.caption(
    "ስለ ሰብል፣ መሬት፣ ዝናብ፣ ድህረ-ምርት እና ተዛማጅ ጥያቄዎች በአማርኛ ይጠይቁ። "
    "መልሶች በመጀመሪያ ከመመሪያ ቤታችን ይደገፋሉ፤ **ምክር፣ የሚከናወኑ እርምጃዎች** እና አስፈላጊ ሲሆን **በጥንቃቄ የተገመተ ውጤት** "
    "ይሰጣሉ። ከበስተኛው ድር/መሳሪያ ሲከፈት ተጨማሪ ምንጮች ይጨመራሉ።"
)

for m in st.session_state.messages:
    _render_message(m)

if not st.session_state.messages:
    st.info(
        "ምሳሌ ጥያቄዎች፦ **ስንዴ ለማምረት ምን ያህል ዝናብ ያስፈልጋል?** "
        "**በዚህ ወቅት ምን ማድረግ ይመራል?** ወይም **የተፈጥሮ ማዳበሪያ ምን ይጠቅማል?** ከታች ይጻፉ።"
    )

prompt = st.chat_input("ጥያቄዎን እዚህ ይጻፉ…")
if prompt:
    prior = [
        {"role": m["role"], "content": (m.get("content") or "").strip()}
        for m in st.session_state.messages
        if (m.get("content") or "").strip()
    ]
    env_kw: dict[str, str | None] = {}
    if tools_on:
        env_kw["RAG_TOOLS"] = "1"
        env_kw["RAG_WEB_MODE"] = web_mode
    with rag_runtime_env(**env_kw):
        with st.spinner("በመፈለግ ላይ… እባክዎ ትንሽ ይጠብቁ።"):
            sources, _retrieval, gen_fn = stream_rag_answer(
                prompt,
                DB,
                top_k,
                fast=fast,
                conversation=prior,
            )
    st.session_state.messages.append({"role": "user", "content": prompt})
    chunks: list[str] = []
    with st.chat_message("assistant"):
        def tracked():
            for t in gen_fn():
                chunks.append(t)
                yield t

        streamed = st.write_stream(tracked())
    full = sanitize_chat_answer((streamed or "").strip() or "".join(chunks).strip())
    st.session_state.messages.append(
        {"role": "assistant", "content": (full or "").strip(), "sources": sources}
    )
    st.rerun()
