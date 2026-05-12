from __future__ import annotations

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

from recommender.youtube import extract_video_id_from_url


def _default_api_url() -> str:
    """
    Prefer explicit env var when set; otherwise try common dev ports.

    Notes:
    - Users often change API ports (8000 vs 8010) and Streamlit caches older sessions.
    - PowerShell `set VAR=...` does NOT persist env vars; prefer `$env:VAR=...`.
    """
    env = os.environ.get("API_URL", "").strip()
    if env:
        return env.rstrip("/")

    for base in ("http://127.0.0.1:8010", "http://127.0.0.1:8000"):
        try:
            r = requests.get(f"{base}/health", timeout=0.75)
            if r.ok:
                return base
        except Exception:
            pass

    # Default aligns with README examples; sidebar can override anytime.
    return "http://127.0.0.1:8010"


def _cached_default_api_url() -> str:
    if "__default_api_url__" not in st.session_state:
        st.session_state.__default_api_url__ = _default_api_url()
    return str(st.session_state.__default_api_url__)


st.set_page_config(page_title="RecomSys v2", layout="wide")
st.title("RecomSys v2 – Deep recommender")
st.caption("Uses Synopsis (deep text embedding) + Genre (multi-hot) to rank items.")


with st.sidebar:
    st.subheader("API")
    # Bump key version to avoid inheriting stale sidebar state from older builds (default was :8000).
    if "api_url_v3" not in st.session_state:
        st.session_state.api_url_v3 = _cached_default_api_url()

    api_url = st.text_input(
        "API base URL",
        key="api_url_v3",
        help="Must match wherever `uvicorn` is running (often :8010 if :8000 is busy). "
        "In PowerShell set: `$env:API_URL=\"http://127.0.0.1:8010\"` before `streamlit run ...`.",
    )
    top_k = st.slider("Top K", min_value=1, max_value=50, value=5)

    try:
        hr = requests.get(f"{api_url}/health", timeout=2)
        if hr.ok:
            hj = hr.json()
            st.session_state["_last_health_json"] = hj
            if hj.get("youtube_quota_exceeded"):
                st.caption(
                    "Trailers: **YouTube API quota exceeded** — embeds disabled until quota resets or you "
                    "raise the daily limit in Google Cloud."
                )
            elif bool(hj.get("tmdb_api_configured")):
                st.caption("Trailers: TMDB is on (preferred).")
            elif bool(hj.get("youtube_api_configured")):
                st.caption("Trailers: YouTube Data API is on.")
            else:
                st.caption(
                    "Trailers: TMDB off — set `TMDB_API_KEY` (preferred) or `TMDB_ACCESS_TOKEN`, then restart uvicorn."
                )
    except Exception:
        st.caption("Trailers: could not reach `/health` — check the API base URL.")

    st.divider()
    mode = st.radio("Mode", ["Query (text)", "Liked items"], index=0)


def _post(path: str, payload: dict):
    r = requests.post(f"{api_url}{path}", json=payload, timeout=120)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _youtube_embed_id(item: dict) -> str:
    """Video id for iframe embed (st.video does not play youtube.com/watch URLs)."""
    vid = (item.get("youtube_video_id") or "").strip()
    if vid:
        return vid
    return extract_video_id_from_url((item.get("youtube_watch_url") or "").strip())


def _youtube_embed_url(video_id: str) -> str:
    return f"https://www.youtube-nocookie.com/embed/{video_id}?rel=0&modestbranding=1"


def _health_for_streamlit() -> dict:
    """Fresh `/health` after recommendations so quota flags stay accurate."""
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        if r.ok:
            hj = r.json()
            st.session_state["_last_health_json"] = hj
            return hj
    except Exception:
        pass
    return dict(st.session_state.get("_last_health_json") or {})


def _render_youtube_embed(video_id: str) -> None:
    """Prefer top-level `st.iframe` (Streamlit >= 1.56); nested HTML iframes often fail for YouTube."""
    url = _youtube_embed_url(video_id)
    if hasattr(st, "iframe"):
        st.iframe(url, width="stretch", height=400)
        return
    components.html(
        f'<iframe width="100%" height="400" src="{url}" title="YouTube trailer" '
        'frameborder="0" '
        'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
        "allowfullscreen></iframe>",
        height=420,
    )


def render_results(data: dict):
    results = data.get("results", [])
    if not results:
        st.warning("No results returned.")
        return

    for item in results:
        with st.container(border=True):
            c1, c2 = st.columns([0.75, 0.25])
            with c1:
                st.markdown(f"**{item['title']}**  \n`{item['imdb_id']}` • {item.get('genre','')}")
            with c2:
                st.metric("Score", f"{item['score']:.3f}")
            synopsis = (item.get("synopsis") or "").strip()
            if synopsis:
                st.write(synopsis)

            embed_id = _youtube_embed_id(item)
            search = (item.get("youtube_search_url") or "").strip()
            watch = (item.get("youtube_watch_url") or "").strip()

            if embed_id:
                # st.video() needs a direct media file — use embed URL via st.iframe (not nested components.html).
                _render_youtube_embed(embed_id)
            else:
                hj = _health_for_streamlit()
                if hj.get("youtube_quota_exceeded"):
                    st.caption(
                        "YouTube Data API **quota exceeded** — no trailer previews until the quota resets or "
                        "you increase the limit in Google Cloud. Use the search link below."
                    )
                elif hj.get("youtube_api_configured"):
                    st.caption("No embeddable trailer match from YouTube for this title. Try the search link below.")
                else:
                    st.caption(
                        "No trailer — set **`TMDB_API_KEY`** (preferred) or **`TMDB_ACCESS_TOKEN`** for uvicorn and restart the API."
                    )

            if watch and not embed_id:
                st.markdown(f"[Open trailer on YouTube]({watch})")
            if search:
                st.markdown(f"[Open YouTube search]({search})")


if mode == "Query (text)":
    query = st.text_area("What do you feel like watching?", height=120, placeholder="e.g. mind-bending sci-fi heist")
    if st.button("Recommend", type="primary", disabled=not query.strip()):
        with st.spinner("Getting recommendations..."):
            data = _post("/recommend/query", {"query": query, "top_k": int(top_k)})
        render_results(data)
else:
    st.subheader("Pick liked items (type → select)")

    if "liked_selected" not in st.session_state:
        st.session_state.liked_selected = []  # list of dicts: {title, imdb_id, genre}

    q = st.text_input("Search by title", placeholder="Start typing a movie/series name…")
    suggestions = {"results": []}
    if q.strip():
        try:
            r = requests.get(f"{api_url}/search", params={"q": q, "limit": 20}, timeout=30)
            r.raise_for_status()
            suggestions = r.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            suggestions = {"results": []}

    options = suggestions.get("results", []) if isinstance(suggestions, dict) else []
    label_to_item = {
        f"{it['title']}  ({it['imdb_id']}) — {it.get('genre','')}": it for it in options
    }
    picked_label = st.selectbox(
        "Suggestions",
        options=[""] + list(label_to_item.keys()),
        index=0,
        help="Select a title to add it to liked items.",
    )

    c_add, c_clear = st.columns([0.2, 0.2])
    with c_add:
        if st.button("Add", disabled=(picked_label == "")):
            it = label_to_item[picked_label]
            existing = {x["imdb_id"] for x in st.session_state.liked_selected}
            if it["imdb_id"] not in existing:
                st.session_state.liked_selected.append(it)
    with c_clear:
        if st.button("Clear liked"):
            st.session_state.liked_selected = []

    if st.session_state.liked_selected:
        st.markdown("**Liked list**")
        for i, it in enumerate(list(st.session_state.liked_selected)):
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.write(f"- {it['title']} (`{it['imdb_id']}`) • {it.get('genre','')}")
            with c2:
                if st.button("Remove", key=f"rm_{it['imdb_id']}"):
                    st.session_state.liked_selected.pop(i)
                    st.rerun()

    st.divider()
    st.caption("Fallback (optional): paste exact titles or IMDb IDs.")
    liked_titles_raw = st.text_area(
        "Liked Titles (one per line, exact match)",
        height=90,
        placeholder="The Shawshank Redemption\nInception",
    )
    liked_imdb_raw = st.text_area(
        "Liked IMDb IDs (one per line)",
        height=70,
        placeholder="tt0111161\ntt1375666",
    )

    lt = [x.strip() for x in liked_titles_raw.splitlines() if x.strip()]
    li = [x.strip() for x in liked_imdb_raw.splitlines() if x.strip()]
    li += [x["imdb_id"] for x in st.session_state.liked_selected]

    if st.button("Recommend", type="primary", disabled=(not lt and not li)):
        with st.spinner("Getting recommendations..."):
            data = _post("/recommend/liked", {"liked_title": lt, "liked_imdb": li, "top_k": int(top_k)})
        render_results(data)

