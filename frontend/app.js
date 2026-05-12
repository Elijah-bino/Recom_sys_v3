const storageKey = "recomsys_v2_api_base_v1";
const tokenKey = "recomsys_v2_token_v1";

function $(id) {
  return document.getElementById(id);
}

function clamp(n, lo, hi) {
  n = Number(n);
  if (Number.isNaN(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function apiBase() {
  const saved = localStorage.getItem(storageKey);
  if (saved && saved.trim()) return saved.trim().replace(/\/+$/, "");

  const embedded =
    typeof window.__RECOMSYS_API_BASE__ === "string"
      ? window.__RECOMSYS_API_BASE__.trim().replace(/\/+$/, "")
      : "";
  if (embedded) return embedded;

  // If this UI is being served by the FastAPI server at /ui, prefer same-origin API calls.
  // This prevents accidentally calling an old port (e.g. :8000) that doesn't have trailers enabled.
  if (window.location && window.location.origin && window.location.pathname.startsWith("/ui")) {
    return window.location.origin.replace(/\/+$/, "");
  }

  return "http://127.0.0.1:8010";
}

function setApiBase(v) {
  localStorage.setItem(storageKey, (v || "").trim().replace(/\/+$/, ""));
}

async function getJSON(url) {
  const r = await fetch(url, { method: "GET" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return await r.json();
}

async function postJSON(url, payload) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const raw = await r.text();
    try {
      const j = JSON.parse(raw);
      const d = j.detail;
      if (typeof d === "string") throw new Error(`${r.status} ${d}`);
      if (Array.isArray(d)) throw new Error(`${r.status} ${JSON.stringify(d)}`);
      throw new Error(`${r.status} ${raw}`);
    } catch (e) {
      if (e instanceof Error && e.message.startsWith(String(r.status))) throw e;
      throw new Error(`${r.status} ${raw}`);
    }
  }
  return await r.json();
}

function getToken() {
  return (localStorage.getItem(tokenKey) || "").trim();
}
function setToken(tok) {
  if (!tok) localStorage.removeItem(tokenKey);
  else localStorage.setItem(tokenKey, tok);
}

async function authedPost(path, payload) {
  const base = apiBase();
  const tok = getToken();
  const r = await fetch(`${base}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(tok ? { Authorization: `Bearer ${tok}` } : {}),
    },
    body: JSON.stringify(payload || {}),
  });
  if (!r.ok) {
    const raw = await r.text();
    try {
      const j = JSON.parse(raw);
      const d = j.detail;
      if (typeof d === "string") throw new Error(`${r.status} ${d}`);
      if (Array.isArray(d)) throw new Error(`${r.status} ${JSON.stringify(d)}`);
      throw new Error(`${r.status} ${raw}`);
    } catch (e) {
      if (e instanceof Error && e.message.startsWith(String(r.status))) throw e;
      throw new Error(`${r.status} ${raw}`);
    }
  }
  return await r.json();
}

async function authedGet(path) {
  const base = apiBase();
  const tok = getToken();
  const r = await fetch(`${base}${path}`, {
    method: "GET",
    headers: {
      ...(tok ? { Authorization: `Bearer ${tok}` } : {}),
    },
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return await r.json();
}

function setStatus(msg, kind = "info") {
  const el = $("status");
  if (!el) return;
  el.textContent = msg || "";
  el.dataset.kind = kind;
}

function setFilterPanelOpen(open) {
  const panel = $("filterPanel");
  const btn = $("filterOpenBtn");
  if (!panel) return;
  const on = !!open;
  panel.classList.toggle("filter-panel--open", on);
  panel.setAttribute("aria-hidden", on ? "false" : "true");
  if (btn) btn.setAttribute("aria-expanded", on ? "true" : "false");
  document.body.style.overflow = on ? "hidden" : "";
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Liked list state
let liked = [];

function renderLiked() {
  const root = $("likedList");
  root.innerHTML = "";
  if (!liked.length) {
    root.innerHTML = `<div class="help">No liked items yet. Search above and click Add.</div>`;
    return;
  }
  for (const it of liked) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `
      <div class="chip__title">${escapeHtml(it.title)}</div>
      <div class="chip__meta">${escapeHtml(it.imdb_id)}</div>
      <button class="chip__x" title="Remove">✕</button>
    `;
    chip.querySelector("button").addEventListener("click", () => {
      liked = liked.filter((x) => x.imdb_id !== it.imdb_id);
      renderLiked();
    });
    root.appendChild(chip);
  }
}

function setSuggestions(items) {
  const sel = $("suggestions");
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "Select a suggestion…";
  sel.appendChild(none);

  for (const it of items) {
    const opt = document.createElement("option");
    opt.value = it.imdb_id;
    opt.textContent = `${it.title} (${it.imdb_id}) — ${it.genre || ""}`;
    opt.dataset.title = it.title;
    opt.dataset.genre = it.genre || "";
    sel.appendChild(opt);
  }
}

function renderResults(results) {
  const root = $("results");
  root.innerHTML = "";

  results.forEach((r) => {
    const div = document.createElement("div");
    div.className = "result";

    const score = Number(r.score || 0).toFixed(3);
    const title = r.title || "";
    const imdb = r.imdb_id || "";
    const genre = r.genre || "";
    const synopsis = (r.synopsis || "").trim();

    const ytId = (r.youtube_video_id || "").trim();
    const ytWatch = (r.youtube_watch_url || "").trim();
    const ytSearch = (r.youtube_search_url || "").trim();
    const embedFromUrl = ytWatch.match(
      /(?:youtube\.com\/embed\/|youtube\.com\/shorts\/|youtu\.be\/|[?&]v=)([A-Za-z0-9_-]{6,})/
    );
    const embedId = ytId || (embedFromUrl ? embedFromUrl[1] : "");

    const links = [];
    if (ytSearch) links.push(`<a class="link" target="_blank" rel="noreferrer" href="${escapeHtml(ytSearch)}">YouTube search</a>`);
    if (ytWatch) links.push(`<a class="link" target="_blank" rel="noreferrer" href="${escapeHtml(ytWatch)}">Open trailer</a>`);
    links.push(`<a class="link" target="_blank" rel="noreferrer" href="https://www.imdb.com/title/${escapeHtml(imdb)}/">IMDb</a>`);

    // Do not pass origin= here — it often breaks embeds on localhost / mixed setups (YouTube error 153).
    const embedSrc = embedId
      ? `https://www.youtube-nocookie.com/embed/${encodeURIComponent(embedId)}?rel=0&modestbranding=1`
      : "";

    div.innerHTML = `
      <div class="result__top">
        <div class="result__title">
          <h3>${escapeHtml(title)}</h3>
          <div class="badge">${escapeHtml(score)}</div>
        </div>
        <div class="meta">${escapeHtml(imdb)} • ${escapeHtml(genre)}</div>
        <div class="syn">${escapeHtml(synopsis)}</div>
        <div class="actions">${links.join("")}</div>
      </div>
      <div class="video">
        ${
          embedId
            ? `<iframe src="${escapeHtml(embedSrc)}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy"></iframe>`
            : `<div style="padding:12px;color:rgba(255,255,255,.72);font-size:13px">${escapeHtml(noTrailerHint())}</div>`
        }
      </div>
    `;

    root.appendChild(div);
  });
}

async function healthCheck(base) {
  const h = await getJSON(`${base}/health`);
  window.__lastHealth = h;
  return h && h.ok;
}

function fmtHealth(h) {
  const tmdb = h && h.tmdb_api_configured ? "tmdb: on" : "tmdb: off";
  const yt = h && h.youtube_api_configured ? "yt: on" : "yt: off";
  if (h && h.youtube_quota_exceeded) return `${tmdb} • ${yt} • quota exceeded`;
  return `${tmdb} • ${yt}`;
}

function noTrailerHint() {
  const h = window.__lastHealth || {};
  if (h.youtube_quota_exceeded) {
    return "YouTube Data API quota is exceeded — trailers are disabled until quota resets or you raise the limit in Google Cloud. Use the YouTube search link below.";
  }
  if (h.tmdb_api_configured) {
    return "No trailer returned from TMDB for this title. Try the YouTube search link below.";
  }
  if (h.youtube_api_configured) {
    return "No embeddable trailer match from YouTube for this title. Try the search link below.";
  }
  return "No trailer picked. Configure TMDB_API_KEY on the API (preferred) or use the YouTube search link above.";
}

function currentFilters() {
  const kind = ($("filterKind").value || "").trim();
  const cert = ($("filterCert").value || "").trim();
  const yearMin = ($("filterYearMin").value || "").trim();
  const yearMax = ($("filterYearMax").value || "").trim();
  const rtMin = ($("filterRuntimeMin").value || "").trim();
  const rtMax = ($("filterRuntimeMax").value || "").trim();
  const genres = Array.from(document.querySelectorAll("#filterGenres input[type=checkbox]:checked")).map((x) => x.value);
  const f = {};
  if (kind) f.kind = kind;
  if (cert) f.certificate = cert;
  if (yearMin) f.year_min = Number(yearMin);
  if (yearMax) f.year_max = Number(yearMax);
  if (rtMin) f.runtime_min = Number(rtMin);
  if (rtMax) f.runtime_max = Number(rtMax);
  if (genres.length) f.genres = genres;
  return Object.keys(f).length ? f : null;
}

function showShare(slug) {
  const bar = $("shareBar");
  if (!slug) {
    bar.style.display = "none";
    bar.innerHTML = "";
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("share", slug);
  bar.style.display = "flex";
  bar.innerHTML = `
    <div>Share this list:</div>
    <div class="mono" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%">${escapeHtml(url.toString())}</div>
    <button id="copyShare" class="btn btn--ghost btn--sm">Copy</button>
  `;
  bar.querySelector("#copyShare").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(url.toString());
      setStatus("Copied share link.", "good");
    } catch {
      setStatus("Could not copy (browser blocked clipboard).", "warn");
    }
  });
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

async function searchTitles(q) {
  const base = apiBase();
  const url = new URL(`${base}/search`);
  url.searchParams.set("q", q);
  url.searchParams.set("limit", "20");
  return await getJSON(url.toString());
}

async function recommendLiked() {
  const base = apiBase();
  const topK = clamp($("topKLiked").value, 1, 50);
  const payload = { liked_imdb: liked.map((x) => x.imdb_id), liked_title: [], top_k: topK, filters: currentFilters() };
  const tok = getToken();
  if (tok) return await authedPost("/recommend/liked", payload);
  return await postJSON(`${base}/recommend/liked`, payload);
}

async function recommendQuery() {
  const base = apiBase();
  const topK = clamp($("topKQuery").value, 1, 50);
  const query = ($("queryInput").value || "").trim();
  const payload = { query, top_k: topK, filters: currentFilters() };
  const tok = getToken();
  if (tok) return await authedPost("/recommend/query", payload);
  return await postJSON(`${base}/recommend/query`, payload);
}

async function loadFilterOptions() {
  const base = apiBase();
  try {
    const opt = await getJSON(`${base}/filters/options`);
    // certs
    const certSel = $("filterCert");
    for (const c of opt.certificates || []) {
      const o = document.createElement("option");
      o.value = c;
      o.textContent = c;
      certSel.appendChild(o);
    }
    // genres
    const gRoot = $("filterGenres");
    gRoot.innerHTML = "";
    for (const g of opt.genres || []) {
      const lab = document.createElement("label");
      lab.className = "gcheck";
      lab.innerHTML = `<input type="checkbox" value="${escapeHtml(g)}" /> <span>${escapeHtml(g)}</span>`;
      gRoot.appendChild(lab);
    }
  } catch (e) {
    // non-fatal
  }
}

async function syncMe() {
  const tok = getToken();
  const btn = $("authBtn");
  if (!tok) {
    btn.textContent = "Sign in";
    return null;
  }
  try {
    const me = await authedGet("/me");
    btn.textContent = me.email || "Account";
    return me;
  } catch {
    setToken("");
    btn.textContent = "Sign in";
    return null;
  }
}

function openAuthModal() {
  const modal = document.createElement("div");
  modal.className = "modal modal--open";
  modal.innerHTML = `
    <div class="modal__panel">
      <div class="modal__head">
        <h2>Account</h2>
        <button class="xbtn" id="closeAuth">Close</button>
      </div>
      <div class="field">
        <label>Email</label>
        <input id="authEmail" class="input" placeholder="you@example.com" />
      </div>
      <div class="field">
        <label>Password</label>
        <input id="authPw" class="input" type="password" placeholder="••••••••" />
      </div>
      <div class="row row--tight">
        <button id="loginBtn" class="btn btn--primary">Login</button>
        <button id="registerBtn" class="btn">Register</button>
        <button id="logoutBtn" class="btn btn--ghost">Logout</button>
      </div>
      <div class="help" id="authMsg"></div>
    </div>
  `;
  document.body.appendChild(modal);

  const close = () => modal.remove();
  modal.querySelector("#closeAuth").addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });

  const msg = modal.querySelector("#authMsg");
  const setMsg = (t) => (msg.textContent = t || "");

  modal.querySelector("#loginBtn").addEventListener("click", async () => {
    const email = modal.querySelector("#authEmail").value.trim();
    const password = modal.querySelector("#authPw").value.trim();
    try {
      const res = await postJSON(`${apiBase()}/auth/login`, { email, password });
      setToken(res.token);
      await syncMe();
      setMsg("Logged in.");
      close();
    } catch (e) {
      setMsg(`Login failed: ${e.message}`);
    }
  });

  modal.querySelector("#registerBtn").addEventListener("click", async () => {
    const email = modal.querySelector("#authEmail").value.trim();
    const password = modal.querySelector("#authPw").value.trim();
    try {
      const res = await postJSON(`${apiBase()}/auth/register`, { email, password });
      setToken(res.token);
      await syncMe();
      setMsg("Registered and logged in.");
      close();
    } catch (e) {
      setMsg(`Register failed: ${e.message}`);
    }
  });

  modal.querySelector("#logoutBtn").addEventListener("click", async () => {
    setToken("");
    await syncMe();
    setMsg("Logged out.");
    close();
  });
}

async function loadShareFromUrl() {
  const url = new URL(window.location.href);
  const slug = (url.searchParams.get("share") || "").trim();
  if (!slug) return;
  try {
    const data = await getJSON(`${apiBase()}/share/${encodeURIComponent(slug)}`);
    try {
      await healthCheck(apiBase());
    } catch (_) {
      /* ignore */
    }
    renderResults(data.results || []);
    showShare(slug);
    setStatus("Loaded shared recommendations.", "good");
  } catch (e) {
    setStatus(`Could not load share: ${e.message}`, "bad");
  }
}

async function init() {
  // API bar
  $("apiBase").value = apiBase();
  $("saveApi").addEventListener("click", async () => {
    const v = ($("apiBase").value || "").trim();
    if (!v) return;
    setApiBase(v);
    setStatus("Saved API base. Checking…");
    try {
      await healthCheck(apiBase());
      const h = window.__lastHealth || {};
      setStatus(`API reachable (${apiBase()}) • ${fmtHealth(h)}`, "good");
    } catch (e) {
      setStatus(`API not reachable: ${e.message}`, "bad");
    }
  });

  renderLiked();
  setSuggestions([]);
  showShare(null);

  // Initial health check
  try {
    await healthCheck(apiBase());
    const h = window.__lastHealth || {};
    setStatus(`API reachable (${apiBase()}) • ${fmtHealth(h)}`, "good");
  } catch (e) {
    setStatus(`API not reachable: ${e.message}`, "bad");
  }

  await loadFilterOptions();
  await syncMe();
  $("authBtn").addEventListener("click", openAuthModal);
  await loadShareFromUrl();

  const openFlt = $("filterOpenBtn");
  const closeFlt = $("filterCloseBtn");
  const backFlt = $("filterBackdrop");
  if (openFlt) openFlt.addEventListener("click", () => setFilterPanelOpen(true));
  if (closeFlt) closeFlt.addEventListener("click", () => setFilterPanelOpen(false));
  if (backFlt) backFlt.addEventListener("click", () => setFilterPanelOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const p = $("filterPanel");
    if (p && p.classList.contains("filter-panel--open")) setFilterPanelOpen(false);
  });

  // Search
  const doSearch = debounce(async () => {
    const q = ($("searchInput").value || "").trim();
    if (!q) {
      $("searchHelp").textContent = "";
      setSuggestions([]);
      return;
    }
    $("searchHelp").textContent = "Searching…";
    try {
      const res = await searchTitles(q);
      setSuggestions(res.results || []);
      $("searchHelp").textContent = `${(res.results || []).length} suggestion(s).`;
    } catch (e) {
      $("searchHelp").textContent = `Search failed: ${e.message}`;
      setSuggestions([]);
    }
  }, 220);
  $("searchInput").addEventListener("input", doSearch);

  // Add liked
  $("addLiked").addEventListener("click", () => {
    const sel = $("suggestions");
    const opt = sel.options[sel.selectedIndex];
    const imdb = (opt && opt.value) || "";
    if (!imdb) return;
    const title = opt.dataset.title || opt.textContent || imdb;
    const genre = opt.dataset.genre || "";
    if (liked.some((x) => x.imdb_id === imdb)) return;
    liked.push({ imdb_id: imdb, title, genre });
    renderLiked();
  });
  $("clearLiked").addEventListener("click", () => {
    liked = [];
    renderLiked();
  });

  // Recommend liked
  $("recommendLiked").addEventListener("click", async () => {
    if (!liked.length) return;
    setStatus("Recommending…");
    try {
      const data = await recommendLiked();
      try {
        await healthCheck(apiBase());
      } catch (_) {
        /* ignore */
      }
      renderResults(data.results || []);
      showShare(data.share_slug || null);
      setStatus(`Done. Returned ${ (data.results || []).length } results.`, "good");
    } catch (e) {
      setStatus(`Recommend failed: ${e.message}`, "bad");
    }
  });

  // Recommend query
  $("recommendQuery").addEventListener("click", async () => {
    const q = ($("queryInput").value || "").trim();
    if (!q) return;
    setStatus("Recommending…");
    try {
      const data = await recommendQuery();
      try {
        await healthCheck(apiBase());
      } catch (_) {
        /* ignore */
      }
      renderResults(data.results || []);
      showShare(data.share_slug || null);
      setStatus(`Done. Returned ${ (data.results || []).length } results.`, "good");
    } catch (e) {
      setStatus(`Recommend failed: ${e.message}`, "bad");
    }
  });
}

init();

