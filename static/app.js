/**
 * DEVCOPILOT — Interactive Client Controller
 * Handles conversation state, markdown parsing, code copy, and API requests.
 */

document.addEventListener("DOMContentLoaded", () => {
  const chatThread = document.getElementById("chat-thread");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const sidebarPanel = document.getElementById("sidebar-panel");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const promptChips = document.querySelectorAll(".prompt-chip");

  // Mobile sidebar drawer toggle
  if (toggleSidebarBtn && sidebarPanel) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebarPanel.classList.toggle("sidebar-open");
    });

    // Close when clicking outside on mobile
    document.addEventListener("click", (e) => {
      if (
        window.innerWidth <= 992 &&
        sidebarPanel.classList.contains("sidebar-open") &&
        !sidebarPanel.contains(e.target) &&
        !toggleSidebarBtn.contains(e.target)
      ) {
        sidebarPanel.classList.remove("sidebar-open");
      }
    });
  }

  // Starter prompt chips
  promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query) {
        userInput.value = query;
        if (window.innerWidth <= 992 && sidebarPanel) {
          sidebarPanel.classList.remove("sidebar-open");
        }
        submitQuery(query);
      }
    });
  });

  // Chat Form Submission
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;
    submitQuery(query);
  });

  async function submitQuery(question) {
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;

    // 1. Append User Message
    appendUserMessage(question);

    // 2. Append Loading Placeholder
    const loadingCardId = "loading-" + Date.now();
    appendLoadingIndicator(loadingCardId);
    scrollToBottom();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${response.status}`);
      }

      const data = await response.json();
      removeElement(loadingCardId);

      // 3. Append Assistant Response with Telemetry
      appendAssistantMessage(data);
    } catch (err) {
      removeElement(loadingCardId);
      appendErrorMessage(err.message);
    } finally {
      userInput.disabled = false;
      sendBtn.disabled = false;
      userInput.focus();
      scrollToBottom();
    }
  }

  function appendUserMessage(text) {
    const article = document.createElement("article");
    article.className = "thread-item user-item";
    article.innerHTML = `
      <div class="neo-card conversation-box user-box">
        <div class="interrupt-tag tag-user">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
            <circle cx="12" cy="7" r="4"></circle>
          </svg>
          <span>Developer</span>
        </div>
        <div class="user-text">${escapeHtml(text)}</div>
      </div>
    `;
    chatThread.appendChild(article);
  }

  function appendLoadingIndicator(id) {
    const article = document.createElement("article");
    article.className = "thread-item assistant-item";
    article.id = id;
    article.innerHTML = `
      <div class="neo-card conversation-box assistant-box">
        <div class="interrupt-tag tag-assistant">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          <span>Pipeline Executing</span>
        </div>
        <div class="message-content">
          <p>Evaluating query across guardrails, semantic vectors, and evaluation scoring...</p>
        </div>
      </div>
    `;
    chatThread.appendChild(article);
  }

  function appendAssistantMessage(data) {
    const article = document.createElement("article");
    article.className = "thread-item assistant-item";

    const routeLabel = formatRouteLabel(data.source);
    const tagClass = getRouteTagClass(data.source);

    let metaHtml = "";
    if (data.source === "capability") {
      metaHtml = `
        <div class="meta-status-card">
          <div class="meta-status-header">
            <span class="meta-route-tag">
              <strong>Scope &amp; Intent:</strong> Engine Capabilities &amp; Stacks
            </span>
            <div class="meta-score-row">
              <span class="meta-score-pill">
                Status: <strong>Supported</strong>
              </span>
              <span class="meta-score-pill">
                Time: <strong>${data.duration}s</strong>
              </span>
            </div>
          </div>
        </div>
      `;
    } else if (data.source !== "guardrail") {
      metaHtml = `
        <div class="meta-status-card">
          <div class="meta-status-header">
            <span class="meta-route-tag">
              <strong>Route:</strong> ${routeLabel}
            </span>
            <div class="meta-score-row">
              ${data.kb_score !== null && data.kb_score !== undefined ? `
                <span class="meta-score-pill">
                  KB Score: <strong>${data.kb_score}/10</strong> ${data.kb_score >= 7 ? "[Direct Match]" : "[Fallback]"}
                </span>
              ` : ""}
              ${data.web_score !== null && data.web_score !== undefined ? `
                <span class="meta-score-pill">
                  Web Score: <strong>${data.web_score}/10</strong> ${data.web_score >= 6 ? "[Verified]" : "[Deep Search]"}
                </span>
              ` : ""}
              <span class="meta-score-pill">
                Time: <strong>${data.duration}s</strong>
              </span>
            </div>
          </div>
          ${data.kb_score_detail ? `
            <div class="meta-detail-box">${escapeHtml(data.kb_score_detail)}</div>
          ` : ""}
          ${data.web_score_detail ? `
            <div class="meta-detail-box">${escapeHtml(data.web_score_detail)}</div>
          ` : ""}
        </div>
      `;
    }

    const parsedAnswer = parseMarkdown(data.answer);

    article.innerHTML = `
      <div class="neo-card conversation-box assistant-box">
        <div class="interrupt-tag ${tagClass}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
          </svg>
          <span>${routeLabel} (${data.duration}s)</span>
        </div>

        ${metaHtml}

        <div class="message-content">
          ${parsedAnswer}
        </div>
      </div>
    `;

    chatThread.appendChild(article);
    attachCodeCopyListeners(article);
  }

  function appendErrorMessage(msg) {
    const article = document.createElement("article");
    article.className = "thread-item assistant-item";
    article.innerHTML = `
      <div class="neo-card conversation-box assistant-box">
        <div class="interrupt-tag tag-blocked">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>Error</span>
        </div>
        <div class="message-content">
          <p><strong>Pipeline Execution Failure:</strong> ${escapeHtml(msg)}</p>
        </div>
      </div>
    `;
    chatThread.appendChild(article);
  }

  function formatRouteLabel(src) {
    switch (src) {
      case "guardrail": return "Scope Guardrail";
      case "capability": return "Assistant Capabilities & Scope";
      case "knowledge_base": return "Knowledge Base Direct";
      case "kb_and_web": return "Knowledge Base + Web Research";
      case "deep_research": return "Deep Web Research";
      default: return "Enterprise Copilot";
    }
  }

  function getRouteTagClass(src) {
    switch (src) {
      case "guardrail": return "tag-blocked";
      case "capability": return "tag-route";
      case "knowledge_base": return "tag-route";
      case "kb_and_web": return "tag-web";
      case "deep_research": return "tag-web";
      default: return "tag-assistant";
    }
  }

  function scrollToBottom() {
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Robust URL Auto-Linker
  function autolinkUrls(text) {
    if (!text) return "";

    // 1. Markdown formatted links: [Label](https://...)
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, (match, label, url) => {
      const cleanUrl = url.replace(/[.,;)]+$/, "");
      return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" class="source-link">${label} &nearr;</a>`;
    });

    // 2. Angle-bracket formatted links: <https://...> or &lt;https://...&gt;
    text = text.replace(/(?:&lt;|<)(https?:\/\/[^\s&>]+)(?:&gt;|>)/g, (match, url) => {
      const cleanUrl = url.replace(/[.,;)]+$/, "");
      return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" class="source-link">${cleanUrl} &nearr;</a>`;
    });

    // 3. Bare URLs outside of HTML tags and anchor links
    const parts = text.split(/(<a\b[^>]*>.*?<\/a>|<code\b[^>]*>.*?<\/code>)/gs);
    for (let i = 0; i < parts.length; i += 2) {
      if (parts[i]) {
        parts[i] = parts[i].replace(/\b(https?:\/\/[^\s<>'")\]]+)/g, (match, rawUrl) => {
          const cleanUrl = rawUrl.replace(/[.,;:)]+$/, "");
          const trailing = rawUrl.slice(cleanUrl.length);
          return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" class="source-link">${cleanUrl} &nearr;</a>${trailing}`;
        });
      }
    }
    return parts.join("");
  }

  // Lightweight Markdown & Code Block Parser
  function parseMarkdown(md) {
    if (!md) return "";

    // 1. Code blocks with language header
    md = md.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || "code";
      return `
        <div class="code-block-container">
          <div class="code-block-header">
            <span>${language}</span>
            <button class="copy-code-btn" type="button">Copy</button>
          </div>
          <pre><code>${escapeHtml(code.trim())}</code></pre>
        </div>
      `;
    });

    // 2. Tables (with autolinking in every cell and responsive container)
    const tableRegex = /(?:^|\n)((?:[ \t]*\|[^\n]+\|\s*\n)(?:[ \t]*\|[-:\s|]+?\|\s*\n)(?:[ \t]*\|[^\n]+\|\s*(?:\n|$))+)/g;
    md = md.replace(tableRegex, (match, tableText) => {
      const lines = tableText.trim().split("\n").map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length < 2) return match;
      const headers = lines[0].replace(/^\||\|$/g, "").split("|").map(h => h.trim());
      const rows = lines.slice(2).map(r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));

      let tableHtml = '<div class="table-container"><table><thead><tr>';
      headers.forEach(h => { tableHtml += `<th>${formatInlineMarkdown(h)}</th>`; });
      tableHtml += "</tr></thead><tbody>";
      rows.forEach(r => {
        tableHtml += "<tr>";
        r.forEach(c => { tableHtml += `<td>${formatInlineMarkdown(c)}</td>`; });
        tableHtml += "</tr>";
      });
      tableHtml += "</tbody></table></div>";
      return "\n\n" + tableHtml + "\n\n";
    });

    // 3. Dedicated Sources / References Card Section
    md = md.replace(/(?:^|\n)(?:[-*_]{3,}\s*)?(?:#{1,4}\s*|\*{2})?(?:Sources?|References?)(?:\*{2})?:?\s*\n([\s\S]*?)(?=(?:\n#{1,4}\s|\n[-*_]{3,}\s*$|$))/gi, (match, items) => {
      const lines = items.trim().split("\n").filter(l => l.trim().length > 0);
      let listHtml = `<div class="sources-card"><div class="sources-header"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg><span>Sources &amp; Citations</span></div><ul class="sources-list">`;
      lines.forEach((line, idx) => {
        const clean = line.replace(/^\d+[\.\)]\s*/, "").replace(/^[-*]\s*/, "").trim();
        const urlMatch = clean.match(/(https?:\/\/[^\s<>'")\]]+)/);
        if (urlMatch) {
          const rawUrl = urlMatch[1].replace(/[.,;:)>\]]+$/, "");
          const desc = clean.replace(/<[^>]+>|https?:\/\/[^\s<>'")\]]+/g, "").replace(/^[–\-—:]\s*/, "").replace(/[–\-—:]\s*$/, "").trim() || rawUrl;
          listHtml += `
            <li class="source-item">
              <span class="source-tag">Source ${idx + 1}</span>
              <a href="${rawUrl}" target="_blank" rel="noopener noreferrer" class="source-link">${escapeHtml(desc)} &nearr;</a>
            </li>
          `;
        } else if (clean) {
          listHtml += `
            <li class="source-item">
              <span class="source-tag">Ref</span>
              <span>${autolinkUrls(escapeHtml(clean))}</span>
            </li>
          `;
        }
      });
      listHtml += `</ul></div>`;
      return "\n\n" + listHtml + "\n\n";
    });

    // 4. Headers
    md = md.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    md = md.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    md = md.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // 5. Unordered Lists
    md = md.replace(/^\s*[-*]\s+(.*$)/gim, "<li>$1</li>");
    md = md.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");
    md = md.replace(/<\/ul>\s*<ul>/g, "");

    // 6. Inline formatters (Bold, inline code, italics)
    md = md.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    md = md.replace(/`([^`]+)`/g, "<code>$1</code>");
    md = md.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // 7. In-text citation badges like [KB Source 1] or [Web 1]
    md = md.replace(/\[(KB Source \d+|Web Source \d+|Source \d+)\]/gi, '<span class="source-tag">$1</span>');

    // 8. Auto-link any remaining URLs in prose
    md = autolinkUrls(md);

    // 9. Paragraphs
    const paras = md.split(/\n{2,}/);
    md = paras.map(p => {
      p = p.trim();
      if (!p) return "";
      if (
        p.startsWith("<div") ||
        p.startsWith("<table") ||
        p.startsWith("<h") ||
        p.startsWith("<ul") ||
        p.startsWith("<ol")
      ) {
        return p;
      }
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    }).join("\n");

    return md;
  }

  function formatInlineMarkdown(text) {
    text = escapeHtml(text);
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
    text = autolinkUrls(text);
    return text;
  }

  function attachCodeCopyListeners(scope) {
    const copyBtns = scope.querySelectorAll(".copy-code-btn");
    copyBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const pre = btn.closest(".code-block-container").querySelector("pre code");
        if (pre) {
          navigator.clipboard.writeText(pre.innerText).then(() => {
            const originalText = btn.innerText;
            btn.innerText = "Copied!";
            setTimeout(() => { btn.innerText = originalText; }, 1800);
          });
        }
      });
    });
  }
});
