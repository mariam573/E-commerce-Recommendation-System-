/* main.js – AIE425Shop shared utilities */

/* ── User panel toggle ─────────────────────────────────────────────────────── */
function toggleUserPanel() {
  const panel = document.getElementById('user-panel');
  if (panel) panel.classList.toggle('open');
}

// Close panel when clicking outside
document.addEventListener('click', function(e) {
  const wrap = document.getElementById('user-selector-wrap');
  if (wrap && !wrap.contains(e.target)) {
    const panel = document.getElementById('user-panel');
    if (panel) panel.classList.remove('open');
  }
});

/* ── Search category select ────────────────────────────────────────────────── */
function applyCategorySearch(sel) {
  const cat = sel.value;
  const url = new URL(window.location.href);
  if (cat) url.searchParams.set('category', cat);
  else url.searchParams.delete('category');
  url.searchParams.delete('page');
  window.location.href = url.toString();
}

/* ── Sort select ───────────────────────────────────────────────────────────── */
function applySort(val) {
  const url = new URL(window.location.href);
  if (val) url.searchParams.set('sort', val);
  else url.searchParams.delete('sort');
  url.searchParams.delete('page');
  window.location.href = url.toString();
}

/* ── Card hover image zoom effect is handled by CSS ──────────────────────── */

/* ── Lazy image error fallback ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('img.card-img, img.pd-img').forEach(function(img) {
    img.addEventListener('error', function() {
      this.src = '/static/images/placeholder.jpg';
      this.onerror = null;
    });
  });
});
