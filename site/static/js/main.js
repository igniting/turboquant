// Keyboard navigation for slides
document.addEventListener('keydown', function(e) {
  // Don't capture if user is typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  var prevBtn = document.querySelector('.nav-prev');
  var nextBtn = document.querySelector('.nav-next');
  var startBtn = document.querySelector('.start-btn');

  switch (e.key) {
    case 'ArrowRight':
    case 'l':
    case ' ':
      e.preventDefault();
      if (nextBtn) nextBtn.click();
      else if (startBtn) startBtn.click();
      break;
    case 'ArrowLeft':
    case 'h':
      e.preventDefault();
      if (prevBtn) prevBtn.click();
      break;
    case 's':
      // Toggle speaker notes
      document.querySelectorAll('.speaker-notes').forEach(function(el) {
        el.classList.toggle('visible');
      });
      break;
    case 'Escape':
      var homeLink = document.querySelector('.nav-home');
      window.location.href = homeLink ? homeLink.href : '/';
      break;
  }
});

// Wrap speaker notes sections
document.addEventListener('DOMContentLoaded', function() {
  var body = document.querySelector('.slide-body');
  if (!body) return;

  var headers = body.querySelectorAll('h2');
  headers.forEach(function(h) {
    if (h.textContent.trim() === 'Speaker Notes') {
      var wrapper = document.createElement('div');
      wrapper.className = 'speaker-notes';
      h.parentNode.insertBefore(wrapper, h);
      wrapper.appendChild(h);

      var next = wrapper.nextElementSibling;
      while (next && next.tagName !== 'H2' && next.tagName !== 'HR') {
        var current = next;
        next = next.nextElementSibling;
        wrapper.appendChild(current);
      }
    }
  });

  // Add keyboard hint on slide pages
  if (document.querySelector('.slide-wrapper')) {
    var hint = document.createElement('div');
    hint.className = 'kbd-hint';
    hint.innerHTML = '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> navigate &nbsp; <kbd>s</kbd> notes &nbsp; <kbd>esc</kbd> home';
    document.body.appendChild(hint);
  }
});
