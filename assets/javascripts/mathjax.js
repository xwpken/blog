window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: "(^| )no-mathjax( |$)",
    processHtmlClass: "arithmatex"
  }
};

var mathTypesetScheduled = false;

function withMathJaxReady(callback, retries) {
  var remaining = typeof retries === "number" ? retries : 60;

  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    callback();
    return;
  }

  if (remaining <= 0) {
    return;
  }

  setTimeout(function () {
    withMathJaxReady(callback, remaining - 1);
  }, 100);
}

function typesetMathNow() {
  withMathJaxReady(function () {
    window.MathJax.typesetPromise();
  });
}

function scheduleTypesetMath() {
  if (mathTypesetScheduled) {
    return;
  }

  mathTypesetScheduled = true;
  requestAnimationFrame(function () {
    mathTypesetScheduled = false;
    typesetMathNow();
  });
}

function scheduleTypesetBurst() {
  scheduleTypesetMath();
  setTimeout(scheduleTypesetMath, 80);
  setTimeout(scheduleTypesetMath, 240);
}

if (typeof document$ !== "undefined") {
  document$.subscribe(scheduleTypesetBurst);
}

document.addEventListener("DOMContentLoaded", scheduleTypesetBurst);
window.addEventListener("load", scheduleTypesetBurst);

if (typeof MutationObserver !== "undefined") {
  var observer = new MutationObserver(function () {
    scheduleTypesetMath();
  });

  document.addEventListener("DOMContentLoaded", function () {
    if (document.body) {
      observer.observe(document.body, { childList: true, subtree: true });
    }
  });
}
