/* The preview viewer: navigation, Show Answer, and nothing else.
 *
 * This file is static. It is inlined verbatim into the exported page and its
 * SHA-256 is what the paired Content-Security-Policy authorises, so it can
 * carry no generated data — every card, label and count is already inert DOM
 * that the renderer escaped. Reading state out of the document rather than out
 * of an injected JSON blob is what lets `script-src` stay hash-only, with no
 * `unsafe-inline` and therefore no inline event-handler authority at all.
 *
 * The card's own `<details>` disclosures are untouched: they are native
 * elements and open and close independently of the flip below.
 */
(function () {
  "use strict";

  var cards = Array.prototype.slice.call(
    document.querySelectorAll("[data-preview-card]")
  );
  if (!cards.length) {
    return;
  }

  var previous = document.getElementById("preview-prev");
  var next = document.getElementById("preview-next");
  var flip = document.getElementById("preview-flip");
  var jump = document.getElementById("preview-jump");
  var position = document.getElementById("preview-position");

  var index = 0;
  var showingAnswer = false;

  function apply() {
    for (var i = 0; i < cards.length; i += 1) {
      var card = cards[i];
      var current = i === index;
      card.hidden = !current;
      var question = card.querySelector("[data-preview-question]");
      var answer = card.querySelector("[data-preview-answer]");
      /* One side at a time, the way the reviewer works. Moving only the
         answer left the question sitting above it, so a flipped card read as
         already answered. */
      if (question) {
        question.hidden = current && showingAnswer;
      }
      if (answer) {
        answer.hidden = !(current && showingAnswer);
      }
    }
    if (flip) {
      flip.textContent = showingAnswer ? "Show Question" : "Show Answer";
      flip.setAttribute("aria-pressed", showingAnswer ? "true" : "false");
    }
    if (position) {
      position.textContent = "Card " + (index + 1) + " of " + cards.length;
    }
    if (previous) {
      previous.disabled = index === 0;
    }
    if (next) {
      next.disabled = index === cards.length - 1;
    }
    if (jump && jump.selectedIndex !== index) {
      jump.selectedIndex = index;
    }
  }

  function select(target) {
    if (target < 0 || target >= cards.length) {
      return;
    }
    index = target;
    /* Every card starts on its question. Carrying the flip across a selection
       would show the next card's answer before it had asked anything. */
    showingAnswer = false;
    apply();
  }

  if (previous) {
    previous.addEventListener("click", function () {
      select(index - 1);
    });
  }
  if (next) {
    next.addEventListener("click", function () {
      select(index + 1);
    });
  }
  if (flip) {
    flip.addEventListener("click", function () {
      showingAnswer = !showingAnswer;
      apply();
    });
  }
  if (jump) {
    jump.addEventListener("change", function () {
      select(jump.selectedIndex);
    });
  }

  document.addEventListener("keydown", function (event) {
    var tag = event.target && event.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
      return;
    }
    if (event.key === "ArrowRight") {
      select(index + 1);
    } else if (event.key === "ArrowLeft") {
      select(index - 1);
    }
  });

  document.body.setAttribute("data-preview-ready", "true");
  apply();
})();
