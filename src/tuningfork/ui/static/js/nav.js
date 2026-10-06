// Leaving a page with unsaved work: the page says so, and navigation asks first.
//
// A view that holds an unsaved edit sets `leaving.guard` to a function returning the
// question to ask (or null when there is nothing to lose). The router asks it before moving
// on, and the browser asks it before the tab closes or reloads.

export const leaving = { guard: null };

export function unsaved() {
  return leaving.guard ? leaving.guard() : null;
}

/** The address the page is at: as the router last saw it, or as a view set it in place. */
export const place = { hash: location.hash };

/** Change the address without leaving the page — for a view that opens something in place
 *  (a step on a run's timeline, a held call), so the address can be shared and reloaded —
 *  and tell the router, which would otherwise take the old address for the current one. */
export function settle(hash) {
  if (hash === location.hash) return;
  history.replaceState(history.state, "", hash);
  place.hash = hash;
}

window.addEventListener("beforeunload", (event) => {
  if (unsaved()) {
    event.preventDefault();
    event.returnValue = ""; // the browser shows its own question
  }
});
