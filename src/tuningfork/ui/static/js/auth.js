// The access token: what lets this page change anything.
//
// `tuningfork serve` prints the UI's address with `?token=…`. The page keeps the token for
// this tab only — sessionStorage, never localStorage — and takes it out of the address bar
// at once, so it is not left in history, a bookmark or a screenshot. Without a valid token
// the page shows everything and changes nothing: every control is marked `data-control`,
// and its action goes through `guard`.

const KEY = "tuningfork.token";

export const auth = {
  token: null,
  required: false, // whether this server takes changes at all (`tuningfork ui` does not)
  valid: false, // whether it accepted this page's token
};

const listeners = new Set();

export function onAuthChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function changed() {
  document.documentElement.classList.toggle("read-only", auth.required && !auth.valid);
  for (const listener of [...listeners]) listener(auth);
}

/** Read `?token=` once, keep it for this tab, and take it out of the address bar. */
export function pickUpToken() {
  const params = new URLSearchParams(location.search);
  const given = params.get("token");
  if (!given) {
    auth.token = readStored();
    return;
  }
  keep(given);
  params.delete("token");
  const rest = params.toString();
  history.replaceState(history.state, "", `${location.pathname}${rest ? `?${rest}` : ""}${location.hash}`);
}

function readStored() {
  try {
    return sessionStorage.getItem(KEY);
  } catch {
    return null; // storage refused: the token lasts as long as this page
  }
}

/** A token pasted into the page, or picked up from its address. */
export function keep(token) {
  auth.token = token ? String(token).trim() : null;
  try {
    if (auth.token) sessionStorage.setItem(KEY, auth.token);
    else sessionStorage.removeItem(KEY);
  } catch {
    // private window: the token lasts for this page only
  }
}

/** What the server said about this page's token (`/status`'s `auth`). */
export function setAuth(status) {
  auth.required = Boolean(status?.required);
  auth.valid = Boolean(status?.valid);
  changed();
}

/** A change was refused for want of a valid token: the page is read-only from now on. */
export function refused() {
  if (!auth.valid) return;
  auth.valid = false;
  changed();
}

export const canAct = () => !auth.required || auth.valid;

/** An action that changes state, refused while the page has no valid token. A checkbox
 *  flipped meanwhile is flipped back, so the page never shows a change that did not happen. */
export function guard(action) {
  return (event, ...rest) => {
    if (canAct()) return action(event, ...rest);
    event?.preventDefault?.();
    if (event?.target?.type === "checkbox") event.target.checked = !event.target.checked;
    for (const listener of [...listeners]) listener(auth, { blocked: true });
    return undefined;
  };
}
