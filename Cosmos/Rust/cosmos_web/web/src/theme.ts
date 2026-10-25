// Colors come from the CSS tokens so the canvas and the 3D scene follow the
// page's light and dark themes. `onThemeChange` fires on an OS switch and on
// the viewer's explicit data-theme toggle.

export function token(name: string): string
{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export const SERIES_TOKENS = ["--series-1", "--series-2", "--series-3", "--series-4"];

export function onThemeChange(callback: () => void): void
{
  window.matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", callback);
  new MutationObserver(callback).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
}
