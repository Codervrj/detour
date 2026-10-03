import "@testing-library/jest-dom/vitest";

// jsdom implements neither of these, and the map component uses both. Without the stubs
// the whole screen throws during render and every assertion fails for the wrong reason.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

if (!HTMLCanvasElement.prototype.getContext) {
  // Returning null is a documented outcome of getContext, and the component handles it.
  HTMLCanvasElement.prototype.getContext = (() => null) as never;
}
