/** The dial must change the list, and every data view needs usable empty and error states. */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Listen } from "../screens/Listen";
import { ApiError } from "../api/client";
import { explorer, listeners, recommendations } from "./fixtures";

const mocks = vi.hoisted(() => ({
  listeners: vi.fn(),
  recommendations: vi.fn(),
  latestReport: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, api: mocks };
});

function renderListen() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Listen />
    </QueryClientProvider>,
  );
}

describe("Listen", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.listeners.mockResolvedValue(listeners);
    mocks.recommendations.mockImplementation((_user: string, lambda: number) =>
      Promise.resolve(recommendations(lambda)),
    );
  });

  it("picks a listener who roams and shows their numbers", async () => {
    renderListen();
    await waitFor(() => expect(screen.getByRole("radiogroup")).toBeInTheDocument());

    expect(await screen.findByText(/410 listens/)).toBeInTheDocument();
    expect(screen.getByText(/scoring 0.44 on the explorer scale/)).toBeInTheDocument();
  });

  it("starts the dial at the listener's own setting", async () => {
    renderListen();
    const dial = await screen.findByRole("slider", { name: /how far out/i });
    expect(dial).toHaveValue(String(explorer.personalised_lambda));
  });

  it("re-ranks the list when the dial moves", async () => {
    renderListen();
    const dial = await screen.findByRole("slider", { name: /how far out/i });

    await waitFor(() => expect(mocks.recommendations).toHaveBeenCalled());
    mocks.recommendations.mockClear();

    // jsdom does not implement native arrow-key stepping on a range input, so the value
    // is set directly. Real keyboard stepping is covered by the Playwright run.
    fireEvent.change(dial, { target: { value: "0.9" } });

    await waitFor(() => expect(mocks.recommendations).toHaveBeenCalled());
    const lambdas = mocks.recommendations.mock.calls.map((call) => call[1] as number);
    expect(Math.max(...lambdas)).toBeGreaterThan(explorer.personalised_lambda);
  });

  it("raises the novelty of the list as the dial goes up", async () => {
    renderListen();
    const dial = await screen.findByRole("slider", { name: /how far out/i });

    fireEvent.change(dial, { target: { value: "0.1" } });
    const low = await screen.findByText(/novelty 0.25 of 1/);
    expect(low).toBeInTheDocument();

    fireEvent.change(dial, { target: { value: "0.9" } });
    await waitFor(() => expect(screen.getByText(/novelty 0.65 of 1/)).toBeInTheDocument());
  });

  it("exposes the dial to keyboard and screen reader users", async () => {
    renderListen();
    const dial = await screen.findByRole("slider", { name: /how far out/i });

    // Reachable in the tab order and not trapped behind a custom widget.
    expect(dial.tagName).toBe("INPUT");
    expect(dial).not.toHaveAttribute("tabindex", "-1");
    dial.focus();
    expect(dial).toHaveFocus();

    // The raw number alone would not say which end means what.
    expect(dial).toHaveAttribute(
      "aria-valuetext",
      expect.stringContaining("familiar") as unknown as string,
    );
  });

  it("lets you jump back to the listener's own setting", async () => {
    renderListen();
    const dial = await screen.findByRole("slider", { name: /how far out/i });

    fireEvent.change(dial, { target: { value: "0.9" } });
    const reset = await screen.findByRole("button", { name: /use their setting/i });
    await userEvent.click(reset);

    await waitFor(() => expect(dial).toHaveValue(String(explorer.personalised_lambda)));
  });

  it("switches listener when another is chosen, and resets the dial to theirs", async () => {
    renderListen();
    await waitFor(() => expect(screen.getByRole("radiogroup")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("radio", { name: /user-001/ }));

    const dial = await screen.findByRole("slider", { name: /how far out/i });
    await waitFor(() => expect(dial).toHaveValue("0.1"));
  });

  it("marks familiar and new artists in words, not only by colour", async () => {
    renderListen();
    const list = await screen.findByRole("list", { name: /recommended tracks/i });
    expect(within(list).getByText("you know them")).toBeInTheDocument();
    expect(within(list).getByText("new to you")).toBeInTheDocument();
  });

  it("reveals the reason a track is in the list", async () => {
    renderListen();
    const buttons = await screen.findAllByRole("button", { name: "why this" });
    expect(buttons[0]).toHaveAttribute("aria-expanded", "false");

    await userEvent.click(buttons[0]);
    expect(buttons[0]).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/You already listen to this artist/)).toBeInTheDocument();
  });

  it("explains how to start the API when it cannot be reached", async () => {
    mocks.listeners.mockRejectedValue(new ApiError("Cannot reach the Detour API.", 0));
    renderListen();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/API is not running/);
    expect(alert).toHaveTextContent("run.py demo");
  });

  it("tells you how to build artefacts when the pipeline has not run", async () => {
    mocks.listeners.mockRejectedValue(new ApiError("No model artefacts loaded.", 503));
    renderListen();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("run.py pipeline");
  });

  it("shows an actionable empty state when there are no listeners", async () => {
    mocks.listeners.mockResolvedValue([]);
    renderListen();

    expect(await screen.findByText(/No listeners in the artefacts/)).toBeInTheDocument();
    expect(screen.getByText(/run.py pipeline/)).toBeInTheDocument();
  });
});
