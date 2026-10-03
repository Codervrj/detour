/** The map screen, its states, and the honesty requirements. */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MapScreen } from "../screens/Map";
import { Results } from "../screens/Results";
import { Islands } from "../screens/Islands";
import { ApiError } from "../api/client";
import { background, listenerMap, report } from "./fixtures";

const mocks = vi.hoisted(() => ({
  map: vi.fn(),
  listeners: vi.fn(),
  listener: vi.fn(),
  me: vi.fn(),
  artist: vi.fn(),
  latestReport: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, api: mocks };
});

function renderWith(node: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.map.mockResolvedValue(background);
  mocks.listeners.mockResolvedValue(["u1", "u2"]);
  mocks.listener.mockResolvedValue(listenerMap);
  mocks.me.mockRejectedValue(new ApiError("No personal history imported.", 404));
  mocks.artist.mockResolvedValue({
    artist_mbid: "mb-pink-floyd",
    name: "Pink Floyd",
    neighbours: [
      { artist_mbid: "mb-genesis", name: "Genesis", similarity: 0.81, x: 0.3, y: 0.3 },
    ],
  });
  mocks.latestReport.mockResolvedValue(report);
});

describe("Map screen", () => {
  it("says plainly when you are looking at someone else", async () => {
    renderWith(<MapScreen />);
    expect(await screen.findByText(/have not imported your own history/i)).toBeInTheDocument();
    expect(screen.getByText(/run.py lastfm/)).toBeInTheDocument();
  });

  it("reports how much of the history the map could read", async () => {
    renderWith(<MapScreen />);
    // Coverage must be visible, not hidden: a map that silently dropped a third of
    // someone's taste would be misleading.
    expect(await screen.findByText(/covering 100% of your plays/i)).toBeInTheDocument();
  });

  it("lists the frontier as text, not only on the canvas", async () => {
    // The canvas is invisible to a screen reader, so the same information must be in the DOM.
    renderWith(<MapScreen />);
    expect(await screen.findByText("Bee Gees")).toBeInTheDocument();
    expect(screen.getByText("The Cars")).toBeInTheDocument();
  });

  it("says the frontier is the ranking that was evaluated", async () => {
    renderWith(<MapScreen />);
    expect(await screen.findByText(/exact ranking the evaluation scored/i)).toBeInTheDocument();
  });

  it("lets you switch listener", async () => {
    renderWith(<MapScreen />);
    const select = await screen.findByLabelText("Listener");
    await userEvent.selectOptions(select, "u2");
    await waitFor(() => expect(mocks.listener).toHaveBeenCalledWith("u2"));
  });

  it("explains how to start the API when it cannot be reached", async () => {
    mocks.map.mockRejectedValue(new ApiError("Cannot reach the Detour API.", 0));
    renderWith(<MapScreen />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/API is not running/);
    expect(alert).toHaveTextContent("run.py demo");
  });

  it("explains how to build the map when there is none", async () => {
    mocks.map.mockRejectedValue(new ApiError("No map loaded.", 503));
    renderWith(<MapScreen />);
    expect(await screen.findByRole("alert")).toHaveTextContent("run.py pipeline");
  });
});

describe("Results screen", () => {
  it("reports the headline against chance", async () => {
    renderWith(<Results />);
    // The figure appears in the verdict and again in the table, so scope to the verdict.
    const verdict = await screen.findByRole("heading", { name: "The answer" });
    const section = verdict.closest("section") as HTMLElement;
    expect(within(section).getByText("0.1158")).toBeInTheDocument();
    expect(within(section).getByText(/0\.4965/)).toBeInTheDocument();
  });

  it("states whether the popularity control was survived", async () => {
    renderWith(<Results />);
    expect(await screen.findByText(/The result survives/i)).toBeInTheDocument();
  });

  it("says so when the control is failed", async () => {
    mocks.latestReport.mockResolvedValue({
      ...report,
      models: {
        ...report.models,
        ppmi_svd: { ...report.models.ppmi_svd, matched_percentile: 0.61 },
      },
    });
    renderWith(<Results />);
    expect(await screen.findByText(/does not survive/i)).toBeInTheDocument();
  });

  it("admits when another model scored better", async () => {
    mocks.latestReport.mockResolvedValue({
      ...report,
      models: {
        ...report.models,
        item2vec: { ...report.models.item2vec, adoption_percentile: 0.05 },
      },
    });
    renderWith(<Results />);
    expect(await screen.findByText(/item2vec scored better/i)).toBeInTheDocument();
  });

  it("marks the popularity control as not meaningful rather than printing 1.0", async () => {
    renderWith(<Results />);
    const table = await screen.findByRole("table", { name: /all baselines/i });
    expect(within(table).getByText("not meaningful")).toBeInTheDocument();
  });

  it("admits confidence intervals are missing", async () => {
    renderWith(<Results />);
    expect(await screen.findByText("not computed yet")).toBeInTheDocument();
  });

  it("explains how to produce a report when none exists", async () => {
    mocks.latestReport.mockRejectedValue(new ApiError("No eval report yet.", 404));
    renderWith(<Results />);
    expect(await screen.findByRole("alert")).toHaveTextContent("run.py eval");
  });
});

describe("Islands screen", () => {
  it("shows each discovered cluster with its size", async () => {
    renderWith(<Islands />);
    expect(await screen.findByRole("heading", { name: "Pink Floyd" })).toBeInTheDocument();
    expect(screen.getByText(/328 artists, 7,415 plays/)).toBeInTheDocument();
  });

  it("describes isolation in words, not only as a number", async () => {
    renderWith(<Islands />);
    expect(await screen.findByText(/heart of your taste/i)).toBeInTheDocument();
    expect(screen.getByText(/off to one side/i)).toBeInTheDocument();
  });

  it("shows the edges of a listener's taste", async () => {
    renderWith(<Islands />);
    expect(await screen.findByText("Pino Donaggio")).toBeInTheDocument();
  });
});
