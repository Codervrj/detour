/** The results page must report the numbers as measured, including against us. */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Results } from "../screens/Results";
import { ApiError } from "../api/client";
import { report } from "./fixtures";

const mocks = vi.hoisted(() => ({
  listeners: vi.fn(),
  recommendations: vi.fn(),
  latestReport: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, api: mocks };
});

function renderResults() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Results />
    </QueryClientProvider>,
  );
}

describe("Results", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.latestReport.mockResolvedValue(report);
  });

  it("says plainly when discovery recall went down", async () => {
    renderResults();
    // ours 0.0793 against ALS 0.1186 is a fall, and the page must not bury it.
    expect(await screen.findByText(/-0.0393/)).toBeInTheDocument();
    expect(screen.getByText(/the headline claim does not hold/i)).toBeInTheDocument();
  });

  it("reports whether the popularity baseline was beaten", async () => {
    renderResults();
    expect(await screen.findByText(/ahead, 0.0508 against 0.0352/)).toBeInTheDocument();
  });

  it("draws the frontier with our model marked on it", async () => {
    renderResults();
    const chart = await screen.findByRole("img", { name: /accuracy against novelty/i });
    expect(chart).toBeInTheDocument();
    expect(chart).toHaveAccessibleName(/personalised model reaches NDCG 0.0508/);
  });

  it("lists every baseline in the comparison table", async () => {
    renderResults();
    const table = await screen.findByRole("table", { name: /all baselines/i });
    for (const label of ["popularity", "ALS", "ALS with a dial per listener"]) {
      expect(table).toHaveTextContent(label);
    }
  });

  it("breaks results down by how adventurous listeners are", async () => {
    renderResults();
    const table = await screen.findByRole("table", { name: /explorer tercile/i });
    expect(table).toHaveTextContent("loyalists");
    expect(table).toHaveTextContent("explorers");
  });

  it("admits when confidence intervals are missing", async () => {
    renderResults();
    expect(await screen.findByText("not computed yet")).toBeInTheDocument();
  });

  it("explains how to produce a report when none exists", async () => {
    mocks.latestReport.mockRejectedValue(new ApiError("No eval report yet.", 404));
    renderResults();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("run.py eval");
  });
});
