/** Loading, empty and error states. Every data view needs all three, and each one says
 * what to do next rather than just reporting a mood. */

interface ErrorProps {
  title: string;
  children: React.ReactNode;
  onRetry?: () => void;
}

export function ErrorState({ title, children, onRetry }: ErrorProps) {
  return (
    <div className="state state--error" role="alert">
      <h3>{title}</h3>
      <p>{children}</p>
      {onRetry && (
        <p style={{ marginTop: "var(--s-3)" }}>
          <button type="button" className="button" onClick={onRetry}>
            Try again
          </button>
        </p>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

/** Skeleton rows sized like the real list, so the layout does not jump when data lands. */
export function LoadingRows({ rows = 6, label }: { rows?: number; label: string }) {
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="visually-hidden">{label}</span>
      {Array.from({ length: rows }, (_, index) => (
        <div
          key={index}
          className="skeleton"
          style={{ marginBottom: "var(--s-3)", width: `${90 - index * 6}%` }}
        />
      ))}
    </div>
  );
}
