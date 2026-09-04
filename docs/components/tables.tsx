import data from "@/data/results.json";
import type { Run } from "./charts";

const runs = data.runs as Run[];

const pct = (value: number | null | undefined) =>
  value === null || value === undefined ? "–" : `${(value * 100).toFixed(1)}%`;
const pt = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}pt`;

function Table({
  head,
  rows,
}: {
  head: string[];
  rows: (string | number)[][];
}) {
  return (
    <div className="my-6 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left">
            {head.map((h) => (
              <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b/50 border-b">
              {row.map((cell, j) => (
                <td
                  key={j}
                  className="px-3 py-1.5 whitespace-nowrap tabular-nums"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Every run of one sweep for one model. The table view the charts are read against. */
export function ModelTable({
  model,
  sweep = "images",
}: {
  model: string;
  sweep?: string;
}) {
  const rows = runs
    .filter((r) => r.sweep === sweep && r.model === model)
    .sort((a, b) => a.numImages - b.numImages || a.searchNum - b.searchNum);

  if (rows.length === 0)
    return <p className="text-fd-muted-foreground text-sm">No runs yet.</p>;

  return (
    <Table
      head={[
        "gen images",
        "M",
        "gen fool",
        "val fool",
        "random",
        "margin",
        "clean top-1",
        "sec",
      ]}
      rows={rows.map((r) => [
        r.numImages,
        r.searchNum,
        pct(r.genFooling),
        pct(r.valFooling),
        pct(r.randomBaseline),
        pt(r.margin),
        pct(r.cleanTop1),
        r.seconds === null ? "–" : Math.round(r.seconds),
      ])}
    />
  );
}

/** Margin for every model at every generation-set size, in one grid. */
export function MarginMatrix({ sweep = "images" }: { sweep?: string }) {
  const rows = runs.filter((r) => r.sweep === sweep);
  if (rows.length === 0)
    return <p className="text-fd-muted-foreground text-sm">No runs yet.</p>;

  const models = [...new Set(rows.map((r) => r.model))];
  const sizes = [...new Set(rows.map((r) => r.numImages))].sort(
    (a, b) => a - b,
  );

  return (
    <Table
      head={["gen images", ...models]}
      rows={sizes.map((size) => [
        size,
        ...models.map((model) => {
          const match = rows.find(
            (r) => r.numImages === size && r.model === model,
          );
          return match ? pt(match.margin) : "–";
        }),
      ])}
    />
  );
}

/** One row per model: what it is, and how it behaves. */
export function ModelRoster({ sweep = "images" }: { sweep?: string }) {
  const rows = runs.filter((r) => r.sweep === sweep);
  if (rows.length === 0)
    return <p className="text-fd-muted-foreground text-sm">No runs yet.</p>;

  const models = [...new Set(rows.map((r) => r.model))];
  return (
    <Table
      head={[
        "model",
        "paradigm",
        "input",
        "clean top-1",
        "random baseline",
        "best margin",
      ]}
      rows={models.map((model) => {
        const mine = rows.filter((r) => r.model === model);
        const best = mine.reduce((a, b) => (b.margin > a.margin ? b : a));
        const cleans = mine
          .map((r) => r.cleanTop1)
          .filter((v): v is number => v !== null);
        const baselines = mine.map((r) => r.randomBaseline);
        const mean = (xs: number[]) =>
          xs.reduce((a, b) => a + b, 0) / xs.length;
        return [
          model,
          mine[0].paradigm,
          mine[0].imageSize ? `${mine[0].imageSize[0]}²` : "–",
          cleans.length ? pct(mean(cleans)) : "–",
          pct(mean(baselines)),
          `${pt(best.margin)} @ n=${best.numImages}`,
        ];
      })}
    />
  );
}

export function RunCount() {
  return <>{data.runCount}</>;
}

export function GeneratedAt() {
  return <>{data.generatedAt.replace("T", " ").replace("+00:00", " UTC")}</>;
}
