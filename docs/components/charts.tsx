"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import data from "@/data/results.json";

export type Run = {
  sweep: string;
  model: string;
  paradigm: string;
  numImages: number;
  numValImages: number | null;
  searchNum: number;
  imageSize: number[] | null;
  genFooling: number;
  valFooling: number;
  randomBaseline: number;
  margin: number;
  cleanTop1: number | null;
  seconds: number | null;
};

const runs = data.runs as Run[];
const SERIES = [
  "var(--viz-1)",
  "var(--viz-2)",
  "var(--viz-3)",
  "var(--viz-4)",
  "var(--viz-5)",
  "var(--viz-6)",
];

// A fixed order, so a model keeps its colour no matter which chart it appears in or how many
// series a filter leaves behind.
const MODEL_ORDER = [
  "inception5h",
  "mobilenet_v2",
  "resnet50",
  "inception_v3",
  "vit_b16",
  "swin_tiny",
  "deit_b16_distilled",
  "convnext_tiny",
  "mobilenet_v3_large",
  "resnet_vd_50_ssld",
  "vgg16",
  "efficientnet_b0",
];

function colourFor(model: string) {
  const index = MODEL_ORDER.indexOf(model);
  return SERIES[(index < 0 ? MODEL_ORDER.length : index) % SERIES.length];
}

export function select(sweep: string, extra?: (run: Run) => boolean) {
  return runs.filter((run) => run.sweep === sweep && (!extra || extra(run)));
}

function modelsIn(rows: Run[]) {
  const present = new Set(rows.map((r) => r.model));
  return MODEL_ORDER.filter((m) => present.has(m)).concat(
    [...present].filter((m) => !MODEL_ORDER.includes(m)),
  );
}

function pivot(
  rows: Run[],
  xKey: "numImages" | "searchNum",
  yKey: keyof Run,
  scale = 100,
) {
  const xs = [...new Set(rows.map((r) => r[xKey] as number))].sort(
    (a, b) => a - b,
  );
  return xs.map((x) => {
    const point: Record<string, number | null> = { x };
    for (const model of modelsIn(rows)) {
      const match = rows.find((r) => r[xKey] === x && r.model === model);
      point[model] = match ? (match[yKey] as number) * scale : null;
    }
    return point;
  });
}

const axisStyle = { fill: "var(--viz-axis)", fontSize: 12 };

function ChartFrame({
  children,
  caption,
  height = 340,
}: {
  children: React.ReactElement;
  caption: string;
  height?: number;
}) {
  return (
    <figure className="my-6 not-prose">
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 text-sm text-fd-muted-foreground">
        {caption}
      </figcaption>
    </figure>
  );
}

function tooltipProps(unit: string) {
  return {
    contentStyle: {
      background: "var(--viz-surface)",
      border: "1px solid var(--viz-grid)",
      borderRadius: 8,
      fontSize: 12,
      color: "var(--viz-rule)",
    },
    // `unknown` parameters keep this assignable to Recharts' own formatter signature,
    // which is wider than the numbers these charts actually plot.
    formatter: (value: unknown, name: unknown): [string, string] => [
      typeof value === "number" ? `${value.toFixed(1)}${unit}` : "—",
      String(name),
    ],
  };
}

/** Held-out fooling rate minus the random baseline, against generation-set size. */
export function MarginVsImages({ sweep = "images" }: { sweep?: string }) {
  const rows = select(sweep);
  const points = pivot(rows, "numImages", "margin");
  const models = modelsIn(rows);

  return (
    <ChartFrame caption="Above zero, the perturbation beats random noise of the same l∞ budget. Below it, it does not.">
      <LineChart
        data={points}
        margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
      >
        <CartesianGrid
          stroke="var(--viz-grid)"
          strokeDasharray="3 3"
          vertical={false}
        />
        <XAxis
          dataKey="x"
          scale="log"
          domain={["dataMin", "dataMax"]}
          type="number"
          ticks={[...new Set(points.map((p) => p.x as number))]}
          tick={axisStyle}
          stroke="var(--viz-grid)"
          label={{
            value: "images fitted to",
            position: "insideBottom",
            offset: -4,
            ...axisStyle,
          }}
        />
        <YAxis
          tick={axisStyle}
          stroke="var(--viz-grid)"
          unit="pt"
          label={{
            value: "val − random",
            angle: -90,
            position: "insideLeft",
            ...axisStyle,
          }}
        />
        <Tooltip {...tooltipProps("pt")} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine y={0} stroke="var(--viz-rule)" strokeOpacity={0.5} />
        {models.map((model) => (
          <Line
            key={model}
            type="linear"
            dataKey={model}
            stroke={colourFor(model)}
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ChartFrame>
  );
}

/** Held-out fooling rate against generation-set size, with each model's random baseline. */
export function FoolingVsImages({ sweep = "images" }: { sweep?: string }) {
  const rows = select(sweep);
  const models = modelsIn(rows);
  const xs = [...new Set(rows.map((r) => r.numImages))].sort((a, b) => a - b);
  const points = xs.map((x) => {
    const point: Record<string, number | null> = { x };
    for (const model of models) {
      const match = rows.find((r) => r.numImages === x && r.model === model);
      point[model] = match ? match.valFooling * 100 : null;
      point[`${model} (random)`] = match ? match.randomBaseline * 100 : null;
    }
    return point;
  });

  return (
    <ChartFrame caption="Solid: the universal perturbation. Dashed: a random perturbation at the same l∞ budget.">
      <LineChart
        data={points}
        margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
      >
        <CartesianGrid
          stroke="var(--viz-grid)"
          strokeDasharray="3 3"
          vertical={false}
        />
        <XAxis
          dataKey="x"
          scale="log"
          domain={["dataMin", "dataMax"]}
          type="number"
          ticks={xs}
          tick={axisStyle}
          stroke="var(--viz-grid)"
          label={{
            value: "images fitted to",
            position: "insideBottom",
            offset: -4,
            ...axisStyle,
          }}
        />
        <YAxis tick={axisStyle} stroke="var(--viz-grid)" unit="%" />
        <Tooltip {...tooltipProps("%")} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {models.map((model) => (
          <Line
            key={model}
            type="linear"
            dataKey={model}
            stroke={colourFor(model)}
            strokeWidth={2}
            dot={{ r: 4 }}
            connectNulls
          />
        ))}
        {models.map((model) => (
          <Line
            key={`${model}-random`}
            type="linear"
            dataKey={`${model} (random)`}
            stroke={colourFor(model)}
            strokeWidth={1.4}
            strokeDasharray="5 4"
            strokeOpacity={0.8}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ChartFrame>
  );
}

/** Held-out fooling rate against the multiplicity M. */
export function FoolingVsMultiplicity() {
  const rows = select("multiplicity");
  if (rows.length === 0) {
    return <NoData what="the multiplicity sweep" />;
  }
  const points = pivot(rows, "searchNum", "valFooling");
  const models = modelsIn(rows);

  return (
    <ChartFrame caption="Attacking more classes per image costs proportionally more compute; the return flattens.">
      <LineChart
        data={points}
        margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
      >
        <CartesianGrid
          stroke="var(--viz-grid)"
          strokeDasharray="3 3"
          vertical={false}
        />
        <XAxis
          dataKey="x"
          type="number"
          domain={["dataMin", "dataMax"]}
          tick={axisStyle}
          stroke="var(--viz-grid)"
          label={{
            value: "multiplicity M",
            position: "insideBottom",
            offset: -4,
            ...axisStyle,
          }}
        />
        <YAxis tick={axisStyle} stroke="var(--viz-grid)" unit="%" />
        <Tooltip {...tooltipProps("%")} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {models.map((model) => (
          <Line
            key={model}
            type="linear"
            dataKey={model}
            stroke={colourFor(model)}
            strokeWidth={2}
            dot={{ r: 4 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ChartFrame>
  );
}

export function NoData({ what }: { what: string }) {
  return (
    <div className="my-6 rounded-lg border border-dashed p-4 text-sm text-fd-muted-foreground">
      No results for {what} yet — this page renders whatever
      `docs/data/results.json` contains, so it fills in once the sweep finishes.
    </div>
  );
}
