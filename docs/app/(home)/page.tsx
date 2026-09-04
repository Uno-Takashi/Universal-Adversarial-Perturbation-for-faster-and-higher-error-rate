import Link from "next/link";
import data from "@/data/results.json";

const headline = [
  { value: "~100", label: "images before the perturbation beats random noise" },
  { value: "100%", label: "fitted-set fooling rate that generalised to 11%" },
  { value: `${data.runCount}`, label: "runs behind these figures" },
];

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center gap-10 px-6 py-16">
      <div className="flex flex-col gap-4">
        <p className="text-fd-muted-foreground text-sm">Experimental record</p>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          How many images does a universal adversarial perturbation actually
          need?
        </h1>
        <p className="text-fd-muted-foreground text-lg">
          A reproduction study of a multi-target universal perturbation
          algorithm, measured on held-out images against a random-perturbation
          baseline — the two things its original evaluation was missing.
        </p>
      </div>

      <dl className="grid gap-4 sm:grid-cols-3">
        {headline.map((item) => (
          <div key={item.label} className="rounded-lg border p-4">
            <dt className="text-2xl font-semibold tabular-nums">
              {item.value}
            </dt>
            <dd className="text-fd-muted-foreground mt-1 text-sm">
              {item.label}
            </dd>
          </div>
        ))}
      </dl>

      <div className="flex flex-wrap gap-3">
        <Link
          href="/docs"
          className="bg-fd-primary text-fd-primary-foreground rounded-md px-4 py-2 text-sm font-medium"
        >
          Read the results
        </Link>
        <Link
          href="/docs/method"
          className="rounded-md border px-4 py-2 text-sm font-medium"
        >
          The method
        </Link>
        <a
          href="https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate"
          className="rounded-md border px-4 py-2 text-sm font-medium"
        >
          Source
        </a>
      </div>
    </main>
  );
}
