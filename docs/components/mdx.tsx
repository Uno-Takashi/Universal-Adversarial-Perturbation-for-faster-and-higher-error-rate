import defaultMdxComponents from "fumadocs-ui/mdx";
import type { MDXComponents } from "mdx/types";
import {
  FoolingVsImages,
  FoolingVsMultiplicity,
  MarginVsImages,
  NoData,
} from "@/components/charts";
import {
  GeneratedAt,
  MarginMatrix,
  ModelRoster,
  ModelTable,
  RunCount,
} from "@/components/tables";

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    // Charts and tables are available to every page without an import.
    MarginVsImages,
    FoolingVsImages,
    FoolingVsMultiplicity,
    NoData,
    ModelTable,
    MarginMatrix,
    ModelRoster,
    RunCount,
    GeneratedAt,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
