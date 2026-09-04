import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

// GitHub Pages serves a project site under /<repo>, so the build needs a base path. It is set by
// the Pages workflow; a local `npm run build` produces a root-relative site.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/** @type {import('next').NextConfig} */
const config = {
  output: "export",
  reactStrictMode: true,
  basePath,
  images: { unoptimized: true },
  trailingSlash: true,
};

export default withMDX(config);
