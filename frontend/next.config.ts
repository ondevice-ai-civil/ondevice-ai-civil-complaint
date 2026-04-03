import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/GovOn',
  assetPrefix: '/GovOn/',
  // Images optimization is not supported with static export, so we need to disable it
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
