# typed: false
# frozen_string_literal: true

class Govon < Formula
  desc "Agentic TUI for Korean public-sector civil complaint workflows"
  homepage "https://github.com/GovOn-Org/GovOn"
  url "https://registry.npmjs.org/govon/-/govon-1.5.0.tgz"
  sha256 :no_check
  license "MIT"

  depends_on "node@22"

  def install
    system "npm", "install", *std_npm_args
    bin.install_symlink libexec.glob("bin/*")
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/govon --version")
  end
end
