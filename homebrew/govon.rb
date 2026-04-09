# typed: false
# frozen_string_literal: true

class Govon < Formula
  include Language::Python::Virtualenv

  desc "Shell-first local agentic runtime for administrative assistance and civil complaint workflows"
  homepage "https://github.com/GovOn-org/GovOn"

  # TODO: PyPI 배포 후 업데이트
  url "https://files.pythonhosted.org/packages/source/G/GovOn/GovOn-1.0.1.tar.gz"
  sha256 "# TODO: PyPI 배포 후 실제 sha256으로 업데이트"
  license "MIT"
  version "1.0.1"

  bottle :unneeded

  depends_on "python@3.12"

  # Core runtime dependencies (inference extras 제외, CLI 실행에 필요한 최소 의존성)
  resource "httpx" do
    # TODO: PyPI 배포 후 업데이트
    url "https://files.pythonhosted.org/packages/source/h/httpx/httpx-0.27.0.tar.gz"
    sha256 "# TODO: 업데이트 필요"
  end

  resource "python-dotenv" do
    # TODO: PyPI 배포 후 업데이트
    url "https://files.pythonhosted.org/packages/source/p/python-dotenv/python_dotenv-1.0.0.tar.gz"
    sha256 "# TODO: 업데이트 필요"
  end

  resource "pyyaml" do
    # TODO: PyPI 배포 후 업데이트
    url "https://files.pythonhosted.org/packages/source/P/PyYAML/PyYAML-6.0.tar.gz"
    sha256 "# TODO: 업데이트 필요"
  end

  resource "loguru" do
    # TODO: PyPI 배포 후 업데이트
    url "https://files.pythonhosted.org/packages/source/l/loguru/loguru-0.7.0.tar.gz"
    sha256 "# TODO: 업데이트 필요"
  end

  resource "rich" do
    # TODO: PyPI 배포 후 업데이트
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-13.0.0.tar.gz"
    sha256 "# TODO: 업데이트 필요"
  end

  resource "prompt_toolkit" do
    # TODO: PyPI 배포 후 업데이트
    url "https://files.pythonhosted.org/packages/source/p/prompt_toolkit/prompt_toolkit-3.0.0.tar.gz"
    sha256 "# TODO: 업데이트 필요"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    # govon CLI 진입점 기본 동작 확인
    system bin/"govon", "--help"
  end
end
