{ pkgs, ... }:
{
  packages = with pkgs; [
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    libXcursor
    libXrandr
    libXi
    libX11
    libxcb
    libXrender
    pkg-config
    gcc
    rustup
    rust-analyzer
  ];

  enterShell = ''
    rustup toolchain install stable --profile minimal
    rustup default stable
  '';
}
