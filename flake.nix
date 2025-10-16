{
  description = "Vulkan water simulation devshell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, devenv, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = devenv.lib.mkShell {
        inherit inputs nixpkgs;
        modules = [ ./devenv.nix ];
      };
    };
}
