{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let 
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in 
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python3

        python3Packages.torch
        python3Packages.torchvision
        python3Packages.pytorch-lightning
        python3Packages.wandb
        python3Packages.optuna
      ];
    };
  };
}
