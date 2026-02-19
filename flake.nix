{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      vendorHash = "sha256-+AJkOdUPhwBZPYPpvYaWIlqj+X0HTS8LX3loLkUyf9E=";
    in
    {
      checks.${system}.default = pkgs.buildGoModule {
        pname = "pico-doorbell-detector-check";
        version = "0.0.1";
        src = ./.;
        inherit vendorHash;
        buildPhase = "true";
        installPhase = "mkdir -p $out";
        doCheck = true;
        checkPhase = ''
          go vet ./...
          go test ./... -v
        '';
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          go
          tinygo
          just
          tio

          ffmpeg
          sox
        ];
      };
    };
}
