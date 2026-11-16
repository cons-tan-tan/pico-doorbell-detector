{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay/5b87ee0513c29127a01ad06e9cb362ea5ce91e78";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    tflm = {
      url = "github:tensorflow/tflite-micro/2c13ceb07010b642785ffb20f02e82dabb2bb435";
      flake = false;
    };
    flatbuffers = {
      url = "github:google/flatbuffers/v25.9.23";
      flake = false;
    };
    embassy = {
      url = "github:embassy-rs/embassy/e23bc27945d76f94a4051b2f96ccd08dc7abdb26";
      flake = false;
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      imports = [
        ./nix/toolchain.nix
        ./nix/apps.nix
        ./nix/checks.nix
        ./nix/dev-shell.nix
      ];
    };
}
