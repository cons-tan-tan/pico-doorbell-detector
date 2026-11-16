{ inputs, ... }:
{
  perSystem =
    { pkgs, system, ... }:
    let
      rust = pkgs.rust-bin.stable."1.100.0".default.override {
        targets = [ "thumbv8m.main-none-eabihf" ];
      };
      firmwareEnv = {
        TFLM_SRC = "${inputs.tflm}";
        FLATBUFFERS_INCLUDE = "${inputs.flatbuffers}/include";
        CYW43_FIRMWARE_DIR = "${inputs.embassy}/cyw43-firmware";
        TARGET_CXX = "${pkgs.gcc-arm-embedded}/bin/arm-none-eabi-g++";
        TARGET_AR = "${pkgs.gcc-arm-embedded}/bin/arm-none-eabi-ar";
      };
      firmwareTools = [
        rust
        pkgs.gcc-arm-embedded
        pkgs.stdenv.cc
        pkgs.picotool
        pkgs.git
        pkgs.coreutils
      ];
      pythonLibraries = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc
        pkgs.zlib
        pkgs.libsndfile
      ];
    in
    {
      _module.args = {
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.rust-overlay.overlays.default ];
        };
        inherit
          rust
          firmwareEnv
          firmwareTools
          pythonLibraries
          ;
      };
    };
}
