{
  perSystem =
    {
      pkgs,
      rust,
      firmwareEnv,
      runners,
      ...
    }:
    let
      rustPlatform = pkgs.makeRustPlatform {
        cargo = rust;
        rustc = rust;
      };
    in
    {
      checks = {
        runners = pkgs.linkFarm "doorbell-runners" (
          pkgs.lib.mapAttrsToList (name: path: { inherit name path; }) runners
        );
        firmware = rustPlatform.buildRustPackage (
          firmwareEnv
          // {
            pname = "doorbell-firmware-check";
            version = "0.1.0";
            src = pkgs.lib.cleanSourceWith {
              src = ../firmware;
              filter = path: _: baseNameOf path != "target";
            };
            cargoHash = "sha256-xN2eY1rhs7EdqPN/N7KIu97y6ackJeqsvFpzX5dE2U0=";
            cargoBuildFlags = [ "--lib" ];
            cargoTestFlags = [ "--lib" ];
            installPhase = "mkdir -p $out";
          }
        );
      };
    };
}
