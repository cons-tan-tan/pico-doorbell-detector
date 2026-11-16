{
  perSystem =
    {
      pkgs,
      firmwareEnv,
      firmwareTools,
      pythonLibraries,
      ...
    }:
    {
      devShells.default = pkgs.mkShell (
        firmwareEnv
        // {
          packages =
            firmwareTools
            ++ (with pkgs; [
              tio
              uv
              ruff
              nixfmt
              secretspec

              ffmpeg
              sox
              fluidsynth
            ]);

          shellHook = ''
            export SSL_CERT_FILE="''${SSL_CERT_FILE:-${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt}"
            export DOORBELL_SOUNDFONT="${pkgs.soundfont-fluid}/share/soundfonts/FluidR3_GM2-2.sf2"
            export LD_LIBRARY_PATH="${pythonLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # TensorFlow 2.21 does not discover cuSOLVER in uv's .venv by default.
            export LD_LIBRARY_PATH="$PWD/ml/.venv/lib/python3.13/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"

            if [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi
          '';
        }
      );
    };
}
