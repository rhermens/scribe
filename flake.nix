{
  description = "Scribe";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        cudaEnabled = pkgs.stdenv.hostPlatform.isLinux;
        # Match the installed driver's CUDA 13.2 support and use a compatible host GCC.
        cudaPackages = pkgs.cudaPackages_13_2;
        buildStdenv = if cudaEnabled then cudaPackages.backendStdenv else pkgs.stdenv;
        craneLib = (crane.mkLib pkgs).overrideScope (_final: _prev: {
          stdenvSelector = _: buildStdenv;
          mkShell = pkgs.mkShell.override { stdenv = buildStdenv; };
        });

        cudaBuildEnv = pkgs.lib.optionalAttrs cudaEnabled {
          # RTX 4070 SUPER (Ada). Explicit targeting also works in GPU-less build sandboxes.
          CMAKE_CUDA_ARCHITECTURES = "89";
          # whisper-rs-sys links libcuda directly; use the toolkit stub only at link time.
          NIX_LDFLAGS = "-L${cudaPackages.cuda_cudart}/lib/stubs";
        };

        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;
          nativeBuildInputs = with pkgs; [ pkg-config cmake rustPlatform.bindgenHook ]
            ++ lib.optionals cudaEnabled [ cudaPackages.cuda_nvcc autoAddDriverRunpath ];
          buildInputs = [ pkgs.openssl ] ++ pkgs.lib.optionals cudaEnabled (with cudaPackages; [
            cccl cuda_cudart cuda_culibos libcublas
          ]);
        } // cudaBuildEnv;

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        bin = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      {
        packages.default = bin;

        checks = {
          inherit bin;
          wt-clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });
        };

        devShells.default = craneLib.devShell ({
          checks = self.checks.${system};
          packages = with pkgs; [ rust-analyzer ];
          shellHook = pkgs.lib.optionalString cudaEnabled ''
            # NixOS exposes the installed NVIDIA driver outside the Nix store.
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        } // cudaBuildEnv);
      }
    );
}
