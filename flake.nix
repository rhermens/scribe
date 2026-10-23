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
        mkBackend = feature:
          let
            cudaEnabled = feature == "cuda";
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
              CUDA_COMPUTE_CAP = "89";
              # Candle's bindgen_cuda locates CUDA through include/cuda.h, not nvcc.
              CUDA_ROOT = "${pkgs.lib.getDev cudaPackages.cuda_cudart}";
              # whisper-rs-sys links libcuda directly; use the toolkit stub only at link time.
              NIX_LDFLAGS = "-L${cudaPackages.cuda_cudart}/lib/stubs";
            };

            commonArgs = {
              src = craneLib.cleanCargoSource ./.;
              strictDeps = true;
              cargoExtraArgs = "--locked" + pkgs.lib.optionalString (feature != "") " --features ${feature}";
              nativeBuildInputs = with pkgs; [ pkg-config cmake rustPlatform.bindgenHook ]
                ++ lib.optionals cudaEnabled [ cudaPackages.cuda_nvcc autoAddDriverRunpath ];
              buildInputs = [ pkgs.openssl ] ++ pkgs.lib.optionals cudaEnabled (with cudaPackages; [
                cccl cuda_cudart cuda_culibos cuda_nvrtc libcublas libcurand
              ]);
            } // cudaBuildEnv;

            cargoArtifacts = craneLib.buildDepsOnly commonArgs;

            bin = craneLib.buildPackage (commonArgs // {
              inherit cargoArtifacts;
            });

            checks = {
              inherit bin;
              wt-clippy = craneLib.cargoClippy (commonArgs // {
                inherit cargoArtifacts;
                cargoClippyExtraArgs = "--all-targets -- --deny warnings";
              });
            };
          in
          {
            package = bin;
            inherit checks;
            shell = craneLib.devShell ({
              inherit checks;
              packages = with pkgs; [ rust-analyzer ];
              shellHook = pkgs.lib.optionalString cudaEnabled ''
                # NixOS exposes the installed NVIDIA driver outside the Nix store.
                export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              '';
            } // cudaBuildEnv);
          };
        cpu = mkBackend "";
        cuda = mkBackend "cuda";
        metal = mkBackend "metal";
        # Host driver paths are visible with --impure (also used by .envrc).
        hasCudaDriver = pkgs.stdenv.hostPlatform.isLinux && (
          builtins.pathExists /run/opengl-driver/lib/libcuda.so
          || builtins.pathExists /proc/driver/nvidia/version
        );
        defaultShell =
          if pkgs.stdenv.hostPlatform.isDarwin then metal.shell
          else if hasCudaDriver then cuda.shell
          else cpu.shell;
      in
      {
        packages = {
          default = cpu.package;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          cuda = cuda.package;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
          metal = metal.package;
        };

        checks = cpu.checks;

        devShells = {
          default = defaultShell;
          cpu = cpu.shell;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          cuda = cuda.shell;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
          metal = metal.shell;
        };
      }
    );
}
