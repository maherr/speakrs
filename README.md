# speakrs (maherr fork)

A thin CLI wrapper around [avencera/speakrs](https://github.com/avencera/speakrs) 0.3.1, plus a vendored copy of the library patched with an `ort::MIGraphX` execution-provider mode for AMD GPUs.

## What this is

- **Root crate:** `speakrs-cli` — produces the `speakrs-cli` binary used by [Witness](https://github.com/maherr/onnxruntime-migraphx-rdna4/tree/main/witness). Wraps the library's diarization pipeline, handles CLI I/O, writes RTTM or JSON.
- **`vendor/speakrs/`:** speakrs 0.3.1 (Apache-2.0, Praveen Perera), vendored via `cargo vendor` and patched to add the `migraphx` Cargo feature gating an `ExecutionMode::MiGraphX` path. See `src/inference/embedding/session.rs` for `with_migraphx_mode()`.
- **`src/bin/ort-migraphx-repro.rs`:** minimal reproducer for the `_dl_fini` heap-corruption race described in the [blog post](https://maherr.dev/rdna4-missing-rung/). Not built by default.

## Build

```sh
cargo build --release --bin speakrs-cli
```

Requires ORT 2.0.0-rc.12 with MIGraphX support — for RDNA 4, that means [`maherr/onnxruntime-migraphx-rdna4`](https://github.com/maherr/onnxruntime-migraphx-rdna4)'s patched build. The [install script](https://github.com/maherr/onnxruntime-migraphx-rdna4/blob/main/scripts/install-witness-stack.sh) in that repo runs this `cargo build` automatically.

## Why this isn't a regular git-history fork

I vendored speakrs with `cargo vendor` rather than forking the repo, because I only needed one feature (`migraphx`) gated on conditionally-compiled code paths — not an invasive rewrite. Upstream git history is lost, but `vendor/speakrs/Cargo.toml.orig` and `vendor/speakrs/LICENSE` preserve attribution and license. If a cleaner upstream PR lands, I'd switch to that.

## License

`speakrs-cli` wrapper: MIT. Vendored speakrs: Apache-2.0 (see `vendor/speakrs/LICENSE`).
