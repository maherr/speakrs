use std::path::Path;

use ort::ep;
use ort::session::Session;

use crate::inference::with_execution_mode;
use crate::profiling;

use super::{EmbeddingModel, ExecutionMode};

impl EmbeddingModel {
    pub(super) fn build_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session_with_graph(model_path, mode, false)
    }

    pub(super) fn build_session_with_graph(
        model_path: &Path,
        mode: ExecutionMode,
        cuda_graph: bool,
    ) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let builder = profiling::enable_ort_profiling(builder, model_path)?;
        let mut builder =
            if cuda_graph && matches!(mode, ExecutionMode::Cuda | ExecutionMode::CudaFast) {
                Self::with_cuda_graph_mode(builder)?
            } else if matches!(mode, ExecutionMode::MiGraphX) {
                Self::with_migraphx_mode(builder, model_path)?
            } else {
                with_execution_mode(builder, mode)?
            };
        builder.commit_from_file(model_path)
    }

    #[cfg(feature = "cuda")]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        Ok(builder.with_execution_providers([ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .with_conv_max_workspace(true)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .with_prefer_nhwc(true)
            .with_cuda_graph(true)
            .build()
            .error_on_failure()])?)
    }

    #[cfg(not(feature = "cuda"))]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        with_execution_mode(builder, ExecutionMode::Cpu)
    }

    fn with_migraphx_mode(
        builder: ort::session::builder::SessionBuilder,
        model_path: &Path,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        let cache_path = model_path.with_file_name(format!(
            "{}.migraphx.mxr",
            model_path.file_name().and_then(|name| name.to_str()).unwrap_or("model.onnx")
        ));
        let cache_path_str = cache_path.to_string_lossy().into_owned();
        let cache_exists = cache_path.is_file()
            && cache_path
                .metadata()
                .map(|meta| meta.len() > 0)
                .unwrap_or(false);
        let mut provider = ep::MIGraphX::default()
            .with_device_id(0)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested);
        provider = if cache_exists {
            provider.with_load_model(&cache_path_str)
        } else {
            provider.with_save_model(&cache_path_str)
        };

        Ok(builder.with_execution_providers([provider.build().error_on_failure()])?)
    }

    pub(super) fn build_fbank_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        let threads = std::thread::available_parallelism()
            .map(|count| count.get().min(4))
            .unwrap_or(1);
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let builder = profiling::enable_ort_profiling(builder, model_path)?;
        let mut builder = if matches!(mode, ExecutionMode::MiGraphX) {
            Self::with_migraphx_mode(builder, model_path)?
        } else {
            with_execution_mode(builder, mode)?
        };
        builder.commit_from_file(model_path)
    }

    pub(super) fn single_execution_mode(mode: ExecutionMode) -> ExecutionMode {
        match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => ExecutionMode::Cpu,
            _ => mode,
        }
    }

    pub(super) fn build_batched_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session(model_path, Self::single_execution_mode(mode))
    }
}

impl Drop for EmbeddingModel {
    fn drop(&mut self) {
        profiling::finish_ort_profile(&mut self.ort.session, "embedding-single");
        if let Some(session) = self.ort.primary_batched_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-batched");
        }
        if let Some(session) = self.ort.split_fbank_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-split-fbank");
        }
        if let Some(session) = self.ort.split_fbank_batched_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-split-fbank-batched");
        }
        if let Some(session) = self.ort.split_tail_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-split-tail");
        }
        if let Some(session) = self.ort.split_tail_batched_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-split-tail-batched");
        }
        if let Some(session) = self.ort.split_primary_tail_batched_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-split-tail-primary-batched");
        }
        if let Some(session) = self.ort.multi_mask_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-multi-mask");
        }
        if let Some(session) = self.ort.multi_mask_batched_session.as_mut() {
            profiling::finish_ort_profile(session, "embedding-multi-mask-batched");
        }
    }
}
