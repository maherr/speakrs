use ndarray::{Array1, s};
use ort::value::TensorRef;

use crate::profiling;

use super::{EmbeddingModel, EmbeddingProfileTag, select_mask};

impl EmbeddingModel {
    /// Extract a speaker embedding from raw audio with a uniform mask
    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let weights = vec![1.0; self.meta.mask_frames];
        self.embed_single(audio, &weights, None)
    }

    /// Extract a speaker embedding weighted by a segmentation mask
    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        self.embed_masked_with_tag(audio, mask, clean_mask, None)
    }

    pub(crate) fn embed_masked_with_tag(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
        profile_tag: Option<EmbeddingProfileTag>,
    ) -> Result<Array1<f32>, ort::Error> {
        let used_mask = select_mask(mask, clean_mask, audio.len(), self.meta.min_num_samples);
        self.embed_single(audio, used_mask, profile_tag)
    }

    fn embed_single(
        &mut self,
        audio: &[f32],
        weights: &[f32],
        profile_tag: Option<EmbeddingProfileTag>,
    ) -> Result<Array1<f32>, ort::Error> {
        let copy_len = audio.len().min(self.meta.window_samples);
        self.buffers
            .waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        if copy_len < self.meta.window_samples {
            self.buffers
                .waveform_buffer
                .slice_mut(s![0, 0, copy_len..])
                .fill(0.0);
        }
        self.prepare_single_weights(weights);

        let waveform_tensor = TensorRef::from_array_view(self.buffers.waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.buffers.weights_buffer.view())?;
        let call_id = profiling::next_call_id();
        let dispatch_start_ns = profiling::now_ns();
        let outputs = self
            .ort
            .session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let dispatch_return_ns = profiling::now_ns();
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let values = data.to_vec();
        let readback_done_ns = profiling::now_ns();
        profiling::emit_embedding_call(
            "masked_single",
            call_id,
            1,
            profile_tag.map(|tag| tag.chunk_idx),
            profile_tag.map(|tag| tag.speaker_idx),
            dispatch_start_ns,
            dispatch_return_ns,
            readback_done_ns,
        );
        Ok(Array1::from_vec(values))
    }
}
