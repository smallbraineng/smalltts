use std::time::Instant;

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array5};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::TensorRef;
use rand::rng;
use rand_distr::{Distribution, StandardNormal};

pub const SR: f32 = 24_000.0;
pub const HOP: f32 = 3_200.0;
const D: usize = 64;
const STEPS: usize = 4;

fn session_from_file(path: &str) -> Result<Session> {
    Ok(Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(path)?)
}

pub struct Pipeline {
    codec_enc: Session,
    cond_enc: Session,
    denoiser: Session,
    codec_dec: Session,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Timing {
    pub codec_enc_ms: f64,
    pub cond_enc_ms: f64,
    pub denoise_ms: f64,
    pub codec_dec_ms: f64,
    pub total_ms: f64,
}

impl Pipeline {
    pub fn load() -> Result<Self> {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets");
        Ok(Self {
            codec_enc: session_from_file(&format!("{base}/codec/encoder.onnx"))?,
            cond_enc: session_from_file(&format!("{base}/dmd/condition_encoder.onnx"))?,
            denoiser: session_from_file(&format!("{base}/dmd/denoiser.onnx"))?,
            codec_dec: session_from_file(&format!("{base}/codec/decoder.onnx"))?,
        })
    }

    pub fn synthesize(
        &mut self,
        ref_audio: &[f32],
        token_ids: &[i64],
        duration_sec: f32,
    ) -> Result<Vec<f32>> {
        let (audio, _) = self.synthesize_timed(ref_audio, token_ids, duration_sec)?;
        Ok(audio)
    }

    pub fn synthesize_timed(
        &mut self,
        ref_audio: &[f32],
        token_ids: &[i64],
        duration_sec: f32,
    ) -> Result<(Vec<f32>, Timing)> {
        let seq_len = ((duration_sec * SR) / HOP).ceil().max(1.0) as usize;
        let t0 = Instant::now();

        let t_enc = Instant::now();
        let ref_latents = self.codec_encode(ref_audio)?;
        let codec_enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

        let ref_len = Array1::from_vec(vec![ref_latents.shape()[1] as i64]);
        let ph = Array2::from_shape_vec((1, token_ids.len()), token_ids.to_vec())?;
        let ph_mask = Array2::from_elem((1, token_ids.len()), true);

        let t_cond = Instant::now();
        let kv = self.cond_encode(&ref_latents, &ref_len, &ph, &ph_mask)?;
        let cond_enc_ms = t_cond.elapsed().as_secs_f64() * 1000.0;

        let rope = rope_freqs(seq_len);
        let mask = Array2::from_elem((1, seq_len), true);
        let mut x = Array3::<f32>::zeros((1, seq_len, D));

        let t_den = Instant::now();
        for t in linspace(1.0, 0.0, STEPS) {
            let (a, sig) = alpha_sigma(t);
            let noise = randn(1, seq_len, D);
            let xt = &x * a + &noise * sig;
            let tarr = Array1::from_vec(vec![t]);
            let vel = self.denoise(&xt, &mask, &tarr, &kv, &rope)?;
            x = &xt * a - &vel * sig;
        }
        let denoise_ms = t_den.elapsed().as_secs_f64() * 1000.0;

        let t_dec = Instant::now();
        let audio = self.codec_decode(&x)?;
        let codec_dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok((
            audio,
            Timing {
                codec_enc_ms,
                cond_enc_ms,
                denoise_ms,
                codec_dec_ms,
                total_ms,
            },
        ))
    }

    fn codec_encode(&mut self, audio: &[f32]) -> Result<Array3<f32>> {
        let inp = Array3::from_shape_vec((1, 1, audio.len()), audio.to_vec())?;
        let out = self
            .codec_enc
            .run(ort::inputs![TensorRef::from_array_view(&inp)?])?;
        extract3(&out[0])
    }

    fn cond_encode(
        &mut self,
        lat: &Array3<f32>,
        len: &Array1<i64>,
        ph: &Array2<i64>,
        mask: &Array2<bool>,
    ) -> Result<CondKV> {
        let out = self.cond_enc.run(ort::inputs![
            TensorRef::from_array_view(lat)?,
            TensorRef::from_array_view(len)?,
            TensorRef::from_array_view(ph)?,
            TensorRef::from_array_view(mask)?,
        ])?;
        Ok(CondKV {
            k_ref: extract5(&out[0])?,
            v_ref: extract5(&out[1])?,
            ref_mask: extract_bool2(&out[2])?,
            k_text: extract5(&out[3])?,
            v_text: extract5(&out[4])?,
            ph_mask: mask.clone(),
        })
    }

    fn denoise(
        &mut self,
        xt: &Array3<f32>,
        mask: &Array2<bool>,
        t: &Array1<f32>,
        kv: &CondKV,
        rope: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let out = self.denoiser.run(ort::inputs![
            TensorRef::from_array_view(xt)?,
            TensorRef::from_array_view(mask)?,
            TensorRef::from_array_view(t)?,
            TensorRef::from_array_view(&kv.k_ref)?,
            TensorRef::from_array_view(&kv.v_ref)?,
            TensorRef::from_array_view(&kv.ref_mask)?,
            TensorRef::from_array_view(&kv.k_text)?,
            TensorRef::from_array_view(&kv.v_text)?,
            TensorRef::from_array_view(&kv.ph_mask)?,
            TensorRef::from_array_view(rope)?,
        ])?;
        extract3(&out[0])
    }

    fn codec_decode(&mut self, latents: &Array3<f32>) -> Result<Vec<f32>> {
        let out = self
            .codec_dec
            .run(ort::inputs![TensorRef::from_array_view(latents)?])?;
        let (_, data) = out[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

struct CondKV {
    k_ref: Array5<f32>,
    v_ref: Array5<f32>,
    ref_mask: Array2<bool>,
    k_text: Array5<f32>,
    v_text: Array5<f32>,
    ph_mask: Array2<bool>,
}

fn extract3(val: &ort::value::DynValue) -> Result<Array3<f32>> {
    let (shape, data) = val.try_extract_tensor::<f32>()?;
    Ok(Array3::from_shape_vec(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        data.to_vec(),
    )?)
}

fn extract5(val: &ort::value::DynValue) -> Result<Array5<f32>> {
    let (shape, data) = val.try_extract_tensor::<f32>()?;
    Ok(Array5::from_shape_vec(
        (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
            shape[4] as usize,
        ),
        data.to_vec(),
    )?)
}

fn extract_bool2(val: &ort::value::DynValue) -> Result<Array2<bool>> {
    let (shape, data) = val.try_extract_tensor::<bool>()?;
    Ok(Array2::from_shape_vec(
        (shape[0] as usize, shape[1] as usize),
        data.to_vec(),
    )?)
}

fn alpha_sigma(t: f32) -> (f32, f32) {
    let t = t.clamp(1e-5, 1.0 - 1e-5);
    let a2 = (std::f32::consts::FRAC_PI_2 * t).cos().powi(2);
    let lsnr = (a2 / (1.0 - a2)).ln() + 2.0 * 0.5_f32.ln();
    let asq = 1.0 / (1.0 + (-lsnr).exp());
    (asq.sqrt(), (1.0 - asq).sqrt())
}

fn rope_freqs(seq_len: usize) -> Array3<f32> {
    let half = D / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000_f32.powf(2.0 * i as f32 / D as f32))
        .collect();
    let mut out = Array3::<f32>::zeros((1, seq_len, D));
    for p in 0..seq_len {
        for (i, &f) in inv.iter().enumerate() {
            let v = p as f32 * f;
            out[[0, p, 2 * i]] = v;
            out[[0, p, 2 * i + 1]] = v;
        }
    }
    out
}

fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * i as f32 / (n - 1) as f32)
        .collect()
}

fn randn(b: usize, t: usize, d: usize) -> Array3<f32> {
    let mut r = rng();
    let data: Vec<f32> = (0..b * t * d)
        .map(|_| StandardNormal.sample(&mut r))
        .collect();
    Array3::from_shape_vec((b, t, d), data).unwrap()
}
