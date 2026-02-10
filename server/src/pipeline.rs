use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis, concatenate, s};
use ort::session::Session;
use ort::value::TensorRef;
use rand::rng;
use rand_distr::{Distribution, StandardNormal};

const SR: f32 = 24_000.0;
const HOP: f32 = 3_200.0;
const D: usize = 64;
const STEPS: usize = 4;
const CFG: f32 = 2.0;

pub struct Pipeline {
    codec_enc: Session,
    cond_enc: Session,
    denoiser: Session,
    codec_dec: Session,
}

impl Pipeline {
    pub fn load() -> Result<Self> {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets");
        Ok(Self {
            codec_enc: Session::builder()?
                .commit_from_file(format!("{base}/codec/encoder.onnx"))?,
            cond_enc: Session::builder()?
                .commit_from_file(format!("{base}/dmd/condition_encoder.onnx"))?,
            denoiser: Session::builder()?.commit_from_file(format!("{base}/dmd/denoiser.onnx"))?,
            codec_dec: Session::builder()?
                .commit_from_file(format!("{base}/codec/decoder.onnx"))?,
        })
    }

    pub fn synthesize(
        &mut self,
        ref_audio: &[f32],
        token_ids: &[i64],
        duration_sec: f32,
    ) -> Result<Vec<f32>> {
        let seq_len = ((duration_sec * SR) / HOP).ceil().max(1.0) as usize;

        let ref_latents = self.codec_encode(ref_audio)?;
        let ref_len = Array1::from_vec(vec![ref_latents.shape()[1] as i64]);

        let ph = Array2::from_shape_vec((1, token_ids.len()), token_ids.to_vec())?;
        let ph_mask = Array2::from_elem((1, token_ids.len()), true);
        let zero_ph = Array2::<i64>::zeros((1, token_ids.len()));
        let false_mask = Array2::from_elem((1, token_ids.len()), false);

        let cond = self.cond_encode(&ref_latents, &ref_len, &ph, &ph_mask)?;
        let uncond = self.cond_encode(&ref_latents, &ref_len, &zero_ph, &false_mask)?;
        let kv = BatchedKV::new(&cond, &uncond, &ph_mask, &false_mask);

        let rope = rope_freqs(seq_len);
        let mask2 = Array2::from_elem((2, seq_len), true);
        let mut x = Array3::<f32>::zeros((1, seq_len, D));

        for t in linspace(1.0, 0.0, STEPS) {
            let (a, sig) = alpha_sigma(t);
            let noise = randn(1, seq_len, D);
            let xt = &x * a + &noise * sig;
            let xt2 = concatenate(Axis(0), &[xt.view(), xt.view()])?;
            let tarr = Array1::from_vec(vec![t, t]);

            let vel = self.denoise(&xt2, &mask2, &tarr, &kv, &rope)?;
            let vc = vel.slice(s![0..1, .., ..]);
            let vu = vel.slice(s![1..2, .., ..]);
            let v = &vc + (&vc - &vu) * CFG;
            x = &xt * a - &v * sig;
        }

        self.codec_decode(&x)
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
            k_ref: extract4(&out[0])?,
            v_ref: extract4(&out[1])?,
            ref_mask: extract_bool2(&out[2])?,
            k_text: extract4(&out[3])?,
            v_text: extract4(&out[4])?,
        })
    }

    fn denoise(
        &mut self,
        xt: &Array3<f32>,
        mask: &Array2<bool>,
        t: &Array1<f32>,
        kv: &BatchedKV,
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
    k_ref: Array4<f32>,
    v_ref: Array4<f32>,
    ref_mask: Array2<bool>,
    k_text: Array4<f32>,
    v_text: Array4<f32>,
}

struct BatchedKV {
    k_ref: Array4<f32>,
    v_ref: Array4<f32>,
    ref_mask: Array2<bool>,
    k_text: Array4<f32>,
    v_text: Array4<f32>,
    ph_mask: Array2<bool>,
}

impl BatchedKV {
    fn new(c: &CondKV, u: &CondKV, cm: &Array2<bool>, um: &Array2<bool>) -> Self {
        let c4 =
            |a: &Array4<f32>, b: &Array4<f32>| concatenate(Axis(0), &[a.view(), b.view()]).unwrap();
        let c2 = |a: &Array2<bool>, b: &Array2<bool>| {
            concatenate(Axis(0), &[a.view(), b.view()]).unwrap()
        };
        Self {
            k_ref: c4(&c.k_ref, &u.k_ref),
            v_ref: c4(&c.v_ref, &u.v_ref),
            ref_mask: c2(&c.ref_mask, &u.ref_mask),
            k_text: c4(&c.k_text, &u.k_text),
            v_text: c4(&c.v_text, &u.v_text),
            ph_mask: c2(cm, um),
        }
    }
}

fn extract3(val: &ort::value::DynValue) -> Result<Array3<f32>> {
    let (shape, data) = val.try_extract_tensor::<f32>()?;
    Ok(Array3::from_shape_vec(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        data.to_vec(),
    )?)
}

fn extract4(val: &ort::value::DynValue) -> Result<Array4<f32>> {
    let (shape, data) = val.try_extract_tensor::<f32>()?;
    Ok(Array4::from_shape_vec(
        (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
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
