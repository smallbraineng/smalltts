use smalltts_server::pipeline::{HOP, Pipeline, SR, Timing};

const WARMUP: usize = 1;
const RUNS: usize = 3;
const REF_AUDIO_SEC: f32 = 2.0;
const PHONEME_LEN: usize = 30;

fn sine_wave(duration_sec: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_sec * sample_rate as f32) as usize;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn main() -> anyhow::Result<()> {
    println!("smalltts-server benchmark");
    println!();

    let mut pipe = Pipeline::load()?;
    println!("pipeline loaded");

    let ref_audio = sine_wave(REF_AUDIO_SEC, SR as u32);
    let tokens: Vec<i64> = (1..=PHONEME_LEN as i64).collect();
    let durations = [2.0_f32, 5.0, 10.0];
    let batches = [1usize, 2, 4, 8];

    for &batch in &batches {
        println!();
        println!("batch = {} (sequential)", batch);
        println!(
            "  {:>6}  {:>9}  {:>9}  {:>9}  {:>9}  {:>10}  {:>6}",
            "dur(s)", "codec_enc", "cond_enc", "denoise", "codec_dec", "total(ms)", "RTF"
        );
        println!(
            "  {:>6}  {:>9}  {:>9}  {:>9}  {:>9}  {:>10}  {:>6}",
            "------", "---------", "---------", "---------", "---------", "----------", "------"
        );

        for &dur in &durations {
            // warmup
            for _ in 0..WARMUP {
                for _ in 0..batch {
                    let _ = pipe.synthesize_timed(&ref_audio, &tokens, dur)?;
                }
            }

            let mut timings = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let mut batch_timing = Timing {
                    codec_enc_ms: 0.0,
                    cond_enc_ms: 0.0,
                    denoise_ms: 0.0,
                    codec_dec_ms: 0.0,
                    total_ms: 0.0,
                };
                let wall_start = std::time::Instant::now();
                for _ in 0..batch {
                    let (_, t) = pipe.synthesize_timed(&ref_audio, &tokens, dur)?;
                    batch_timing.codec_enc_ms += t.codec_enc_ms;
                    batch_timing.cond_enc_ms += t.cond_enc_ms;
                    batch_timing.denoise_ms += t.denoise_ms;
                    batch_timing.codec_dec_ms += t.codec_dec_ms;
                }
                batch_timing.total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
                timings.push(batch_timing);
            }

            let avg = avg_timing(&timings);
            let seq_len = ((dur * SR) / HOP).ceil() as usize;
            let audio_sec = seq_len as f64 * HOP as f64 / SR as f64;
            let rtf = avg.total_ms / 1000.0 / (audio_sec * batch as f64);

            println!(
                "  {:>6.1}  {:>9.1}  {:>9.1}  {:>9.1}  {:>9.1}  {:>10.1}  {:>6.3}",
                dur,
                avg.codec_enc_ms / batch as f64,
                avg.cond_enc_ms / batch as f64,
                avg.denoise_ms / batch as f64,
                avg.codec_dec_ms / batch as f64,
                avg.total_ms,
                rtf
            );
        }
    }

    println!();
    Ok(())
}

fn avg_timing(ts: &[Timing]) -> Timing {
    let n = ts.len() as f64;
    Timing {
        codec_enc_ms: ts.iter().map(|t| t.codec_enc_ms).sum::<f64>() / n,
        cond_enc_ms: ts.iter().map(|t| t.cond_enc_ms).sum::<f64>() / n,
        denoise_ms: ts.iter().map(|t| t.denoise_ms).sum::<f64>() / n,
        codec_dec_ms: ts.iter().map(|t| t.codec_dec_ms).sum::<f64>() / n,
        total_ms: ts.iter().map(|t| t.total_ms).sum::<f64>() / n,
    }
}
