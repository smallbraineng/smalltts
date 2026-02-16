use std::sync::Arc;
use std::time::Instant;

use alloy_signer_local::PrivateKeySigner;
use x402_chain_eip155::{V1Eip155ExactClient, V2Eip155ExactClient};
use x402_reqwest::{ReqwestWithPayments, ReqwestWithPaymentsBuild, X402Client};

const DEFAULT_SERVER: &str = "https://smalltts-service.smallbrain.xyz";

fn sine_wav(duration_sec: f32, sample_rate: u32) -> Vec<u8> {
    let n = (duration_sec * sample_rate as f32) as usize;
    let mut buf = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        w.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    w.finalize().unwrap();
    buf.into_inner()
}

fn build_multipart(wav: &[u8], text: &str) -> (String, Vec<u8>) {
    let boundary = "----x402testboundary";
    let mut body = Vec::new();

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"audio\"; filename=\"ref.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"text\"\r\n\r\n");
    body.extend_from_slice(text.as_bytes());
    body.extend_from_slice(b"\r\n");

    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let content_type = format!("multipart/form-data; boundary={boundary}");
    (content_type, body)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "warn,x402_reqwest=debug,x402_chain_eip155=debug"
                    .parse()
                    .unwrap()
            }),
        )
        .init();

    let private_key =
        std::env::var("PRIVATE_KEY").expect("PRIVATE_KEY env var required (hex-encoded)");
    let server = std::env::var("SERVER_URL").unwrap_or_else(|_| DEFAULT_SERVER.to_string());
    let duration: f64 = std::env::var("DURATION")
        .unwrap_or_else(|_| "2.0".to_string())
        .parse()
        .expect("DURATION must be a number");

    println!("server:   {server}");
    println!("duration: {duration}s");

    let signer: Arc<PrivateKeySigner> = Arc::new(private_key.parse()?);
    println!("wallet:   {:?}", signer.address());

    let x402 = X402Client::new()
        .register(V1Eip155ExactClient::new(signer.clone()))
        .register(V2Eip155ExactClient::new(signer));

    let client = reqwest::Client::new().with_payments(x402).build();

    let wav = sine_wav(1.0, 24_000);
    let (content_type, body) = build_multipart(
        &wav,
        "Hello world, this is a test of the x402 payment protocol.",
    );

    let url = format!("{server}/synthesize?duration={duration}");
    println!("POST {url}");

    let t0 = Instant::now();
    let resp = client
        .post(&url)
        .header("content-type", content_type)
        .body(body)
        .send()
        .await?;
    let elapsed = t0.elapsed();

    println!("status:   {}", resp.status());
    println!("elapsed:  {:.1}s", elapsed.as_secs_f64());

    for (k, v) in resp.headers() {
        if k.as_str().starts_with("x402")
            || k.as_str().starts_with("payment")
            || k.as_str() == "x-error"
        {
            println!("header:   {k}: {}", v.to_str().unwrap_or("<binary>"));
        }
    }

    if resp.status().is_success() {
        let bytes = resp.bytes().await?;
        println!("response: {} bytes", bytes.len());
        std::fs::write("output.wav", &bytes)?;
        println!("saved to: output.wav");
    } else {
        let body = resp.text().await?;
        if !body.is_empty() {
            eprintln!("body:     {body}");
        }

        eprintln!("hint:     check that wallet has USDC on Base (eip155:8453)");
    }

    Ok(())
}
