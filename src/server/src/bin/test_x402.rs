use std::sync::Arc;
use std::time::Instant;

use alloy_signer_local::PrivateKeySigner;
use x402_chain_eip155::{V1Eip155ExactClient, V2Eip155ExactClient};
use x402_reqwest::{ReqwestWithPayments, ReqwestWithPaymentsBuild, X402Client};

const DEFAULT_SERVER: &str = "https://smalltts-service.smallbrain.xyz";

fn speech_wav() -> Vec<u8> {
    let output = std::process::Command::new("espeak")
        .args([
            "This is a reference voice for cloning.",
            "-w",
            "/tmp/smalltts_ref.wav",
        ])
        .output()
        .expect("espeak not found -- install with: brew install espeak / apt install espeak");
    assert!(
        output.status.success(),
        "espeak failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read("/tmp/smalltts_ref.wav").expect("failed to read espeak output")
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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        .init();

    let private_key =
        std::env::var("PRIVATE_KEY").expect("PRIVATE_KEY env var required (hex-encoded)");
    let server = std::env::var("SERVER_URL").unwrap_or_else(|_| DEFAULT_SERVER.to_string());
    let duration: f64 = std::env::var("DURATION")
        .unwrap_or_else(|_| "3.0".to_string())
        .parse()
        .expect("DURATION must be a number");
    let text = std::env::var("TEXT").unwrap_or_else(|_| {
        "Hello world, this is a test of the x402 payment protocol.".to_string()
    });

    println!("server:   {server}");
    println!("duration: {duration}s");
    println!("text:     {text}");

    let signer: Arc<PrivateKeySigner> = Arc::new(private_key.parse()?);
    println!("wallet:   {:?}", signer.address());

    let x402 = X402Client::new()
        .register(V1Eip155ExactClient::new(signer.clone()))
        .register(V2Eip155ExactClient::new(signer));

    let client = reqwest::Client::new().with_payments(x402).build();

    let wav = if let Ok(path) = std::env::var("REF_WAV") {
        println!("ref:      {path}");
        std::fs::read(&path)?
    } else {
        println!("ref:      espeak (generated)");
        speech_wav()
    };

    let (content_type, body) = build_multipart(&wav, &text);

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
