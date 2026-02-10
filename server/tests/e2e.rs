use std::net::TcpListener;
use std::time::Duration;

use alloy_primitives::address;
use reqwest::multipart;
use x402_chain_eip155::{KnownNetworkEip155, V2Eip155Exact};
use x402_types::networks::USDC;

const PAYMENT_ADDRESS: alloy_primitives::Address =
    address!("0xBAc675C310721717Cd4A37F6cbeA1F081b1C2a07");
const RATE_PER_SECOND: u64 = 833;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

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

async fn start_server(port: u16) {
    tokio::spawn(async move {
        let x402 = x402_axum::X402Middleware::try_from("https://facilitator.x402.rs").unwrap();

        let app = axum::Router::new()
            .route("/health", axum::routing::get(|| async { "ok" }))
            .route(
                "/.well-known/x402",
                axum::routing::get(move || async move {
                    axum::Json(serde_json::json!({
                        "version": 1,
                        "resources": [format!("http://127.0.0.1:{port}/synthesize")],
                        "instructions": "# smalltts\n\nText-to-speech API."
                    }))
                }),
            )
            .route(
                "/synthesize",
                axum::routing::post(|| async { (axum::http::StatusCode::OK, "synthesized") })
                    .layer(x402.with_dynamic_price(move |_headers, uri, _base_url| {
                        let duration = uri
                            .query()
                            .and_then(|q| {
                                q.split('&')
                                    .find_map(|p| p.strip_prefix("duration="))
                                    .and_then(|v| v.parse::<f64>().ok())
                            })
                            .unwrap_or(1.0)
                            .max(0.1);
                        let amount = (duration * RATE_PER_SECOND as f64).ceil() as u64;
                        async move {
                            vec![V2Eip155Exact::price_tag(
                                PAYMENT_ADDRESS,
                                USDC::base_sepolia().amount(amount),
                            )]
                        }
                    })),
            );

        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
            .await
            .unwrap();
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
}

#[tokio::test]
async fn test_health_returns_ok() {
    let port = free_port();
    start_server(port).await;

    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn test_unpaid_synthesize_returns_402() {
    let port = free_port();
    start_server(port).await;

    let wav = sine_wav(0.5, 24_000);
    let form = multipart::Form::new()
        .part(
            "audio",
            multipart::Part::bytes(wav)
                .file_name("ref.wav")
                .mime_str("audio/wav")
                .unwrap(),
        )
        .text("text", "hello world");

    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/synthesize?duration=2.0"))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 402);
}

#[tokio::test]
async fn test_402_response_has_payment_info() {
    let port = free_port();
    start_server(port).await;

    let wav = sine_wav(0.5, 24_000);
    let form = multipart::Form::new()
        .part(
            "audio",
            multipart::Part::bytes(wav)
                .file_name("ref.wav")
                .mime_str("audio/wav")
                .unwrap(),
        )
        .text("text", "hello world");

    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/synthesize?duration=2.0"))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 402);

    let pr_header = resp
        .headers()
        .get("payment-required")
        .expect("402 should have 'payment-required' header")
        .to_str()
        .unwrap()
        .to_string();

    use base64::Engine;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&pr_header)
        .expect("payment-required header should be valid base64");
    let body: serde_json::Value =
        serde_json::from_slice(&decoded).expect("decoded header should be valid json");

    assert!(
        body.get("accepts").is_some(),
        "payment-required should contain 'accepts': {body}"
    );
    let accepts = body["accepts"].as_array().unwrap();
    assert!(!accepts.is_empty());

    let first = &accepts[0];
    assert_eq!(first["scheme"], "exact");
    assert!(first.get("network").is_some());
    assert!(first.get("payTo").is_some());
}

#[tokio::test]
async fn test_price_scales_with_duration() {
    let port = free_port();
    start_server(port).await;

    let parse_amount = |resp_headers: &reqwest::header::HeaderMap| -> u64 {
        use base64::Engine;
        let pr = resp_headers
            .get("payment-required")
            .unwrap()
            .to_str()
            .unwrap();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(pr)
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        body["accepts"][0]["amount"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap()
    };

    let wav = sine_wav(0.5, 24_000);

    // 1 second
    let form = multipart::Form::new()
        .part(
            "audio",
            multipart::Part::bytes(wav.clone()).file_name("r.wav"),
        )
        .text("text", "hi");
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/synthesize?duration=1.0"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let amount_1s = parse_amount(resp.headers());

    // 5 seconds
    let form = multipart::Form::new()
        .part(
            "audio",
            multipart::Part::bytes(wav.clone()).file_name("r.wav"),
        )
        .text("text", "hi");
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/synthesize?duration=5.0"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let amount_5s = parse_amount(resp.headers());

    assert_eq!(amount_1s, RATE_PER_SECOND); // 1s = 5000
    assert_eq!(amount_5s, 5 * RATE_PER_SECOND); // 5s = 25000
    assert!(amount_5s > amount_1s);
}

#[tokio::test]
async fn test_402_without_body_still_returns_402() {
    let port = free_port();
    start_server(port).await;

    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/synthesize?duration=1.0"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 402);
}

#[tokio::test]
async fn test_wav_fixture_is_valid() {
    let wav = sine_wav(0.5, 24_000);
    assert!(wav.len() > 44);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
}

#[tokio::test]
async fn test_audio_decode_wav() {
    let wav = sine_wav(0.5, 24_000);
    let reader = hound::WavReader::new(std::io::Cursor::new(&wav)).unwrap();
    assert_eq!(reader.spec().sample_rate, 24_000);
    assert_eq!(reader.spec().channels, 1);
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .filter_map(Result::ok)
        .collect();
    assert_eq!(samples.len(), 12_000);
}

#[tokio::test]
async fn test_health_not_gated_by_x402() {
    let port = free_port();
    start_server(port).await;

    for _ in 0..3 {
        let resp = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}

#[tokio::test]
async fn test_discovery_endpoint() {
    let port = free_port();
    start_server(port).await;

    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/.well-known/x402"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["version"], 1);

    let resources = body["resources"].as_array().unwrap();
    assert!(!resources.is_empty());
    assert!(
        resources[0].as_str().unwrap().contains("/synthesize"),
        "resource should reference /synthesize"
    );

    assert!(body.get("instructions").is_some());
}
