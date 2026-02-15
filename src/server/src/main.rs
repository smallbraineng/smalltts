use std::sync::Arc;

use alloy_primitives::Address;
use anyhow::Result;
use axum::{
    Router,
    extract::{Multipart, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use tokio::sync::Mutex;
use tower_http::limit::RequestBodyLimitLayer;
use tracing_subscriber::EnvFilter;
use x402_axum::X402Middleware;
use x402_chain_eip155::{KnownNetworkEip155, V2Eip155Exact};
use x402_types::networks::USDC;

mod audio;
mod phonemize;
mod pipeline;

type SharedPipeline = Arc<Mutex<pipeline::Pipeline>>;

// $0.05 per minute = 50000 USDC smallest units (6 decimals) per 60 seconds
const RATE_PER_SECOND: u64 = 833; // ceil(50000 / 60)

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let facilitator_url = std::env::var("FACILITATOR_URL")
        .unwrap_or_else(|_| "https://facilitator.x402.rs".to_string());
    let payment_address: Address = std::env::var("PAYMENT_ADDRESS")
        .unwrap_or_else(|_| "0xBAc675C310721717Cd4A37F6cbeA1F081b1C2a07".to_string())
        .parse()
        .expect("invalid PAYMENT_ADDRESS");
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());

    let pipe = pipeline::Pipeline::load()?;
    tracing::info!("pipeline loaded");

    let x402 =
        X402Middleware::try_from(facilitator_url.as_str()).map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(
        "x402 facilitator: {}, rate: ${:.2}/min",
        x402.facilitator_url(),
        RATE_PER_SECOND as f64 * 60.0 / 1_000_000.0
    );

    let state: SharedPipeline = Arc::new(Mutex::new(pipe));
    let app = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route("/.well-known/x402", axum::routing::get(discovery))
        .route(
            "/synthesize",
            axum::routing::post(synthesize).layer(x402.with_dynamic_price(
                move |_headers, uri, _base_url| {
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
                            payment_address,
                            USDC::base().amount(amount),
                        )]
                    }
                },
            )),
        )
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Deserialize)]
struct SynthesizeParams {
    duration: f32,
}

async fn synthesize(
    State(pipe): State<SharedPipeline>,
    Query(params): Query<SynthesizeParams>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(bad)? {
        match field.name().unwrap_or("") {
            "audio" => audio_bytes = Some(field.bytes().await.map_err(bad)?.to_vec()),
            "text" => text = Some(field.text().await.map_err(bad)?),
            _ => {}
        }
    }

    let audio_bytes = audio_bytes.ok_or_else(|| e(StatusCode::BAD_REQUEST, "missing 'audio'"))?;
    let text = text.ok_or_else(|| e(StatusCode::BAD_REQUEST, "missing 'text'"))?;
    let duration = params.duration;

    let samples = audio::decode_and_resample(&audio_bytes, 24_000).map_err(|err| {
        e(
            StatusCode::BAD_REQUEST,
            &format!("audio decode failed: {err}"),
        )
    })?;

    let token_ids = phonemize::phonemize(&text).await.map_err(|err| {
        e(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("phonemize failed: {err}"),
        )
    })?;

    let output = {
        let mut p = pipe.lock().await;
        p.synthesize(&samples, &token_ids, duration)
            .map_err(|err| {
                e(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("inference failed: {err}"),
                )
            })?
    };

    let wav = audio::encode_wav(&output, 24_000).map_err(|err| {
        e(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("wav encode failed: {err}"),
        )
    })?;

    Ok((StatusCode::OK, [("content-type", "audio/wav")], wav))
}

async fn discovery() -> impl IntoResponse {
    let base = std::env::var("BASE_URL").unwrap_or_else(|_| "http://localhost:3000".into());
    axum::Json(serde_json::json!({
        "version": 1,
        "resources": [format!("{base}/synthesize")],
        "instructions": "# smalltts\n\nText-to-speech API. POST /synthesize?duration=N with multipart audio + text.\n\nPricing: $0.05/min of generated audio."
    }))
}

fn bad<E: std::fmt::Display>(err: E) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, err.to_string())
}

fn e(code: StatusCode, msg: &str) -> (StatusCode, String) {
    (code, msg.to_string())
}
