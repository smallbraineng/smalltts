use std::sync::Arc;

use anyhow::Result;
use axum::{
    Router,
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
};
use tokio::sync::Mutex;
use tower_http::limit::RequestBodyLimitLayer;
use tracing_subscriber::EnvFilter;

mod audio;
mod phonemize;
mod pipeline;

type SharedPipeline = Arc<Mutex<pipeline::Pipeline>>;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let pipe = pipeline::Pipeline::load()?;
    tracing::info!("pipeline loaded");

    let state: SharedPipeline = Arc::new(Mutex::new(pipe));
    let app = Router::new()
        .route("/synthesize", axum::routing::post(synthesize))
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn synthesize(
    State(pipe): State<SharedPipeline>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;
    let mut duration: Option<f32> = None;

    while let Some(field) = multipart.next_field().await.map_err(bad)? {
        match field.name().unwrap_or("") {
            "audio" => audio_bytes = Some(field.bytes().await.map_err(bad)?.to_vec()),
            "text" => text = Some(field.text().await.map_err(bad)?),
            "duration" => {
                let s = field.text().await.map_err(bad)?;
                duration = Some(
                    s.parse::<f32>()
                        .map_err(|_| e(StatusCode::BAD_REQUEST, "invalid duration"))?,
                );
            }
            _ => {}
        }
    }

    let audio_bytes = audio_bytes.ok_or_else(|| e(StatusCode::BAD_REQUEST, "missing 'audio'"))?;
    let text = text.ok_or_else(|| e(StatusCode::BAD_REQUEST, "missing 'text'"))?;
    let duration = duration.ok_or_else(|| e(StatusCode::BAD_REQUEST, "missing 'duration'"))?;

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

fn bad<E: std::fmt::Display>(err: E) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, err.to_string())
}

fn e(code: StatusCode, msg: &str) -> (StatusCode, String) {
    (code, msg.to_string())
}
