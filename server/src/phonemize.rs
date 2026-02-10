use anyhow::{Context, Result, bail};
use tokio::process::Command;

pub async fn phonemize(text: &str) -> Result<Vec<i64>> {
    let output = Command::new("uv")
        .args(["run", "python", "scripts/phonemize.py", text])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
        .env(
            "PHONEMIZER_ESPEAK_LIBRARY",
            "/opt/homebrew/lib/libespeak.dylib",
        )
        .output()
        .await
        .context("failed to spawn phonemize process")?;

    if !output.status.success() {
        bail!(
            "phonemize exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).context("failed to parse phonemize output")
}
