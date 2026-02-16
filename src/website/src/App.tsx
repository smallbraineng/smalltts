import { useState, useRef, useCallback } from "react";
import { usePrivy, useWallets, useX402Fetch } from "@privy-io/react-auth";

const API_URL =
  import.meta.env.VITE_API_URL || "https://smalltts-service.smallbrain.xyz";
const RATE_PER_MIN = 0.05;

export default function App() {
  const { ready, authenticated, login, logout, user } = usePrivy();
  const { wallets } = useWallets();
  const { wrapFetchWithPayment } = useX402Fetch();

  const [text, setText] = useState(
    "Hello, this is a test of the smalltts text to speech system."
  );
  const [duration, setDuration] = useState(3);
  const [refAudio, setRefAudio] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const cost = ((duration / 60) * RATE_PER_MIN).toFixed(4);

  const handleSynthesize = useCallback(async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setAudioUrl(null);

    try {
      const wallet = wallets[0];
      if (!wallet) {
        setError("no wallet connected");
        return;
      }

      const fetchWithPayment = wrapFetchWithPayment({
        walletAddress: wallet.address,
        fetch,
      });

      const form = new FormData();
      if (refAudio) {
        form.append("audio", refAudio, refAudio.name);
      } else {
        const silence = createSilentWav();
        form.append("audio", new Blob([silence], { type: "audio/wav" }), "ref.wav");
      }
      form.append("text", text);

      const resp = await fetchWithPayment(
        `${API_URL}/synthesize?duration=${duration}`,
        { method: "POST", body: form }
      );

      if (!resp.ok) {
        const body = await resp.text();
        setError(`${resp.status}: ${body || resp.statusText}`);
        return;
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [text, duration, refAudio, wallets, wrapFetchWithPayment]);

  if (!ready) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-sm text-gray-500">loading...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white text-black">
      <div className="max-w-lg mx-auto px-4 py-12">
        {/* header */}
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-lg font-bold tracking-tight">smalltts x402</h1>
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/smallbraineng/smalltts"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-black"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
            </a>
            {authenticated ? (
              <button
                onClick={logout}
                className="text-xs border border-gray-300 px-3 py-1.5 hover:bg-gray-50"
              >
                {user?.wallet?.address?.slice(0, 6)}...
                {user?.wallet?.address?.slice(-4)} &middot; disconnect
              </button>
            ) : (
              <button
                onClick={login}
                className="text-xs border border-black px-3 py-1.5 hover:bg-gray-100"
              >
                connect wallet
              </button>
            )}
          </div>
        </div>
        <p className="text-sm text-gray-500 mb-10">
          a superfast text to speech model for expressive realtime characters.
          <br />
          can [cough], [groan] and even [laughter].
        </p>

        {!authenticated ? (
          <p className="text-sm text-gray-500">
            connect a wallet to get started. you need USDC on Base.
          </p>
        ) : (
          <div className="space-y-6">
            {/* text input */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                className="w-full border border-gray-300 px-3 py-2 text-sm resize-none focus:outline-none focus:border-black"
                placeholder="enter text to synthesize..."
              />
            </div>

            {/* duration */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                duration: {duration}s &middot; ${cost}
              </label>
              <input
                type="range"
                min={1}
                max={10}
                step={0.5}
                value={duration}
                onChange={(e) => setDuration(parseFloat(e.target.value))}
                className="w-full accent-black"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                <span>1s</span>
                <span>10s</span>
              </div>
            </div>

            {/* reference audio */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                reference audio{" "}
                <span className="text-gray-400">(optional, for voice cloning)</span>
              </label>
              <div className="flex items-center gap-2">
                <label className="text-xs border border-gray-300 px-3 py-1.5 cursor-pointer hover:bg-gray-50">
                  {refAudio ? refAudio.name : "choose file"}
                  <input
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={(e) => setRefAudio(e.target.files?.[0] || null)}
                  />
                </label>
                {refAudio && (
                  <button
                    onClick={() => setRefAudio(null)}
                    className="text-xs text-gray-400 hover:text-black"
                  >
                    clear
                  </button>
                )}
              </div>
            </div>

            {/* submit */}
            <button
              onClick={handleSynthesize}
              disabled={loading || !text.trim()}
              className="w-full border border-black px-4 py-2.5 text-sm font-medium hover:bg-black hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "generating..." : `synthesize · $${cost}`}
            </button>

            {/* error */}
            {error && (
              <p className="text-xs text-red-600 bg-red-50 border border-red-200 px-3 py-2">
                {error}
              </p>
            )}

            {/* audio player */}
            {audioUrl && (
              <div className="border border-gray-200 p-4 space-y-3">
                <p className="text-xs text-gray-500">output</p>
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  controls
                  autoPlay
                  className="w-full"
                />
                <a
                  href={audioUrl}
                  download="smalltts-output.wav"
                  className="inline-block text-xs border border-gray-300 px-3 py-1.5 hover:bg-gray-50"
                >
                  download wav
                </a>
              </div>
            )}

            <p className="text-xs text-gray-400 mt-4">
              powered by x402. $0.05/min of audio. payment in USDC on Base.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function createSilentWav(): ArrayBuffer {
  const sampleRate = 24000;
  const numSamples = sampleRate; // 1 second
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++)
      view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);

  // Generate a very short tone burst so the codec has something to encode
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    const sample = Math.sin(2 * Math.PI * 440 * t) * 0.3;
    view.setInt16(44 + i * 2, sample * 32767, true);
  }

  return buffer;
}
