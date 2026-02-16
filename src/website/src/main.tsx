import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { PrivyProvider } from "@privy-io/react-auth";
import { base } from "viem/chains";
import App from "./App";
import "./index.css";

const appId = import.meta.env.VITE_PRIVY_APP_ID;
if (!appId) {
  throw new Error("VITE_PRIVY_APP_ID is required");
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <PrivyProvider
      appId={appId}
      config={{
        supportedChains: [base],
        defaultChain: base,
        appearance: {
          theme: "light",
          accentColor: "#000",
        },
      }}
    >
      <App />
    </PrivyProvider>
  </StrictMode>
);
