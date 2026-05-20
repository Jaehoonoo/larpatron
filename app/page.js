"use client";

import { useState, useRef } from "react";
import { pipeline, env } from "@huggingface/transformers";

const CHUNK_MS = 3000; // 3s chunks — enough audio for Whisper to be accurate
const MODEL = "Xenova/whisper-small"; // small > tiny for Korean accuracy, still fast on A18

export default function TranslatorApp() {
  const [status, setStatus] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [lines, setLines] = useState([]);
  const [ready, setReady] = useState(false);
  const [listening, setListening] = useState(false);
  const transcriber = useRef(null);
  const intervalRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const isProcessing = useRef(false);

  const downloadModel = async () => {
    try {
      setStatus(
        "Downloading Whisper (this may take a minute on first load)...",
      );
      env.backends.onnx.wasm.numThreads = 1;
      transcriber.current = await pipeline(
        "automatic-speech-recognition",
        MODEL,
        {
          device: "wasm",
          dtype: "fp32",
          progress_callback: (data) => {
            if (data.status === "progress")
              setProgress(Math.round(data.progress || 0));
          },
        },
      );
      setReady(true);
      setStatus("Ready — tap Start to begin");
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  };

  const startListening = async () => {
    streamRef.current = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
      },
    });
    audioContextRef.current = new AudioContext({ sampleRate: 16000 });
    setListening(true);
    setStatus("Listening...");
    setLines([]);

    const captureChunk = () => {
      if (isProcessing.current) return; // skip if last chunk still running
      const recorder = new MediaRecorder(streamRef.current);
      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = async () => {
        isProcessing.current = true;
        try {
          const blob = new Blob(chunks, { type: recorder.mimeType });
          const arrayBuffer = await blob.arrayBuffer();
          const audioBuffer =
            await audioContextRef.current.decodeAudioData(arrayBuffer);
          const audioData = audioBuffer.getChannelData(0);

          // Whisper translate task: Korean audio → English text directly, no LLM needed
          const result = await transcriber.current(audioData, {
            language: "korean",
            task: "translate",
          });

          const text = result.text.trim();
          if (text.length > 0) {
            setLines((prev) => {
              const next = [...prev, text];
              return next.slice(-6); // keep last 6 lines visible
            });
          }
        } catch (err) {
          console.error("Chunk processing failed:", err);
        } finally {
          isProcessing.current = false;
        }
      };
      recorder.start();
      setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, CHUNK_MS);
    };

    captureChunk();
    intervalRef.current = setInterval(captureChunk, CHUNK_MS);
  };

  const stopListening = () => {
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioContextRef.current?.close();
    setListening(false);
    setStatus("Stopped");
  };

  return (
    <div
      style={{
        padding: "1.5rem",
        fontFamily: "sans-serif",
        maxWidth: "500px",
        margin: "0 auto",
      }}
    >
      <h2 style={{ marginBottom: "0.25rem" }}>🎧 Korean → English</h2>
      <p style={{ margin: "0 0 0.75rem", color: "#555", fontSize: "0.9rem" }}>
        Status: <strong>{status}</strong>
      </p>

      {!ready && (
        <>
          <progress
            value={progress}
            max="100"
            style={{ width: "100%", height: "16px", marginBottom: "1rem" }}
          />
          <button
            onClick={downloadModel}
            style={{
              padding: "0.75rem 1.5rem",
              fontSize: "1rem",
              cursor: "pointer",
              width: "100%",
            }}
          >
            Download Model (Wi-Fi recommended)
          </button>
        </>
      )}

      {ready && !listening && (
        <button
          onClick={startListening}
          style={{
            padding: "0.75rem 1.5rem",
            fontSize: "1rem",
            cursor: "pointer",
            backgroundColor: "#0070f3",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            width: "100%",
          }}
        >
          ▶ Start Listening
        </button>
      )}

      {listening && (
        <button
          onClick={stopListening}
          style={{
            padding: "0.75rem 1.5rem",
            fontSize: "1rem",
            cursor: "pointer",
            backgroundColor: "#e00",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            width: "100%",
          }}
        >
          ■ Stop
        </button>
      )}

      <div
        style={{
          marginTop: "1.5rem",
          background: "#000",
          padding: "1rem 1.25rem",
          borderRadius: "12px",
          minHeight: "160px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          gap: "0.4rem",
        }}
      >
        {lines.length === 0 ? (
          <p style={{ color: "#555", fontSize: "1rem", margin: 0 }}>
            Translation will appear here...
          </p>
        ) : (
          lines.map((line, i) => (
            <p
              key={i}
              style={{
                margin: 0,
                fontSize: i === lines.length - 1 ? "1.25rem" : "0.95rem",
                color: i === lines.length - 1 ? "#fff" : "#888",
                transition: "all 0.2s",
              }}
            >
              {line}
            </p>
          ))
        )}
      </div>
    </div>
  );
}
