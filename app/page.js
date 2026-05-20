"use client";

import { useState, useRef } from "react";
import { pipeline, env } from "@huggingface/transformers";
import * as webllm from "@mlc-ai/web-llm";

export default function TranslatorApp() {
  const [status, setStatus] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [output, setOutput] = useState("");
  const [engine, setEngine] = useState(null);
  const mediaRecorder = useRef(null);
  const textBuffer = useRef("");
  const transcriber = useRef(null);

  const downloadModels = async () => {
    try {
      setStatus("Downloading STT (Whisper v4)...");
      env.backends.onnx.wasm.numThreads = 1;
      await pipeline(
        "automatic-speech-recognition",
        "onnx-community/whisper-tiny",
        {
          device: "wasm",
          progress_callback: (data) => {
            if (data.status === "progress") setProgress(data.progress || 0);
          },
        },
      );

      setStatus("Downloading LLM (Llama 3.2 1B)...");
      const localEngine = new webllm.MLCEngine();
      localEngine.setInitProgressCallback((report) =>
        setProgress(report.progress * 100),
      );
      await localEngine.reload("Llama-3.2-1B-Instruct-q4f16_1-MLC");

      setEngine(localEngine);
      setStatus("Ready");
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const startEngine = async () => {
    transcriber.current = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-tiny",
      { device: "wasm" },
    );

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder.current = new MediaRecorder(stream);

    mediaRecorder.current.ondataavailable = async (e) => {
      if (e.data.size > 0 && engine) {
        const arrayBuffer = await e.data.arrayBuffer();
        const audioContext = new AudioContext({ sampleRate: 16000 });
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const audioData = audioBuffer.getChannelData(0);

        const result = await transcriber.current(audioData, {
          language: "korean",
        });
        textBuffer.current += " " + result.text;

        if (textBuffer.current.split(" ").length > 4) {
          const messages = [
            {
              role: "system",
              content:
                "Predict the verb and translate the Korean SOV fragment into an English SVO fragment immediately. Output only the translation.",
            },
            { role: "user", content: textBuffer.current },
          ];

          const reply = await engine.chat.completions.create({
            messages,
            stream: false,
          });
          const englishText = reply.choices[0].message.content;
          setOutput(englishText);

          const utterance = new SpeechSynthesisUtterance(englishText);
          window.speechSynthesis.speak(utterance);

          textBuffer.current = "";
        }
      }
    };

    mediaRecorder.current.start(2000);
  };

  return (
    <div
      style={{
        padding: "2rem",
        fontFamily: "sans-serif",
        maxWidth: "600px",
        margin: "0 auto",
      }}
    >
      <h2>iOS Offline WebGPU Translator</h2>
      <p>
        Status: <strong>{status}</strong>
      </p>
      <progress
        value={progress}
        max="100"
        style={{ width: "100%", height: "20px" }}
      />

      <div style={{ marginTop: "1.5rem" }}>
        {!engine && (
          <button
            onClick={downloadModels}
            style={{
              padding: "0.75rem 1.5rem",
              fontSize: "1rem",
              cursor: "pointer",
            }}
          >
            Sync to Device (Use Wi-Fi Only)
          </button>
        )}
        {engine && (
          <button
            onClick={startEngine}
            style={{
              padding: "0.75rem 1.5rem",
              fontSize: "1rem",
              cursor: "pointer",
              backgroundColor: "#0070f3",
              color: "#fff",
              border: "none",
              borderRadius: "4px",
            }}
          >
            Start Listening
          </button>
        )}
      </div>

      <div
        style={{
          marginTop: "2rem",
          background: "#f5f5f5",
          padding: "1rem",
          borderRadius: "8px",
          minHeight: "100px",
        }}
      >
        <h3>English Translation Audio Stream:</h3>
        <p style={{ fontSize: "1.1rem", color: "#333" }}>
          {output || "Waiting for speech input..."}
        </p>
      </div>
    </div>
  );
}
