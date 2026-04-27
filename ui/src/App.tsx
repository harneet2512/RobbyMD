import { useEffect, useRef } from "react";
import { connectMock, connectLive, type Connection } from "@/api/client";
import { ReasoningCanvas } from "@/components/ReasoningCanvas/ReasoningCanvas";

const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

export default function App() {
  const conn = useRef<Connection | null>(null);

  useEffect(() => {
    if (USE_MOCK) {
      conn.current = connectMock();
    } else {
      conn.current = connectLive("demo");
    }
    return () => {
      conn.current?.disconnect();
      conn.current = null;
    };
  }, []);

  return <ReasoningCanvas />;
}
