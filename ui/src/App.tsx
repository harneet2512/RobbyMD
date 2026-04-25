import { useEffect, useRef } from "react";
import { connectMock, type Connection } from "@/api/client";
import { ReasoningCanvas } from "@/components/ReasoningCanvas/ReasoningCanvas";

export default function App() {
  const conn = useRef<Connection | null>(null);

  useEffect(() => {
    conn.current = connectMock();
    return () => {
      conn.current?.disconnect();
      conn.current = null;
    };
  }, []);

  return <ReasoningCanvas />;
}
