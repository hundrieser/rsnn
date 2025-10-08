import RSNNDesigner from "./RSNNDesigner";
import "./App.css";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center">
      <div className="w-full max-w-6xl px-4 pt-6">
        <RSNNDesigner />
      </div>
    </div>
  );
}
