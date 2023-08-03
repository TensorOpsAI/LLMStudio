export const getChatProvider = (model, uppercase) => {
  if (model === "gpt-3.5-turbo" || model === "gpt-4")
    return uppercase ? "OpenAI" : "openai";
  if (model === "text-bison@001" || model === "chat-bison@001")
    return uppercase ? "Vertex AI" : "vertexai";
};

export const getStatusColor = (status) => {
  if (status === "idle") return "bg-slate-400";
  if (status === "waiting") return "bg-yellow-400";
  if (status === "done") return "bg-green-500";
  if (status === "error") return "bg-red-600";
};
