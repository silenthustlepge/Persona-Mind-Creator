import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Increase payload limit for base64 images upload/retrieve
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));

  // API Route: Secure server-side Gemini API Proxy
  app.post("/api/gemini/generateContent", async (req, res) => {
    try {
      const payload = req.body;
      const apiKey = process.env.GEMINI_API_KEY;

      if (!apiKey) {
        console.error("Error: GEMINI_API_KEY environment variable is not defined on the server side.");
        return res.status(403).json({
          error: {
            message: "Gemini API key is not configured on the server. Please add it to your Settings > Secrets.",
            code: 403,
            status: "PERMISSION_DENIED"
          }
        });
      }

      // Initialize server-side Gemini client
      const serverAi = new GoogleGenAI({
        apiKey
      });

      // Call the Google GenAI SDK
      const response = await serverAi.models.generateContent(payload);

      // Map getters to serializable plain properties
      res.json({
        ...response,
        text: response.text,
        functionCalls: response.functionCalls,
      });
    } catch (error: any) {
      console.error("Gemini API server Proxy encounters an error:", error);
      const httpStatus = typeof error.status === 'number' ? error.status : 500;
      res.status(httpStatus).json({
        error: {
          message: error.message || "An error occurred on the Gemini server proxy.",
          code: error.code || httpStatus,
          status: error.status || "INTERNAL"
        }
      });
    }
  });

  // API Health check route
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  app.get("/api/debug", (req, res) => {
    res.json({
      hasEnvKey: !!process.env.GEMINI_API_KEY,
      keyPrefix: process.env.GEMINI_API_KEY ? process.env.GEMINI_API_KEY.substring(0, 4) : "NONE"
    });
  });

  // Front-end Vite Dev Middleware / Static Assets serving
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in development mode with Vite middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in production mode...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server is running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start fullstack server:", err);
});
