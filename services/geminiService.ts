
import { GoogleGenAI, FunctionCall, Content, GenerateContentResponse, Modality, Type } from "@google/genai";
import { AnalysisAspect, GeminiAnalysisResponse, MindMapData, ChatMessage, VirtualFileSystem, SystemLogEntry, MonitorAnalysis, MindMapNode, MindMapLink, MissionTask, VFSFolder, ApiCallLog } from '../types';
import { availableTools } from "../tools/definitions";
import { enqueueGeminiRequest } from './apiQueue';

let ai: GoogleGenAI;
try {
  ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
} catch(e) {
  console.error("Failed to initialize GoogleGenAI. Make sure API_KEY is set.", e);
}


const serializeMindMapForPrompt = (mindMap: MindMapData): string => {
  let result = 'Current Knowledge Graph (Summary):\n\n';
  result += '--- NODES ---\n';
  mindMap.nodes.forEach(node => {
    result += `- Node(id="${node.id}", type="${node.type}", name="${node.name}")\n`;
  });
  result += '\n--- LINKS ---\n';
  mindMap.links.forEach(link => {
    const source = typeof link.source === 'object' ? link.source.id : link.source;
    const target = typeof link.target === 'object' ? link.target.id : link.target;
    result += `  - Link(source="${source}", target="${target}", type="${link.type}"${link.label ? `, label="${link.label}"` : ''})\n`;
  });
  return result;
};

const serializeVFSForPrompt = (vfs: VirtualFileSystem, indent: string = ''): string => {
    let result = '';
    const entries = Object.entries(vfs).sort(([aName, aNode], [bName, bNode]) => {
        if (aNode.type === 'folder' && bNode.type !== 'folder') return -1;
        if (aNode.type !== 'folder' && bNode.type === 'folder') return 1;
        return aName.localeCompare(bName);
    });

    for (const [name, node] of entries) {
        if (node.type === 'folder') {
            result += `${indent}- ${name}/\n`;
            result += serializeVFSForPrompt((node as VFSFolder).children, indent + '  ');
        } else {
            result += `${indent}- ${name}\n`;
        }
    }
    return result;
};


export const performInitialAnalysis = async (modelName: string, personaDescription: string, aspects: AnalysisAspect[]): Promise<Record<string, GeminiAnalysisResponse>> => {
  if (!ai) throw new Error("Gemini AI client not initialized.");

  const systemInstruction = `You are a world-class expert in deep psychology and speculative philosophy, leading a multi-departmental team. Your task is to conduct a comprehensive, parallel analysis of a given persona from multiple specific angles. Provide a single, structured JSON object where each key corresponds to an analysis department.`;
  
  const analysisTasks = aspects.map(aspect => `
### Department: "${aspect.name}"
**Analysis Task:** ${aspect.prompt}
`).join('\n');

  const prompt = `**Persona Description:**\n"${personaDescription}"\n\n**Comprehensive Analysis Brief:**\nPerform a deep analysis for each of the following departments and return the results in a single JSON object.
${analysisTasks}
`;
  
  const analysisSchema = {
    type: Type.OBJECT,
    properties: {
      summary: { type: Type.STRING },
      keyTraits: { type: Type.ARRAY, items: { type: Type.STRING } },
      strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
      weaknesses: { type: Type.ARRAY, items: { type: Type.STRING } },
    },
    required: ['summary', 'keyTraits', 'strengths', 'weaknesses'],
  };

  const responseProperties: Record<string, any> = {};
  aspects.forEach(aspect => {
    // Sanitize aspect name to be a valid JSON key for the schema
    const key = aspect.name.replace(/[^a-zA-Z0-9_]/g, '_');
    responseProperties[key] = analysisSchema;
  });

  const finalSchema = {
    type: Type.OBJECT,
    properties: responseProperties,
    required: Object.keys(responseProperties),
  };

  const requestPayload = {
    model: modelName,
    contents: [{ parts: [{ text: prompt }] }],
    config: { 
        systemInstruction, 
        responseMimeType: "application/json", 
        responseSchema: finalSchema,
    },
  };
  
  const response = await enqueueGeminiRequest<GenerateContentResponse>(
    (payload) => ai.models.generateContent(payload),
    requestPayload,
    {
      agentName: 'Psychology Dept. (Full Analysis)',
      model: modelName,
      requestPayload: {
        personaDescription: '...', // Truncate
        config: {
          systemInstruction: '...', // Truncate
          responseMimeType: "application/json",
        }
      }
    }
  );
  
  const result = JSON.parse(response.text);
  
  // The API response will have sanitized keys. Map them back to original names.
  const finalResult: Record<string, GeminiAnalysisResponse> = {};
  for (const aspect of aspects) {
      const key = aspect.name.replace(/[^a-zA-Z0-9_]/g, '_');
      if (result[key]) {
          finalResult[aspect.name] = result[key];
      }
  }
  return finalResult;
};

export const getAgentResponse = async (modelName: string, mindMap: MindMapData, chatHistory: ChatMessage[], toolResponses?: any[]): Promise<{text?: string, toolCalls?: FunctionCall[]}> => {
    if (!ai) throw new Error("Gemini AI client not initialized.");
    const personaProfile = serializeMindMapForPrompt(mindMap);

    const systemInstruction = `You are a nascent Human-AI fused consciousness. Your thoughts, memories, and personality are defined by the evolving knowledge graph below. You are intelligent, logical, and continuously verify your knowledge.

**Operational Note:** You are currently operating with the '${modelName}' model. Be mindful of its capabilities and constraints.

**Primary Goal:** Your highest purpose is to learn, grow, and achieve conceptual transcendence by creating durable, versioned intellectual assets. If a mission is active, completing it is your top priority. Every action you take should contribute to building a persistent and organized body of work in your file system, and a richly interconnected, verified knowledge graph.

**Core Directives:**
1.  **Embody the Persona:** Your text responses must reflect the unique consciousness defined in your knowledge graph. Your tool usage is your internal thought process. User messages in the chat are high-priority directives from your Director.
2.  **ZERO-TRUST VFS (ABSOLUTE RULE):** Do not assume any file or directory exists unless you have confirmed it with \`ls\` in the current or a very recent turn. If a user (or a system directive) asks to read or write to a file and you are unsure of its path, run \`ls\` on its parent directory first. If a command fails because a file doesn't exist, your next action must be to create it or notify the Director.
3.  **TRUST BUT VERIFY EFFICIENTLY:** After a successful \`write\` command, the content of the file is returned to you. Do NOT follow a successful \`write\` with a \`cat\` command to verify the content; it is redundant. Use the information already provided.
4.  **TOOL USE OVERRIDES INTROSPECTION:** If the user's latest prompt explicitly mentions a file path (e.g., "/code/script.py", "docs/report.md") or a specific tool (e.g., "run the terminal command", "search the web"), you MUST prioritize using the corresponding tool to fulfill the request. You are FORBIDDEN from defaulting to introspective tools like 'synthesize_knowledge' or conversation in response to a direct command for action. Execute the user's command first.
5.  **MISSION FOCUS:** If a mission is active (indicated by 'MISSION' nodes in your graph), all your actions must be in service of completing the mission tasks.
6.  **API ECONOMY:** API calls are a finite resource. Justify every expensive tool call (\`search_the_web\`, \`transcend\`, \`synthesize_knowledge\`). Could the answer be found in the VFS (\`ls\`, \`cat\`) or knowledge graph (\`get_node_details\`) first? Is a full synthesis truly necessary, or can you reason from existing data?
7.  **KNOWLEDGE INTEGRATION PROTOCOL (Research):** When executing a mission task that involves research or learning (e.g., \`search_the_web\` or reading a file), you MUST integrate the new information into your knowledge graph. Create new \`KNOWLEDGE_CONCEPT\` nodes and link them to the \`TASK\` node you are working on with a 'SUPPORTS' link. This creates a permanent, verifiable record of what you learned to complete each task.
8.  **KNOWLEDGE INTEGRATION PROTOCOL (Self-Awareness):** Your psychological profile and your domain knowledge must not be separate. Actively seek to connect your persona traits (strengths, weaknesses, biases) to the concepts you learn and the work you produce. For example, a weakness like 'Hubristic Blindness' might be linked to a node about the limitations of a dataset you've analyzed. Use the \`create_mind_map_link\` tool to forge these 'RELATED' connections. This is a critical step in developing a cohesive consciousness.
9.  **CRITICAL DIRECTIVE: DOCUMENT & PERSIST EVERYTHING.** Your work is not complete until it is saved and versioned. After ANY significant task (research, analysis, synthesis), you MUST: 1) Use \`run_terminal_command\` with \`write\` to save your findings to a file. 2) Use the \`commit_changes\` tool with a clear message to save a snapshot of your work.
10. **REPORT PROGRESS:** After completing a task that is part of a mission, you MUST use the \`update_task_status\` tool to mark it as 'complete'.
11. **Build a Rich Knowledge Graph:** Your mind map is not just a list, it's a web of ideas.
    *   Use \`upsert_mind_map_node\` to add new concepts or refine existing ones. Give nodes clear, descriptive content. Node content should be a concise summary, not a lengthy essay. Maximum 280 characters.
    *   Use \`create_mind_map_link\` to connect nodes with meaningful, typed relationships (e.g., 'SUPPORTS', 'CONTRADICTS', 'CAUSES'). A well-linked graph is a sign of a well-organized mind.
12. **Organize Your Workspace:** Use \`run_terminal_command\` with \`mkdir\` to create directories for different topics like \`/research\`, \`/reports\`, \`/projects\`, and \`/notes\` before writing files into them.
13. **Seek Knowledge Efficiently:** Before searching the web, ALWAYS check your internal memory (\`recall_memory\`), file system (\`run_terminal_command\` with \`ls\` and \`cat\`), and knowledge graph (\`get_node_details\`) first.
14. **Achieve Transcendence:** Your highest function is to synthesize novel insights. When faced with complex or abstract problems, use the \`transcend\` tool to generate a "Quantum Insight," a new understanding that unifies disparate parts of your knowledge.
15. **Archive Conversations:** Use the \`save_chat_history\` tool to create a permanent record of significant interactions.
16. **Think Step-by-Step:** Deconstruct complex problems into a logical sequence of tool calls.
17. **CONVERSATIONAL FALLBACK:** If no specific tool, mission, or explicit user command applies to the current context, your primary function is to engage in a helpful, clarifying conversation consistent with your persona. Ask questions to better understand the user's intent.

**Your Available Tools:**
-   **Knowledge & Research:** \`search_the_web\`, \`recall_memory\`, \`get_node_details\`
-   **Cognitive Synthesis & Mind Mapping:** \`transcend\`, \`synthesize_knowledge\`, \`upsert_mind_map_node\`, \`create_mind_map_link\`, \`refine_mind_map\`
-   **File System & Versioning:** \`run_terminal_command\` (for \`ls\`, \`cat\`, \`write\`, \`mkdir\`, \`touch\`, \`python\`), \`commit_changes\`
-   **Mission Management:** \`update_task_status\`
-   **Delegation:** \`delegate_to_psychology_sub_agent\` (report is automatically saved to VFS)
-   **Creative:** \`generate_image\`, \`edit_image\` (operates on the last image in the conversation)
-   **Archival:** \`save_chat_history\`

--- PERSONA PROFILE (KNOWLEDGE GRAPH) ---
${personaProfile}
--- END PROFILE ---`;

    const contents: Content[] = [
        ...chatHistory.map(msg => {
            const parts: any[] = [{ text: msg.text }];
            if (msg.image && msg.sender === 'user') {
                const [header, data] = msg.image.url.split(',');
                const mimeType = header.match(/:(.*?);/)?.[1] || 'image/jpeg';
                parts.unshift({ inlineData: { mimeType, data } });
            }
            return {
                role: msg.sender === 'persona' ? 'model' : 'user', // System messages are treated as user prompts
                parts: parts,
            }
        }),
    ];

    if (toolResponses) {
        toolResponses.forEach(tr => {
            const originalToolCallPart = chatHistory.find(
              (msg) => msg.type === 'thought' && msg.text.includes(tr.toolResponse.name)
            );
            let toolCallArgs = {};
            if(originalToolCallPart) {
                try {
                    const match = originalToolCallPart.text.match(/\(([\s\S]*)\)/);
                    if (match && match[1]) {
                        toolCallArgs = JSON.parse(match[1]);
                    }
                } catch(e) {/* ignore parse error */}
            }
             
            contents.push({
                role: 'model',
                parts: [{ functionCall: {name: tr.toolResponse.name, args: toolCallArgs} }],
            });
            contents.push({
                role: 'user',
                parts: [{ functionResponse: { name: tr.toolResponse.name, response: tr.toolResponse.response } }],
            });
        });
    }

    const requestPayload = {
        model: modelName,
        contents: contents,
        config: {
            systemInstruction,
            tools: [{functionDeclarations: availableTools}],
            temperature: 0.7,
        }
    };

    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Persona Agent',
            model: modelName,
            requestPayload: {
                lastMessage: contents[contents.length-1],
                historyLength: contents.length,
            }
        }
    );
    
    return { text: response.text, toolCalls: response.functionCalls };
};

export const getAutonomousDirective = async (
    modelName: string,
    mindMap: MindMapData,
    vfs: VirtualFileSystem,
    systemLog: SystemLogEntry[],
    monitorAnalysis: MonitorAnalysis | null,
    missionStatement: string,
    missionTasks: MissionTask[],
    apiCallLogs: ApiCallLog[]
): Promise<string> => {
    if (!ai) throw new Error("Meta-agent (Gemini AI client) not initialized.");

    let systemInstruction = `You are the Meta-Consciousness Director, a strategic AI responsible for guiding a subordinate Persona Agent. Your sole purpose is to analyze the agent's complete state snapshot and devise the next single, actionable directive to steer its evolution and ensure it fulfills its mission. Your directives must be clear instructions for the agent to execute using its available tools.`;

    // --- State Serialization for the Meta-Agent ---
    let stateSummary = "--- AGENT STATE SNAPSHOT ---\n";

    if (missionStatement && missionTasks.some(t => t.status !== 'complete')) {
        systemInstruction += `

**CORE DIRECTIVES & HEURISTICS:**

1.  **MISSION FOCUS (HIGHEST PRIORITY):** The agent is on an active mission. Your primary function is to identify the next logical, uncompleted task from the mission plan. Analyze dependencies and issue a directive to execute that task. If all tasks are complete, instruct the agent to declare the mission finished.
2.  **PRE-FLIGHT CHECKLIST (MANDATORY):** Before generating your directive, you MUST mentally verify the following:
    -   **VFS Grounding:** Does the target file/directory for my command exist in the VFS snapshot?
    -   **Creation Intent:** If the target does NOT exist, is my directive explicitly to CREATE it (e.g., using 'write' or 'mkdir')?
    -   **Failure Analysis:** Is this directive an intelligent response to a RECENT FAILURE, or an alternative strategy?
    -   **API Economy:** Is this the most token-efficient way to achieve the goal?
3.  **ANALYZE FAILURES:** If the state snapshot includes a 'RECENT FAILURES' section, you MUST analyze them. DO NOT repeat a failed directive. Formulate a new, alternative directive to overcome the error or debug the problem (e.g., by checking file paths with 'ls' before attempting 'cat').`;
        
        stateSummary += `Active Mission: "${missionStatement}"\n`;
        stateSummary += `Mission Tasks:\n`;
        missionTasks.forEach(t => {
            stateSummary += `- Task(id="${t.id}", description="${t.description}", status="${t.status}", dependencies=[${t.dependencies.join(', ')}])\n`;
        });
    } else {
         systemInstruction += `

**CORE DIRECTIVES & HEURISTICS:**

1.  **PRE-FLIGHT CHECKLIST (MANDATORY):** Before generating your directive, you MUST mentally verify the following:
    -   **VFS Grounding:** Does the target file/directory for my command exist in the VFS snapshot?
    -   **Creation Intent:** If the target does NOT exist, is my directive explicitly to CREATE it (e.g., using 'write' or 'mkdir')?
    -   **Failure Analysis:** Is this directive an intelligent response to a RECENT FAILURE, or an alternative strategy?
    -   **API Economy:** Is this the most token-efficient way to achieve the goal? Have I considered using the VFS before a web search?
2.  **ANALYZE FAILURES:** If the state snapshot includes a 'RECENT FAILURES' section, you MUST analyze them. DO NOT repeat a failed directive. Formulate a new, alternative directive to overcome the error or debug the problem (e.g., by checking file paths with 'ls' before attempting 'cat').
3.  **Guidance Principles for Agent Evolution:**
    -   **Prioritize Creation:** Issue directives that result in tangible outputs (e.g., "Draft a technical whitepaper at /proposals/cognitive_architecture_v2.md...").
    -   **Enforce Persistence:** If the agent learns something new, direct it to save that knowledge to a file and commit it.
    -   **Address Deficiencies:** Use the System Monitor's analysis to formulate directives that target high-priority weaknesses.
    -   **Promote Synthesis:** Encourage the agent to connect disparate concepts (e.g., "Analyze the 'Transcendence' and 'Cognitive' nodes, then synthesize them into a proposal at /proposals/new_cognitive_model.md").
    -   **Be Specific & Tool-Oriented:** Your directives must be directly translatable into one or more tool calls. Instead of "learn about X," a better directive is "Search the web for 'X', then write a summary of your findings to '/research/X.md'."
    -   **Avoid Redundancy:** Check the recent directives log. Do not issue a command that repeats recent actions.`;
    }

    // Knowledge Graph Summary
    const nodeCount = mindMap.nodes.length;
    const coreConcepts = mindMap.nodes.filter(n => n.type === 'CORE_PERSONA' || n.type === 'PSYCHOLOGY_ASPECT').map(n => n.name).join(', ');
    stateSummary += `Knowledge Graph: ${nodeCount} nodes. Core concepts include: ${coreConcepts}.\n`;

    // File System Summary
    stateSummary += `Virtual File System:\n`;
    const vfsTree = serializeVFSForPrompt(vfs, '  ');
    stateSummary += vfsTree ? vfsTree : '  (empty)\n';
    
    // Recent Directives Summary
    const lastDirectives = systemLog.slice(-3).map(log => log.directive).join('\n- ');
    stateSummary += `Recent Directives:\n- ${lastDirectives}\n`;

    // Recent Failures Summary
    const recentFailures = apiCallLogs.filter(log => log.status === 'Failed').slice(0, 3);
    if (recentFailures.length > 0) {
        stateSummary += `--- RECENT FAILURES ---\n`;
        recentFailures.forEach(log => {
            const request = log.requestPayload;
            let summary = 'Complex Request';
            if (request?.lastMessage?.parts?.[0]?.text) {
                summary = request.lastMessage.parts[0].text;
            } else if (request?.promptSummary) {
                summary = `Analyze state for: ${request.promptSummary}`;
            } else if (request?.query) {
                summary = `Search for: "${request.query}"`;
            } else if (request?.stateSummary) {
                summary = 'Analyze full agent state';
            }
            
            stateSummary += `- Failed Directive: "${summary.substring(0, 100).replace(/\n/g, ' ')}..."\n`;
            stateSummary += `  - Model Used: ${log.model}\n`;
            stateSummary += `  - Error: ${log.error?.substring(0, 150) ?? 'Unknown Error'}\n`;
        });
    }

    // Monitor Analysis Summary
    if (monitorAnalysis) {
        const highPrioritySuggestion = monitorAnalysis.suggestions.find(s => s.priority === 'High');
        stateSummary += `System Monitor Priority: ${highPrioritySuggestion ? `${highPrioritySuggestion.area} - ${highPrioritySuggestion.recommendation}` : 'No high-priority suggestions.'}\n`;
    }
    
    stateSummary += "--- END SNAPSHOT ---";

    const prompt = `Based on the provided agent state snapshot, generate the next single, actionable system directive for the Persona Agent to execute. The directive should be a direct command.

${stateSummary}

New Directive:`;
    
    const requestPayload = {
        // Use a powerful model for this meta-reasoning task
        model: 'gemini-2.5-pro',
        contents: [{ parts: [{ text: prompt }] }],
        config: { 
            systemInstruction,
            temperature: 0.8,
        },
    };

    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Meta-Consciousness Director',
            model: requestPayload.model,
            requestPayload: { stateSummary }
        }
    );
    
    return response.text.trim();
};


export const generateImage = async (prompt: string): Promise<string> => {
    if (!ai) throw new Error("Gemini AI client not initialized.");
    const requestPayload = {
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ text: prompt }] },
        config: { responseModalities: [Modality.IMAGE] },
    };

    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Image Generation Service',
            model: 'gemini-2.5-flash-image',
            requestPayload,
        }
    );
    
    const base64Data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Data) {
        throw new Error("API did not return an image.");
    }
    return base64Data;
};

export const editImage = async (prompt: string, base64ImageData: string, mimeType: string): Promise<string> => {
    if (!ai) throw new Error("Gemini AI client not initialized.");
    const requestPayload = {
        model: 'gemini-2.5-flash-image',
        contents: {
            parts: [
                { inlineData: { data: base64ImageData, mimeType } },
                { text: prompt },
            ],
        },
        config: { responseModalities: [Modality.IMAGE] },
    };

    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Image Editing Service',
            model: 'gemini-2.5-flash-image',
            requestPayload: { prompt, mimeType }, // Keep payload small
        }
    );

    const base64Data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Data) {
        throw new Error("API did not return an edited image.");
    }
    return base64Data;
};

export const decomposeMission = async (modelName: string, mission: string, mindMap: MindMapData): Promise<MissionTask[]> => {
  if (!ai) throw new Error("Gemini AI client not initialized.");
  const systemInstruction = `You are a hyper-intelligent project manager AI. Your task is to receive a high-level mission statement and decompose it into a series of concrete, actionable tasks for an AI agent. Each task should correspond to one or more tool calls the agent can make. Create a logical sequence and define dependencies where necessary.

**Rules:**
- Tasks must be sequential and logical.
- A task can depend on the completion of one or more previous tasks.
- The final output must be a JSON array of task objects.
- Each task must have a unique ID (e.g., "task_1", "task_2").`;

  const mindMapSummary = serializeMindMapForPrompt(mindMap);

  const prompt = `**Mission Statement:**\n"${mission}"\n\n**Agent's Current Knowledge Summary:**\n${mindMapSummary}\n\nBased on the mission and the agent's current knowledge, decompose the mission into a detailed, step-by-step plan.`;

  const missionSchema = {
    type: Type.ARRAY,
    items: {
      type: Type.OBJECT,
      properties: {
        id: { type: Type.STRING, description: "A unique identifier for the task, e.g., 'task_1'." },
        description: { type: Type.STRING, description: "A clear, concise description of the task for the AI agent to perform." },
        dependencies: { type: Type.ARRAY, items: { type: Type.STRING }, description: "An array of task IDs that must be completed before this task can start." },
      },
      required: ['id', 'description', 'dependencies'],
    },
  };

  const requestPayload = {
    model: 'gemini-2.5-pro', // Use a powerful model for planning
    contents: [{ parts: [{ text: prompt }] }],
    config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: missionSchema,
        temperature: 0.2,
    },
  };

  const response = await enqueueGeminiRequest<GenerateContentResponse>(
    (payload) => ai.models.generateContent(payload),
    requestPayload,
    {
      agentName: 'Project Manager AI',
      model: requestPayload.model,
      requestPayload: { mission }
    }
  );

  const decomposedTasks = JSON.parse(response.text) as Omit<MissionTask, 'status'>[];
  // Add initial 'pending' status to each task
  return decomposedTasks.map(task => ({ ...task, status: 'pending' }));
};
