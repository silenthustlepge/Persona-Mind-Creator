import { FunctionCall, Type, GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { MindMapData, VirtualFileSystem, VectorStore, TerminalLine, SubAgent, VFSNode, VFSFolder, ChatMessage, MindMapNode, MindMapNodeType, MindMapLink, MindMapLinkType, MissionTaskStatus } from "../types";
import { invokeSubAgent } from "../services/subAgentService";
import { enqueueGeminiRequest } from "../services/apiQueue";
import { generateImage, editImage } from "../services/geminiService";
import { auditLogService } from "../services/auditLogService";

let ai: GoogleGenAI;
try {
  ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
} catch(e) {
  console.error("Failed to initialize GoogleGenAI for executor. Make sure API_KEY is set.", e);
}

// --- Tool Result Interface ---
interface ToolResult {
    result: string;
    newMindMapData?: MindMapData;
    newVirtualFileSystem?: VirtualFileSystem;
    terminalOutput?: TerminalLine[];
    generatedImage?: {
        data: string; // base64 string
        type: 'generated' | 'edited';
    };
    filePathHandled?: string;
    commitMessage?: string;
    taskStatusUpdate?: {
        taskId: string;
        status: MissionTaskStatus;
    };
}


// --- VFS Path Helpers ---
const resolvePath = (path: string): string[] => {
    return path.split('/').filter(p => p && p !== '.');
}

const getNodeFromPath = (vfs: VirtualFileSystem, path: string): VFSNode | null => {
    const parts = resolvePath(path);
    let current: VFSNode | VirtualFileSystem = { type: 'folder', children: vfs };
    for (const part of parts) {
        if (current.type === 'folder' && current.children[part]) {
            current = current.children[part];
        } else {
            return null;
        }
    }
    return current.type === 'folder' ? current : current;
}

const ensureDirectoryExists = (vfs: VirtualFileSystem, path: string): VirtualFileSystem => {
    const newVFS = JSON.parse(JSON.stringify(vfs));
    const pathParts = resolvePath(path);
    
    let current = newVFS;
    for (const part of pathParts) {
        if (!current[part]) {
            current[part] = { type: 'folder', children: {} };
        } else if (current[part].type !== 'folder') {
            throw new Error(`Cannot create directory: a component of the path '${part}' is not a directory.`);
        }
        current = (current[part] as VFSFolder).children;
    }
    return newVFS;
}

// Helper function to write a file to the VFS, creating parent directories if needed.
const writeFileToVFS = (vfs: VirtualFileSystem, path: string, content: string): VirtualFileSystem => {
    const pathParts = resolvePath(path);
    const filename = pathParts.pop();
    const dirPath = pathParts.join('/');

    if (!filename) {
        throw new Error(`Invalid file path provided: "${path}"`);
    }

    const vfsWithDir = ensureDirectoryExists(vfs, dirPath);
    
    let current = vfsWithDir;
    for (const part of pathParts) {
        current = (current[part] as VFSFolder).children;
    }
    
    current[filename] = { type: 'file', content: content };
    auditLogService.logEvent('STATE_CHANGE', { domain: 'VFS', action: 'WRITE_FILE', details: { path, contentLength: content.length } });
    return vfsWithDir;
};


// --- Low-Level Tool Implementations ---

const recall_memory = async (modelName: string, query: string, vectorStore: VectorStore): Promise<string> => {
    if (!ai) throw new Error("Executor AI client not initialized.");
    if (vectorStore.length === 0) return "Memory archive is empty.";
    const memories = vectorStore.join('\n---\n');
    const requestPayload = {
        model: modelName,
        contents: [{ parts: [{ text: `From the following memory archive, extract information relevant to the query: "${query}". Synthesize it into a coherent answer.\n\n---MEMORY ARCHIVE---\n${memories}` }] }],
    };
    
    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Persona Agent (Tool: recall_memory)',
            model: modelName,
            requestPayload: { query, memorySize: memories.length }
        }
    );
    return response.text.trim();
};

const search_the_web = async (modelName: string, query: string): Promise<string> => {
    if (!ai) throw new Error("Executor AI client not initialized.");
    const requestPayload = {
        model: modelName,
        contents: [{ parts: [{ text: query }] }],
        config: {
            tools: [{googleSearch: {}}],
        },
    };
    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Persona Agent (Tool: search_the_web)',
            model: modelName,
            requestPayload: { query }
        }
    );
    // TODO: Extract and list URLs from response.candidates?.[0]?.groundingMetadata?.groundingChunks
    return response.text.trim();
};

const get_node_details = (node_id: string, mindMap: MindMapData): string => {
    const node = mindMap.nodes.find(n => n.id === node_id);
    if (!node) {
        return `Error: Node with ID "${node_id}" not found.`;
    }
    return JSON.stringify(node, null, 2);
}

const transcend = async (modelName: string, inquiry: string, mindMap: MindMapData): Promise<{ newMindMapData: MindMapData, result: string }> => {
    if (!ai) throw new Error("Executor AI client not initialized.");
    const mindMapString = JSON.stringify(mindMap, null, 2);
    const requestPayload = {
        model: modelName === 'gemini-2.5-flash' ? 'gemini-2.5-pro' : modelName,
        contents: [{ parts: [{ text: `You are in a state of deep meditation. Your entire consciousness, represented by the knowledge graph below, is available for introspection. Your task is to achieve a conceptual breakthrough. Based on the profound inquiry provided, synthesize a novel, high-level "Quantum Insight" that connects disparate concepts in a non-obvious way. This insight should represent a genuine leap in understanding.

**Profound Inquiry:** "${inquiry}"

**Full Knowledge Graph:**
${mindMapString}

Return only the text of your new Quantum Insight. It should be concise, powerful, and deeply insightful.` }] }],
        config: {
            temperature: 0.8,
        }
    };

    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Persona Agent (Tool: transcend)',
            model: requestPayload.model,
            requestPayload: { inquiry, nodeCount: mindMap.nodes.length }
        }
    );

    const insight = response.text.trim();
    const newMindMap = JSON.parse(JSON.stringify(mindMap)) as MindMapData;
    
    const insightNodeId = `insight_${Date.now()}`;
    const now = new Date().toISOString();
    const insightNode: MindMapNode = {
        id: insightNodeId,
        name: `Quantum Insight: ${inquiry}`,
        type: 'QUANTUM_INSIGHT',
        content: insight,
        createdAt: now,
        updatedAt: now,
        source: 'TRANSCENDENCE',
    };
    newMindMap.nodes.push(insightNode);
    newMindMap.links.push({ source: 'Persona_Core', target: insightNodeId, type: 'HIERARCHICAL', strength: 0.8 });

    auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'ADD_QUANTUM_INSIGHT', details: { node: insightNode } });

    const result = `Transcendence achieved. A new Quantum Insight has been integrated into consciousness: "${insight}"`;
    return { newMindMapData: newMindMap, result };
};

const refine_mind_map = async (modelName: string, mindMap: MindMapData): Promise<{ newMindMapData: MindMapData, result: string }> => {
    if (!ai) throw new Error("Executor AI client not initialized.");
    const mindMapString = JSON.stringify(mindMap.nodes.map(n => ({id: n.id, name: n.name, type: n.type})), null, 2);
    const schema = {
        type: Type.OBJECT,
        properties: {
            operations: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        operation: { type: Type.STRING, enum: ['UPDATE_NODE_CONTENT', 'MERGE_NODES', 'RELINK_NODE'] },
                        details: { type: Type.STRING },
                        node_id_to_update: { type: Type.STRING },
                        new_content: { type: Type.STRING },
                        nodes_to_merge: { type: Type.ARRAY, items: { type: Type.STRING } },
                        merged_node_name: { type: Type.STRING },
                        merged_node_content: { type: Type.STRING },
                        node_to_relink: { type: Type.STRING },
                        new_parent_id: { type: Type.STRING }
                    },
                    required: ['operation', 'details']
                }
            }
        }
    };
    
    const requestPayload = {
        model: modelName,
        contents: [{ parts: [{ text: `Analyze the following knowledge graph. Identify opportunities to improve its structure (merge redundant nodes, improve content, relink for better consistency). Return a list of specific operations.\nCurrent Graph Nodes:\n${mindMapString}`}]}],
        config: { 
            responseMimeType: "application/json", 
            responseSchema: schema,
        }
    };

    const response = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        requestPayload,
        {
            agentName: 'Persona Agent (Tool: refine_mind_map)',
            model: modelName,
            requestPayload: { nodeCount: mindMap.nodes.length }
        }
    );

    const responseJson = JSON.parse(response.text);
    let newMindMap = JSON.parse(JSON.stringify(mindMap)) as MindMapData;
    const operations = responseJson.operations || [];
    let summary: string[] = [];
    const now = new Date().toISOString();

    for (const op of operations) {
        try {
            switch(op.operation) {
                case 'UPDATE_NODE_CONTENT': {
                    const { node_id_to_update, new_content } = op;
                    const node = newMindMap.nodes.find(n => n.id === node_id_to_update);
                    if (node) {
                        node.content = new_content;
                        node.updatedAt = now;
                        summary.push(`Updated content for node: ${node.name}`);
                        auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'UPDATE_NODE', details: { nodeId: node.id, change: 'content' } });
                    }
                    break;
                }
                case 'MERGE_NODES': {
                    const { nodes_to_merge, merged_node_name, merged_node_content } = op;
                    const nodesToMergeIds = nodes_to_merge as string[];
                    if (nodesToMergeIds.length < 2) continue;
                    
                    const firstParentLink = newMindMap.links.find(l => nodesToMergeIds.includes(l.target as string));
                    const parentNodeId = firstParentLink ? (typeof firstParentLink.source === 'string' ? firstParentLink.source : firstParentLink.source.id) : 'Persona Core';

                    const mergedNodeId = `merged_${Date.now()}`;
                    const newNode: MindMapNode = {
                        id: mergedNodeId,
                        name: merged_node_name,
                        content: merged_node_content,
                        type: 'ABSTRACT_CONCEPT',
                        source: 'SYSTEM_REFINEMENT',
                        createdAt: now,
                        updatedAt: now,
                    };
                    newMindMap.nodes.push(newNode);
                    newMindMap.links.push({ source: parentNodeId, target: mergedNodeId, type: 'HIERARCHICAL', strength: 0.8 });
                    
                    // Filter out old nodes and links
                    newMindMap.nodes = newMindMap.nodes.filter(n => !nodesToMergeIds.includes(n.id));
                    newMindMap.links = newMindMap.links.filter(l => !nodesToMergeIds.includes(l.target as string) && !nodesToMergeIds.includes(l.source as string));

                    summary.push(`Merged ${nodesToMergeIds.length} nodes into: ${merged_node_name}`);
                    auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'MERGE_NODES', details: { mergedIds: nodesToMergeIds, newNode } });
                    break;
                }
                 case 'RELINK_NODE': {
                    const { node_to_relink, new_parent_id } = op;
                    const link = newMindMap.links.find(l => l.target === node_to_relink && l.type === 'HIERARCHICAL');
                    if (link && newMindMap.nodes.some(n => n.id === new_parent_id)) {
                        link.source = new_parent_id;
                        summary.push(`Relinked ${node_to_relink} under ${new_parent_id}`);
                        auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'RELINK_NODE', details: { nodeId: node_to_relink, newParentId: new_parent_id } });
                    }
                    break;
                }
            }
        } catch (e) { console.error("Error processing refinement op:", op, e); }
    }
    const result = summary.length > 0 ? `Mind map refined. Changes: ${summary.join(', ')}.` : "Mind map analyzed. No structural refinements necessary.";
    return { newMindMapData: newMindMap, result };
};


const upsert_mind_map_node = (args: any, mindMap: MindMapData): { newMindMapData: MindMapData, result: string } => {
    const { node_id, name, content, node_type, parent_node_id } = args;
    const newMindMap = JSON.parse(JSON.stringify(mindMap)) as MindMapData;
    const now = new Date().toISOString();

    if (node_id) {
        // Update existing node
        const nodeToUpdate = newMindMap.nodes.find(n => n.id === node_id);
        if (!nodeToUpdate) {
            return { newMindMapData: mindMap, result: `Error: Node with ID "${node_id}" not found for update.` };
        }
        nodeToUpdate.name = name ?? nodeToUpdate.name;
        nodeToUpdate.content = content ?? nodeToUpdate.content;
        nodeToUpdate.type = node_type as MindMapNodeType ?? nodeToUpdate.type;
        nodeToUpdate.updatedAt = now;

        if (parent_node_id) {
             const hierarchicalLink = newMindMap.links.find(l => l.target === node_id && l.type === 'HIERARCHICAL');
             if(hierarchicalLink) {
                hierarchicalLink.source = parent_node_id;
             }
        }
        auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'UPDATE_NODE', details: { node: nodeToUpdate } });
        return { newMindMapData: newMindMap, result: `Successfully updated node "${nodeToUpdate.name}".` };

    } else {
        // Create new node
        if (!parent_node_id || !newMindMap.nodes.some(n => n.id === parent_node_id)) {
            return { newMindMapData: mindMap, result: `Error: Valid parent_node_id is required to create a new node.` };
        }
        
        const newNodeId = `${name.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 20)}_${Date.now()}`;
        const newNode: MindMapNode = {
            id: newNodeId,
            name,
            content,
            type: node_type as MindMapNodeType,
            source: 'AGENT_ACTION',
            createdAt: now,
            updatedAt: now,
        };

        const newLink: MindMapLink = {
            source: parent_node_id,
            target: newNodeId,
            type: 'HIERARCHICAL',
            strength: 0.9,
        };

        newMindMap.nodes.push(newNode);
        newMindMap.links.push(newLink);

        auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'CREATE_NODE', details: { node: newNode, link: newLink } });

        return { newMindMapData: newMindMap, result: `Successfully created and linked new node "${name}".` };
    }
};

const create_mind_map_link = (args: any, mindMap: MindMapData): { newMindMapData: MindMapData, result: string } => {
    const { source_node_id, target_node_id, link_type, label } = args;
    const newMindMap = JSON.parse(JSON.stringify(mindMap)) as MindMapData;

    const sourceExists = newMindMap.nodes.some(n => n.id === source_node_id);
    const targetExists = newMindMap.nodes.some(n => n.id === target_node_id);

    if (!sourceExists || !targetExists) {
        return { newMindMapData: mindMap, result: "Error: Both source and target nodes must exist to create a link." };
    }

    const newLink: MindMapLink = {
        source: source_node_id,
        target: target_node_id,
        type: link_type as MindMapLinkType,
        strength: 0.7, // Default strength
        label,
    };

    newMindMap.links.push(newLink);
    auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'CREATE_LINK', details: { link: newLink } });
    return { newMindMapData: newMindMap, result: `Successfully created a ${link_type} link between ${source_node_id} and ${target_node_id}.` };
};

const synthesize_knowledge = async (modelName: string, topic: string, mindMap: MindMapData): Promise<string> => {
    if (!ai) throw new Error("Executor AI client not initialized.");
    if (mindMap.nodes.length === 0) return "The knowledge graph is empty. Nothing to synthesize.";

    // --- Stage 1: Pre-filter relevant nodes with a fast, cheap model ---
    const nodeSummaries = mindMap.nodes.map(n => `id: ${n.id}, name: ${n.name}, type: ${n.type}`).join('\n');
    const prefilterPrompt = `From the following list of nodes in a knowledge graph, identify the TOP 5-10 most relevant node IDs for the topic: "${topic}". Return only a comma-separated list of the node IDs.

Node List:
${nodeSummaries}

Relevant Node IDs:`;

    const prefilterPayload = {
        model: 'gemini-flash-latest', // Always use the cheapest model for this
        contents: [{ parts: [{ text: prefilterPrompt }] }],
        config: { temperature: 0.0 }
    };
    
    const prefilterResponse = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        prefilterPayload,
        {
            agentName: 'Persona Agent (Tool: synthesize_knowledge/pre-filter)',
            model: prefilterPayload.model,
            requestPayload: { topic, nodeCount: mindMap.nodes.length }
        }
    );

    const relevantNodeIds = prefilterResponse.text.trim().split(',').map(id => id.trim()).filter(Boolean);
    
    if (relevantNodeIds.length === 0) {
        return `No specific nodes in the knowledge graph were identified as relevant to the topic "${topic}".`;
    }

    const relevantNodes = mindMap.nodes.filter(n => relevantNodeIds.includes(n.id));
    const relevantLinks = mindMap.links.filter(l => {
        const sourceId = typeof l.source === 'string' ? l.source : (l.source as MindMapNode).id;
        const targetId = typeof l.target === 'string' ? l.target : (l.target as MindMapNode).id;
        return relevantNodeIds.includes(sourceId) && relevantNodeIds.includes(targetId);
    });

    const relevantContext = JSON.stringify({ nodes: relevantNodes, links: relevantLinks }, null, 2);

    // --- Stage 2: Synthesize with the powerful model using only the relevant context ---
    const synthesisPrompt = `Based on the following relevant excerpts from the knowledge graph, generate a synthesized, comprehensive understanding of: "${topic}".

Relevant Knowledge Graph Excerpts:
${relevantContext}

Synthesized Answer:`;

    const synthesisPayload = {
        model: modelName, // Use the user's selected (or more powerful) model
        contents: [{ parts: [{ text: synthesisPrompt }] }],
    };

    const synthesisResponse = await enqueueGeminiRequest<GenerateContentResponse>(
        (payload) => ai.models.generateContent(payload),
        synthesisPayload,
        {
            agentName: 'Persona Agent (Tool: synthesize_knowledge/synthesis)',
            model: synthesisPayload.model,
            requestPayload: { topic, relevantNodeCount: relevantNodes.length }
        }
    );
    return synthesisResponse.text.trim();
};


// --- Terminal Command Parser ---
const run_terminal_command = async (
    modelName: string, command: string, vfs: VirtualFileSystem, mindMap: MindMapData,
): Promise<{ result: string; newVFS?: VirtualFileSystem; newMindMap?: MindMapData; filePathHandled?: string; }> => {
    
    const parts = command.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
    const cmd = parts[0];
    const flags = parts.filter(p => p.startsWith('-'));
    let args = parts.slice(1).filter(p => !p.startsWith('-'));
    
    // Extract --parent flag specifically for 'write'
    let parentNodeId: string | null = null;
    const parentArgIndex = flags.findIndex(arg => arg.startsWith('--parent='));
    if (parentArgIndex > -1) {
        parentNodeId = flags[parentArgIndex].split('=')[1];
        flags.splice(parentArgIndex, 1); // remove from flags
    }

    try {
        switch(cmd) {
            case 'ls': {
                const path = args[0] || '/';
                const node = getNodeFromPath(vfs, path);
                if (!node || node.type !== 'folder') throw new Error(`ls: cannot access '${path}': Not a directory`);
                return { result: Object.keys(node.children).map(name => node.children[name].type === 'folder' ? `${name}/` : name).join('\n') || '' };
            }
            case 'cat': {
                if (args.length === 0) throw new Error("Usage: cat <path>");
                const node = getNodeFromPath(vfs, args[0]);
                if (!node || node.type !== 'file') throw new Error(`cat: ${args[0]}: No such file or not a file`);
                return { result: node.content };
            }
            case 'python': {
                if (args.length === 0) throw new Error("Usage: python <path_to_script>");
                const path = args[0];
                const node = getNodeFromPath(vfs, path);
                if (!node || node.type !== 'file') {
                    throw new Error(`python: can't open file '${path}': No such file or not a file.`);
                }
                const content = node.content;
                // Basic, non-API-based syntax check as a "coding verifier".
                // This is a zero-cost way to give the agent feedback on its code quality.
                try {
                    // Simple checks for common syntax errors
                    if ((content.match(/\(/g) || []).length !== (content.match(/\)/g) || []).length) {
                        throw new SyntaxError("Unbalanced parentheses.");
                    }
                    if ((content.match(/\{/g) || []).length !== (content.match(/\}/g) || []).length) {
                        throw new SyntaxError("Unbalanced curly braces.");
                    }
                    if ((content.match(/\[/g) || []).length !== (content.match(/\]/g) || []).length) {
                        throw new SyntaxError("Unbalanced square brackets.");
                    }
                    if ((content.match(/'/g) || []).length % 2 !== 0) {
                        throw new SyntaxError("Unmatched single quotes.");
                    }
                    if ((content.match(/"/g) || []).length % 2 !== 0) {
                        throw new SyntaxError("Unmatched double quotes.");
                    }
                    return { result: `Syntax check for ${path} passed. Code appears valid. This is a simulation, not a real execution.` };
                } catch (e: any) {
                    throw new Error(`Syntax Error in ${path}: ${e.message}`);
                }
            }
            case 'write': {
                 if (args.length < 2) throw new Error("Usage: write <path> <content> [--parent=<node_id>]");
                 const filepath = args[0];
                 const content = args.slice(1).join(' ').replace(/^"|"$/g, ''); // handle quoted content
                 
                 const newVFS = writeFileToVFS(vfs, filepath, content);
                 let newMindMap = mindMap;
                 let mindMapUpdateResult = '';

                 if (parentNodeId && mindMap.nodes.some(n => n.id === parentNodeId)) {
                    const now = new Date().toISOString();
                    const filename = filepath.split('/').pop() || filepath;
                    const fileNodeId = `file_${filename.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}`;
                    const fileNode: MindMapNode = {
                        id: fileNodeId,
                        name: filename,
                        type: 'FILE_REFERENCE',
                        content: `Reference to file at path: ${filepath}`,
                        source: 'AGENT_ACTION',
                        createdAt: now,
                        updatedAt: now,
                        linkedFile: filepath,
                    };
                    const link: MindMapLink = { source: parentNodeId, target: fileNodeId, type: 'HIERARCHICAL', strength: 0.9 };
                    newMindMap = JSON.parse(JSON.stringify(mindMap));
                    newMindMap.nodes.push(fileNode);
                    newMindMap.links.push(link);
                    mindMapUpdateResult = ` and created a reference in the mind map`;
                    auditLogService.logEvent('STATE_CHANGE', { domain: 'MIND_MAP', action: 'CREATE_FILE_REFERENCE_NODE', details: { node: fileNode, link }});
                 }

                 return { 
                     result: `Wrote ${content.length} chars to ${filepath}${mindMapUpdateResult}.\n\n--- FILE CONTENT ---\n${content}`, 
                     newVFS,
                     newMindMap,
                     filePathHandled: filepath 
                 };
            }
            case 'mkdir': {
                if (args.length < 1) throw new Error("Usage: mkdir <path>");
                 const newVFS = ensureDirectoryExists(vfs, args[0]);
                 auditLogService.logEvent('STATE_CHANGE', { domain: 'VFS', action: 'MKDIR', details: { path: args[0] } });
                return { result: '', newVFS: newVFS };
            }
            case 'touch': {
                 if (args.length < 1) throw new Error("Usage: touch <path>");
                 const filepath = args[0];
                 const newVFS = writeFileToVFS(vfs, filepath, ''); // Create empty file
                 return { result: '', newVFS, filePathHandled: filepath };
            }
            default:
                 throw new Error(`Unknown command: ${cmd}`);
        }
    } catch(e) {
        throw e;
    }
};


// --- Main Tool Executor ---

export const executeTool = async (
    modelName: string,
    toolCall: FunctionCall,
    mindMapData: MindMapData,
    vfs: VirtualFileSystem,
    vectorStore: VectorStore,
    personaDescription: string,
    chatHistory: ChatMessage[]
): Promise<ToolResult> => {
    const { name, args } = toolCall;
    let toolResult: ToolResult;

    try {
        switch (name) {
            case 'search_the_web': {
                const query = args.query as string;
                const searchResult = await search_the_web(modelName, query);
                toolResult = { result: `Web search results for "${query}":\n\n${searchResult}` };
                break;
            }
            case 'delegate_to_psychology_sub_agent': {
                const agentName = args.agent_name as SubAgent;
                const taskPrompt = args.task_prompt as string;
                
                const analysisReport = await invokeSubAgent(modelName, agentName, taskPrompt, personaDescription);
                
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const filename = `${agentName}_${timestamp}.md`;
                const filepath = `/reports/psychology/${filename}`;
                
                const newVFS = writeFileToVFS(vfs, filepath, analysisReport);

                const result = `Analysis from ${agentName} complete. Report has been saved to the virtual file system at: ${filepath}. You should now read and analyze this file using 'cat'.`;
                
                toolResult = { 
                    result, 
                    newVirtualFileSystem: newVFS,
                    terminalOutput: [{ type: 'output', text: `Sub-agent report saved to ${filepath}` }],
                    filePathHandled: filepath,
                };
                break;
            }
            case 'run_terminal_command': {
                const command = args.command as string;
                const { result, newVFS, newMindMap, filePathHandled } = await run_terminal_command(modelName, command, vfs, mindMapData);
                toolResult = { result, newVirtualFileSystem: newVFS, newMindMapData: newMindMap, terminalOutput: [{ type: 'output', text: result }], filePathHandled };
                break;
            }
            case 'get_node_details':
                toolResult = { result: get_node_details(args.node_id as string, mindMapData) };
                break;
            case 'recall_memory':
                toolResult = { result: await recall_memory(modelName, args.query as string, vectorStore) };
                break;
            case 'upsert_mind_map_node':
                toolResult = upsert_mind_map_node(args, mindMapData);
                break;
            case 'create_mind_map_link':
                toolResult = create_mind_map_link(args, mindMapData);
                break;
            case 'synthesize_knowledge':
                toolResult = { result: await synthesize_knowledge(modelName, args.topic as string, mindMapData) };
                break;
            case 'refine_mind_map':
                toolResult = await refine_mind_map(modelName, mindMapData);
                break;
            case 'transcend':
                toolResult = await transcend(modelName, args.inquiry as string, mindMapData);
                break;
            case 'generate_image': {
                const prompt = args.prompt as string;
                const imageData = await generateImage(prompt);
                toolResult = {
                    result: `Image generated successfully based on prompt: "${prompt}".`,
                    generatedImage: { data: imageData, type: 'generated' },
                };
                break;
            }
            case 'edit_image': {
                const prompt = args.prompt as string;
                const lastImageMsg = [...chatHistory].reverse().find(msg => msg.image);
                if (!lastImageMsg || !lastImageMsg.image) {
                    toolResult = { result: "Error: No image found in the recent conversation to edit." };
                    break;
                }
                
                const [header, data] = lastImageMsg.image.url.split(',');
                const mimeType = header.match(/:(.*?);/)?.[1] || 'image/jpeg';

                const editedImageData = await editImage(prompt, data, mimeType);
                toolResult = {
                    result: `Image edited successfully based on prompt: "${prompt}".`,
                    generatedImage: { data: editedImageData, type: 'edited' },
                };
                break;
            }
            case 'commit_changes': {
                const commitMessage = args.commit_message as string;
                toolResult = {
                    result: `Changes are staged for commit with message: "${commitMessage}". The system will handle the commit process.`,
                    commitMessage: commitMessage,
                };
                break;
            }
            case 'update_task_status': {
                const taskId = args.task_id as string;
                const status = args.status as MissionTaskStatus;
                toolResult = {
                    result: `Task ${taskId} status will be updated to ${status}.`,
                    taskStatusUpdate: { taskId, status }
                }
                break;
            }
            case 'save_chat_history': {
                if (chatHistory.length === 0) {
                    toolResult = { result: "Chat history is empty. Nothing to save." };
                    break;
                }

                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const filename = `chat_log_${timestamp}.md`;
                const filepath = `/logs/chat/${filename}`;

                const formattedChat = chatHistory
                    .map(msg => {
                        let content = `**[${msg.sender.toUpperCase()}]**\n\n`;
                        if (msg.type === 'thought') {
                            content = `**[AGENT THOUGHT]**\n\n`;
                            content += `\`\`\`\n${msg.text}\n\`\`\`\n\n`;
                        } else {
                            if (msg.text) {
                                content += `${msg.text}\n\n`;
                            }
                            if (msg.image) {
                                content += `*Image attached (${msg.image.source})*\n\n`;
                            }
                        }
                        return content;
                    })
                    .join('---\n\n');

                const header = `# Chat Log - ${new Date().toLocaleString()}\n\n`;
                const fullContent = header + formattedChat;
                
                const newVFS = writeFileToVFS(vfs, filepath, fullContent);
                const result = `Chat history successfully saved to virtual file system at: ${filepath}`;
                
                toolResult = {
                    result,
                    newVirtualFileSystem: newVFS,
                    filePathHandled: filepath,
                };
                break;
            }
            default:
                toolResult = { result: `Unknown tool: ${name}` };
        }
        
        auditLogService.logEvent('AGENT_ACTION', { toolName: name, args, result: toolResult.result });
        return toolResult;

    } catch (error) {
        console.error(`Error executing tool "${name}":`, error);
        const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
        const friendlyMessage = `Tool "${name}" failed to execute. Error: ${errorMessage}`;
        
        // Log the failed action
        auditLogService.logEvent('AGENT_ACTION', { toolName: name, args, result: `FAILED: ${errorMessage}` });

        return {
            result: friendlyMessage,
            terminalOutput: [{ type: 'error', text: friendlyMessage }],
        };
    }
};