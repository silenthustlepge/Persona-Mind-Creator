import { apiMonitorService } from './apiMonitorService';
import { calculateCost } from '../utils/pricing';
import { auditLogService } from './auditLogService';
import { ApiCallLog } from '../types';

const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 1000;
const MAX_CONCURRENT_REQUESTS = 1;
const GOVERNOR_INTERVAL_MS = 250; // Check the queue every 250ms
const REQUEST_CACHE_EXPIRY_MS = 1000 * 60 * 5; // Cache identical requests for 5 minutes

// In-memory cache for API requests to save quota
const requestCache = new Map<string, { result: any; timestamp: number }>();

const hashPayload = (payload: any): string => {
    try {
        return btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
    } catch {
        return JSON.stringify(payload); // Fallback
    }
};

const MODEL_FALLBACK_HIERARCHY: Record<string, string | null> = {
  'gemini-2.5-pro': 'gemini-2.5-flash',
  'gemini-2.5-flash': 'gemini-flash-latest',
  'gemini-flash-latest': null,
  'gemini-2.5-flash-image': null, // No fallback for image models
};

type ApiCallFunction<T> = (payload: any) => Promise<T>;

interface EnqueueOptions {
    agentName: string;
    model: string;
    requestPayload: any;
}

const requestQueue: {
    apiCallFn: ApiCallFunction<any>;
    initialPayload: any;
    resolve: (value: any) => void;
    reject: (reason?: any) => void;
    options: EnqueueOptions;
}[] = [];

let activeRequests = 0;
let governorInterval: number | null = null;
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL_MS = 2000;


const processRequest = async (
    apiCallFn: ApiCallFunction<any>,
    initialPayload: any,
    resolve: (value: any) => void,
    reject: (reason?: any) => void,
    options: EnqueueOptions
) => {
    let currentPayload = { ...initialPayload };
    
    // Check Cache
    const payloadHash = hashPayload(currentPayload);
    const cached = requestCache.get(payloadHash);
    if (cached && (Date.now() - cached.timestamp < REQUEST_CACHE_EXPIRY_MS)) {
        console.log("Serving request from cache, saving quota!");
        resolve(cached.result);
        return; 
    }

    activeRequests++;
    const logId = apiMonitorService.addLog({
        agentName: options.agentName,
        model: options.model,
        requestPayload: options.requestPayload,
    });


    try {
        apiMonitorService.updateLog(logId, { status: 'Processing' });
        
        const auditLogData: Omit<ApiCallLog, 'id'> = {
            startTime: Date.now(),
            status: 'Processing',
            ...options,
        };
        auditLogService.logEvent('API_CALL', { ...auditLogData, id: logId });

        let attempts = 0;
        while (attempts < MAX_RETRIES) {
            try {
                const response = await fetch("/api/gemini/generateContent", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(currentPayload),
                });
                
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    const errMsg = errData.error?.message || `HTTP ${response.status}: ${response.statusText}`;
                    const errStatus = errData.error?.status || "INTERNAL_PROXY_ERROR";
                    const errCode = errData.error?.code || response.status;
                    const errorObj: any = new Error(errMsg);
                    errorObj.status = errStatus;
                    errorObj.code = errCode;
                    throw errorObj;
                }
                
                const result = await response.json();
                
                const endTime = Date.now();
                const duration = endTime - auditLogData.startTime;

                const usage = (result as any).usageMetadata;
                let usageData: Partial<ApiCallLog> = {};
                if (usage) {
                    const promptTokens = usage.promptTokenCount;
                    const candidateTokens = usage.candidatesTokenCount;
                    const totalTokens = usage.totalTokenCount;

                    if (typeof promptTokens === 'number' && typeof candidateTokens === 'number') {
                        const estimatedCost = calculateCost(options.model, promptTokens, candidateTokens);
                        usageData = { 
                            promptTokens, 
                            candidateTokens, 
                            totalTokens: typeof totalTokens === 'number' ? totalTokens : promptTokens + candidateTokens, 
                            estimatedCost 
                        };
                    }
                }
                
                const successLog: ApiCallLog = { ...auditLogData, id: logId, status: 'Success', responsePayload: result, endTime, duration, ...usageData };
                apiMonitorService.updateLog(logId, successLog);
                auditLogService.logEvent('API_CALL', successLog);

                requestCache.set(payloadHash, { result, timestamp: Date.now() });

                resolve(result);
                return;
            } catch (error: any) {
                 let parsedMessage;
                 try {
                     parsedMessage = JSON.parse(error.message);
                 } catch (e) { /* not a JSON message */ }

                 const status = error.status || parsedMessage?.error?.status;
                 const isQuotaError = status === 'RESOURCE_EXHAUSTED' || error.code === 429;

                 // Handle quota error by falling back to another model
                 if (isQuotaError) {
                     const currentModel = currentPayload.model;
                     const fallbackModel = MODEL_FALLBACK_HIERARCHY[currentModel as keyof typeof MODEL_FALLBACK_HIERARCHY];
                     
                     if (fallbackModel) {
                         console.warn(`Quota exceeded for model ${currentModel}. Falling back to ${fallbackModel}.`);
                         const newErrorMsg = `Quota exceeded. Falling back to ${fallbackModel}.`;
                         apiMonitorService.updateLog(logId, { status: 'Retrying', error: newErrorMsg });
                         auditLogService.logEvent('API_CALL', { ...auditLogData, id: logId, status: 'Retrying', error: newErrorMsg });

                         currentPayload = { ...currentPayload, model: fallbackModel };
                         options.model = fallbackModel; // Update options for logging consistency
                         // Reset attempts for the new model and continue the loop
                         attempts = 0;
                         continue;
                     }
                 }

                // If not a quota error, or no fallback is available, proceed with normal retry logic
                attempts++;
                const isRetryable = (error?.code >= 500 && error?.code < 600) || error?.code === 429;
                
                let detailedError = new Error(error.message || 'An unexpected API error occurred.');
                
                if (isRetryable && attempts < MAX_RETRIES) {
                    let backoffTime = INITIAL_BACKOFF_MS * Math.pow(2, attempts - 1);
                    const retryMatch = error.message.match(/retry in (\d+(?:\.\d+)?)s/i);
                    if (retryMatch) {
                        const requestedDelay = parseFloat(retryMatch[1]) * 1000;
                        backoffTime = Math.max(backoffTime, requestedDelay + 1000); // add 1s buffer
                    }
                    console.warn(`API call failed. Retrying in ${backoffTime}ms... (Attempt ${attempts}/${MAX_RETRIES})`);
                    
                    const retryLog: Partial<ApiCallLog> = { status: 'Retrying', error: detailedError.message };
                    apiMonitorService.updateLog(logId, retryLog);
                    auditLogService.logEvent('API_CALL', { ...auditLogData, id: logId, ...retryLog });

                    await new Promise(res => setTimeout(res, backoffTime));
                } else {
                    console.error(`API call failed after ${attempts} attempts.`, error);
                    
                    const endTime = Date.now();
                    const duration = endTime - auditLogData.startTime;
                    const failureLog: ApiCallLog = { ...auditLogData, id: logId, status: 'Failed', error: detailedError.message, responsePayload: error, endTime, duration };
                    apiMonitorService.updateLog(logId, failureLog);
                    auditLogService.logEvent('API_CALL', failureLog);

                    reject(detailedError);
                    return;
                }
            }
        }
    } finally {
        activeRequests--;
    }
};

const startGovernor = () => {
    if (governorInterval !== null) return; // Governor already running

    governorInterval = window.setInterval(() => {
        // If there are requests to process and we have available slots and cooldown has passed
        const now = Date.now();
        if (requestQueue.length > 0 && activeRequests < MAX_CONCURRENT_REQUESTS && (now - lastRequestTime >= MIN_REQUEST_INTERVAL_MS)) {
            lastRequestTime = now;
            const { apiCallFn, initialPayload, resolve, reject, options } = requestQueue.shift()!;
            processRequest(apiCallFn, initialPayload, resolve, reject, options);
        }
        
        // If the queue is empty and no requests are active, we can stop the governor
        // to save resources. It will be restarted when a new request is enqueued.
        if (requestQueue.length === 0 && activeRequests === 0 && governorInterval) {
            clearInterval(governorInterval);
            governorInterval = null;
        }
    }, GOVERNOR_INTERVAL_MS);
};


export const enqueueGeminiRequest = <T>(
    apiCallFn: ApiCallFunction<T>,
    initialPayload: any,
    options: EnqueueOptions,
): Promise<T> => {
    return new Promise<T>((resolve, reject) => {
        requestQueue.push({ apiCallFn, initialPayload, resolve, reject, options });
        // Ensure the governor is running whenever a request is added.
        startGovernor();
    });
};