/* ProsperoAI Desktop — preload bridge (contextIsolation + sandbox). */

import { contextBridge, ipcRenderer } from 'electron';
import type {
  ChatStreamDelta,
  DesktopEvent,
  HubProgress,
  ProsperoApi,
} from '../shared/types';

function subscribe<T>(channel: string, callback: (payload: T) => void): () => void {
  const listener = (_event: Electron.IpcRendererEvent, payload: T) => callback(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

const api: ProsperoApi = {
  getAppInfo: () => ipcRenderer.invoke('app-info'),
  getSettings: () => ipcRenderer.invoke('settings-get'),
  setSettings: (patch) => ipcRenderer.invoke('settings-set', patch),

  checkGateway: (baseUrl) => ipcRenderer.invoke('gateway-check', baseUrl),
  completeChat: (baseUrl, model, messages, params) =>
    ipcRenderer.invoke('gateway-chat', baseUrl, model, messages, params),
  streamChat: (requestId, baseUrl, model, messages, params, onDelta) => {
    const unsubscribe = subscribe<ChatStreamDelta>('chat-delta', (delta) => {
      if (delta.requestId === requestId) onDelta(delta);
    });
    return ipcRenderer
      .invoke('chat-stream-start', requestId, baseUrl, model, messages, params)
      .finally(unsubscribe);
  },
  cancelChatStream: (requestId) => {
    void ipcRenderer.invoke('chat-stream-cancel', requestId);
  },

  listDevices: () => ipcRenderer.invoke('devices-list'),
  addDevice: (input) => ipcRenderer.invoke('devices-add', input),
  removeDevice: (id) => ipcRenderer.invoke('devices-remove', id),
  probeDevices: () => ipcRenderer.invoke('devices-probe'),

  listEvents: () => ipcRenderer.invoke('events-list'),
  logEvent: (level, subsystem, message) => ipcRenderer.invoke('events-log', level, subsystem, message),
  exportDiagnostics: () => ipcRenderer.invoke('diagnostics-export'),
  onEvent: (callback) => subscribe<DesktopEvent>('desktop-event', callback),

  listLibrary: () => ipcRenderer.invoke('library-list'),
  importModels: () => ipcRenderer.invoke('library-import'),
  optimizeModel: (id) => ipcRenderer.invoke('library-optimize', id),

  runBenchmark: (model, options) => ipcRenderer.invoke('benchmark-run', model, options),
  listBenchmarks: () => ipcRenderer.invoke('benchmark-list'),
  deleteBenchmark: (id) => ipcRenderer.invoke('benchmark-delete', id),

  selectModelFiles: () => ipcRenderer.invoke('select-model-files'),
  listLocalModelFiles: () => ipcRenderer.invoke('list-local-model-files'),
  searchHub: (request) => ipcRenderer.invoke('hub-search', request),
  startHubPull: (request) => ipcRenderer.invoke('hub-pull-start', request),
  cancelHubPull: (operationId) => ipcRenderer.invoke('hub-pull-cancel', operationId),
  onHubProgress: (callback) => subscribe<HubProgress>('hub-progress', callback),
};

contextBridge.exposeInMainWorld('prospero', api);
