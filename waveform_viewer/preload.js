// preload.js
const { contextBridge, ipcRenderer } = require("electron");

console.log("Preload script loaded");

contextBridge.exposeInMainWorld("electronAPI", {
  decompress: (compressedData, compressionType) =>
    ipcRenderer.invoke("decompress", compressedData, compressionType),
});
