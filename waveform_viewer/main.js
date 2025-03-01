const {
  app,
  BrowserWindow,
  ipcMain,
  nativeTheme,
  screen,
} = require("electron/main");

const path = require("node:path");
const pako = require("pako");

function createWindow() {
  // Get the primary display's dimensions
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.workAreaSize;

  const win = new BrowserWindow({
    width: width,
    height: height,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  win.loadFile("index.html");
  win.webContents.openDevTools();
}

// IPC handler to decompress data
ipcMain.handle('decompress', async (event, compressedData, compressionType) => {
  try {
    if (compressionType === 'gzip') {
      const byteArray = new Uint8Array(compressedData); // Convert input to Uint8Array
      const decompressed = pako.inflate(byteArray); // Uint8Array
      return Array.from(decompressed); // Convert to array for IPC serialization
    }
    return compressedData; // No compression, return as-is
  } catch (error) {
    console.error('Decompression error:', error);
    throw error;
  }
});

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  app.quit();
});
