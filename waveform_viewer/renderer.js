class WaveformRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.waveformData = null;
    this.zoom = 1;
    this.offset = 0;
    this.selectedBands = new Set();
    this.displayMode = "stacked"; // 'stacked' or 'overlay'

    // Color palette for different bands
    this.defaultColors = [
      "#4CAF50",
      "#2196F3",
      "#F44336",
      "#9C27B0",
      "#FF9800",
      "#795548",
      "#607D8B",
      "#E91E63",
      "#FFEB3B",
      "#009688",
      "#673AB7",
      "#CDDC39",
    ];

    this.colorScheme = {
      low: "#0055e2",
      low_mid: "#aa6d28",
      mid: "#f2aa3c",
      high: "#ffffff",
      ultra: "#ffffff",
    };

    this.bandColors = {};
    this.colors = {
      background: "#000000",
      centerLine: "#cccccc",
    };
    this.renderStyle = "line";

    this.resizeCanvas();
    window.addEventListener("resize", () => this.resizeCanvas());
  }

  resizeCanvas() {
    const container = this.canvas.parentElement;
    this.canvas.width = container.clientWidth;
    this.canvas.height = container.clientHeight;
    this.render();
  }

  async loadData(data) {
    console.log("Loading waveform data:", data);
    this.waveformData = await this.decodeWaveformData(data);
    this.zoom = 1;
    this.offset = 0;

    // Initialize selected bands
    const bandNames =
      this.waveformData.type === "multiband"
        ? this.waveformData.data.multiband.bands.map((b) => b.name)
        : ["fullrange"];
    this.selectedBands = new Set(bandNames);

    this.setupBandColors();
    this.createBandCheckboxes();
    this.updateFileInfo();
    this.render();
    this.updateTimeInfo();
    this.enableControls();
    this.updateScrollBar();
  }

  async decodeWaveformData(data) {
    // Handle the JSON structure from the waveform generator
    const decodedData = { ...data };

    if (data.type === "multiband") {
      for (const band of decodedData.data.multiband.bands) {
        band.samples = await this.decodeSamples(
          band.samples,
          band.sample_format,
          band.compression,
          data.bits_per_sample
        );
      }
    } else if (data.type === "fullrange") {
      decodedData.data.fullrange.samples = await this.decodeSamples(
        data.data.fullrange.samples,
        data.data.fullrange.sample_format,
        data.data.fullrange.compression,
        data.bits_per_sample
      );
    }

    return decodedData;
  }

  async decodeSamples(samples, sampleFormat, compression, bits) {
    // Decode base64 string
    const binaryString = atob(samples);
    let byteArray;

    // Handle compression
    if (compression === "gzip") {
      const compressedBytes = [...binaryString].map((ch) => ch.charCodeAt(0));
      const decompressed = await window.electronAPI.decompress(
        compressedBytes,
        compression
      );
      byteArray = new Uint8Array(decompressed); // Convert plain array to Uint8Array
    } else {
      byteArray = new Uint8Array(
        [...binaryString].map((ch) => ch.charCodeAt(0))
      );
    }

    // Decode based on sample format
    if (sampleFormat === "base64_json") {
      const jsonString = new TextDecoder().decode(byteArray);
      return JSON.parse(jsonString);
    } else {
      // base64_binary
      const dataView = new DataView(byteArray.buffer);
      const sampleCount = byteArray.length / (bits === 8 ? 1 : 2);
      const samplesArray = new Int16Array(sampleCount);

      for (let i = 0; i < sampleCount; i++) {
        if (bits === 8) {
          samplesArray[i] = dataView.getInt8(i) * 256;
        } else {
          samplesArray[i] = dataView.getInt16(i * 2, true);
        }
      }
      return Array.from(samplesArray);
    }
  }

  setupBandColors() {
    this.bandColors = {};
    if (!this.waveformData) return;

    const bands =
      this.waveformData.type === "multiband"
        ? this.waveformData.data.multiband.bands
        : [{ name: "fullrange" }];

    bands.forEach((band, index) => {
      const bandName = band.name;
      this.bandColors[bandName] =
        this.colorScheme[bandName] ||
        this.defaultColors[index % this.defaultColors.length];
    });
  }

  createBandCheckboxes() {
    const container = document.getElementById("band-checkboxes");
    container.innerHTML = "";

    if (!this.waveformData) return;

    const bands =
      this.waveformData.type === "multiband"
        ? this.waveformData.data.multiband.bands
        : [{ name: "fullrange" }];

    bands.forEach((band) => {
      const color = this.bandColors[band.name];
      const checkbox = document.createElement("label");
      checkbox.className = "band-checkbox";

      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = this.selectedBands.has(band.name);
      input.dataset.band = band.name;
      input.addEventListener("change", () => {
        this.selectedBands[input.checked ? "add" : "delete"](band.name);
        this.render();
      });

      const colorIndicator = document.createElement("span");
      colorIndicator.className = "color-indicator";
      colorIndicator.style.backgroundColor = color;

      checkbox.appendChild(input);
      checkbox.appendChild(document.createTextNode(band.name));
      checkbox.appendChild(colorIndicator);
      container.appendChild(checkbox);
    });
  }

  updateFileInfo() {
    if (!this.waveformData) return;

    const infoPanel = document.getElementById("file-info");
    const {
      channels,
      sample_rate,
      bits_per_sample,
      samples_per_pixel,
      duration,
      type,
      version,
      metadata,
    } = this.waveformData;
    const bands =
      type === "multiband"
        ? this.waveformData.data.multiband.bands.map((b) => b.name).join(", ")
        : "fullrange";

    infoPanel.innerHTML = `
      <strong>File Info:</strong>
      Sample Rate: ${sample_rate}Hz |
      Channels: ${channels} |
      Resolution: ${bits_per_sample}bit |
      Samples per Pixel: ${samples_per_pixel} |
      Duration: ${duration.toFixed(2)}s |
      Version: ${version} |
      Type: ${type} |
      Bands: ${bands} |
      Compression: ${metadata.compression}
    `;
  }

  zoomIn() {
    if (this.zoom < 10) {
      const centerOffset = this.offset + this.getVisibleWidth() / 2;
      this.zoom *= 1.5;
      this.offset = centerOffset - this.getVisibleWidth() / 2;
      this.clampOffset();
      this.render();
      this.updateTimeInfo();
      this.updateZoomLabel();
      this.updateScrollBar();
    }
  }

  zoomOut() {
    if (this.zoom > 0.1) {
      const centerOffset = this.offset + this.getVisibleWidth() / 2;
      this.zoom /= 1.5;
      this.offset = centerOffset - this.getVisibleWidth() / 2;
      this.clampOffset();
      this.render();
      this.updateTimeInfo();
      this.updateZoomLabel();
      this.updateScrollBar();
    }
  }

  clampOffset() {
    const maxOffset = Math.max(0, this.getFullWidth() - this.getVisibleWidth());
    this.offset = Math.min(Math.max(0, this.offset), maxOffset);
  }

  getVisibleWidth() {
    return this.canvas.width / this.zoom;
  }

  getFullWidth() {
    if (!this.waveformData) return 0;

    const samples =
      this.waveformData.type === "multiband"
        ? this.waveformData.data.multiband.bands[0].samples
        : this.waveformData.data.fullrange.samples;
    return samples.length / 2; // Each point has min and max
  }

  render() {
    if (!this.waveformData) {
      this.renderEmptyState();
      return;
    }

    const { width, height } = this.canvas;
    this.ctx.fillStyle = this.colors.background;
    this.ctx.fillRect(0, 0, width, height);

    const selectedBands = Array.from(this.selectedBands);
    if (selectedBands.length === 0) {
      this.renderEmptyState("No bands selected.");
      return;
    }

    this.displayMode === "stacked"
      ? this.renderStackedBands(selectedBands, width, height)
      : this.renderOverlayBands(selectedBands, width, height);
  }

  renderStackedBands(selectedBands, width, height) {
    const bandHeight = height / selectedBands.length;

    selectedBands.forEach((bandName, index) => {
      const bandTop = index * bandHeight;
      const bandCenterY = bandTop + bandHeight / 2;

      this.ctx.beginPath();
      this.ctx.moveTo(0, bandCenterY);
      this.ctx.lineTo(width, bandCenterY);
      this.ctx.strokeStyle = this.colors.centerLine;
      this.ctx.stroke();

      const bandColor = this.bandColors[bandName];
      this.renderBandWaveform(
        bandName,
        bandTop,
        bandHeight,
        bandCenterY,
        bandColor
      );

      this.ctx.fillStyle = "#333";
      this.ctx.font = "12px Arial";
      this.ctx.fillText(bandName, 5, bandTop + 15);
    });
  }

  renderOverlayBands(selectedBands, width, height) {
    const centerY = height / 2;

    // Draw center line
    this.ctx.beginPath();
    this.ctx.moveTo(0, centerY);
    this.ctx.lineTo(width, centerY);
    this.ctx.strokeStyle = this.colors.centerLine;
    this.ctx.stroke();

    // Define band rendering order (low first, then mid, then high)
    const renderOrder = ["low", "mid", "high", "ultra"];

    // Sort the bands based on render order
    const orderedBands = [...selectedBands].sort((a, b) => {
      const indexA = renderOrder.indexOf(a);
      const indexB = renderOrder.indexOf(b);
      // If not found in order, put at end
      return (indexA === -1 ? 999 : indexA) - (indexB === -1 ? 999 : indexB);
    });

    // Render bands in the correct order
    orderedBands.forEach((bandName) => {
      const bandColor = this.bandColors[bandName];
      this.renderBandWaveform(
        bandName,
        0,
        height,
        centerY,
        bandColor,
        bandName
      );
    });

    // Draw band names
    this.ctx.font = "12px Arial";
    orderedBands.forEach((bandName, index) => {
      const bandColor = this.bandColors[bandName];
      this.ctx.fillStyle = bandColor;
      this.ctx.fillText(bandName, 5 + index * 60, 15);
    });
  }

  renderBandWaveform(bandName, bandTop, bandHeight, centerY, color) {
    const samples = this.getBandSamples(bandName);
    if (!samples) return;

    const visibleStart = Math.floor(this.offset);
    const visibleEnd = Math.min(
      Math.ceil(this.offset + this.getVisibleWidth()),
      samples.length / 2
    );

    switch (this.renderStyle) {
      case "solid":
        this.renderSolidStyle(
          samples,
          visibleStart,
          visibleEnd,
          bandHeight,
          centerY,
          color
        );
        break;
      case "bars":
        this.renderBarStyle(
          samples,
          visibleStart,
          visibleEnd,
          bandHeight,
          centerY,
          color
        );
        break;
      case "points":
        this.renderPointStyle(
          samples,
          visibleStart,
          visibleEnd,
          bandHeight,
          centerY,
          color
        );
        break;
      case "club": // New "club" style
        this.renderClubStyle(
          samples,
          visibleStart,
          visibleEnd,
          bandHeight,
          centerY,
          color
        );
        break;
      case "line":
      default:
        this.renderLineStyle(
          samples,
          visibleStart,
          visibleEnd,
          bandHeight,
          centerY,
          color
        );
        break;
    }
  }

  getBandSamples(bandName) {
    if (this.waveformData.type === "multiband") {
      const band = this.waveformData.data.multiband.bands.find(
        (b) => b.name === bandName
      );
      return band ? band.samples : null;
    }
    return this.waveformData.data.fullrange.samples;
  }

  renderLineStyle(samples, visibleStart, visibleEnd, height, centerY, color) {
    this.ctx.beginPath();
    for (let i = visibleStart; i < visibleEnd; i++) {
      const x = (i - this.offset) * this.zoom;
      const minY = samples[i * 2];
      const maxY = samples[i * 2 + 1];
      const scaleFactor = (height / 2) * 0.8;
      const yMin = centerY - (minY / 32768) * scaleFactor;
      const yMax = centerY - (maxY / 32768) * scaleFactor;
      this.ctx.moveTo(x, yMin);
      this.ctx.lineTo(x, yMax);
    }
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 1;
    this.ctx.stroke();
  }

  // Just the renderClubStyle function with overlay logic added

  renderClubStyle(
    samples,
    visibleStart,
    visibleEnd,
    height,
    centerY,
    color,
    bandName
  ) {
    const ctx = this.ctx;
    const scaleFactor = (height / 2) * 0.8;

    // Apply appropriate blending mode based on band name
    if (this.displayMode === "overlay") {
      if (bandName === "low") {
        ctx.globalCompositeOperation = "source-over"; // Base layer
      } else if (bandName === "mid") {
        ctx.globalCompositeOperation = "lighten"; // To blend with low
      } else if (bandName === "high" || bandName === "ultra") {
        ctx.globalCompositeOperation = "lighten"; // To preserve white
      } else {
        ctx.globalCompositeOperation = "source-over"; // Default
      }
    }

    // Set transparency based on band
    if (bandName === "high" || bandName === "ultra") {
      ctx.globalAlpha = 0.9;
    } else if (bandName === "low") {
      ctx.globalAlpha = 0.8;
    } else if (bandName === "mid") {
      ctx.globalAlpha = 0.7;
    } else {
      ctx.globalAlpha = 0.8; // Default
    }

    // Draw filled waveform
    ctx.beginPath();
    ctx.moveTo((visibleStart - this.offset) * this.zoom, centerY);

    // Upper waveform
    for (let i = visibleStart; i < visibleEnd; i++) {
      const x = (i - this.offset) * this.zoom;
      const maxY = samples[i * 2 + 1]; // Raw upper peak
      const yMax = centerY - (maxY / 32768) * scaleFactor;
      ctx.lineTo(x, yMax);
    }

    // Complete the path back to center
    const lastX = (visibleEnd - 1 - this.offset) * this.zoom;
    ctx.lineTo(lastX, centerY);

    // Lower waveform (in reverse)
    for (let i = visibleEnd - 1; i >= visibleStart; i--) {
      const x = (i - this.offset) * this.zoom;
      const minY = samples[i * 2]; // Raw lower peak
      const yMin = centerY - (minY / 32768) * scaleFactor;
      ctx.lineTo(x, yMin);
    }

    // Close the path
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();

    // Add outline with slight glow
    ctx.lineWidth = 1;
    ctx.strokeStyle = color;
    ctx.stroke();

    // Special effect for high frequencies
    if (bandName === "high" || bandName === "ultra") {
      ctx.shadowColor = "#ffffff";
      ctx.shadowBlur = 4;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Reset canvas properties
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0;
  }

  renderSolidStyle(samples, visibleStart, visibleEnd, height, centerY, color) {
    this.ctx.beginPath();
    this.ctx.moveTo((visibleStart - this.offset) * this.zoom, centerY);
    for (let i = visibleStart; i < visibleEnd; i++) {
      const x = (i - this.offset) * this.zoom;
      const maxY = samples[i * 2 + 1];
      const scaleFactor = (height / 2) * 0.8;
      this.ctx.lineTo(x, centerY - (maxY / 32768) * scaleFactor);
    }
    for (let i = visibleEnd - 1; i >= visibleStart; i--) {
      const x = (i - this.offset) * this.zoom;
      const minY = samples[i * 2];
      const scaleFactor = (height / 2) * 0.8;
      this.ctx.lineTo(x, centerY - (minY / 32768) * scaleFactor);
    }
    this.ctx.closePath();
    this.ctx.fillStyle = color;
    this.ctx.globalAlpha = 0.5;
    this.ctx.fill();
    this.ctx.globalAlpha = 1.0;
  }

  renderBarStyle(samples, visibleStart, visibleEnd, height, centerY, color) {
    const barWidth = Math.max(2, Math.floor(this.zoom - 1));
    this.ctx.fillStyle = color;
    for (let i = visibleStart; i < visibleEnd; i++) {
      const x = (i - this.offset) * this.zoom;
      const minY = samples[i * 2];
      const maxY = samples[i * 2 + 1];
      const scaleFactor = (height / 2) * 0.8;
      const yMin = centerY - (minY / 32768) * scaleFactor;
      const yMax = centerY - (maxY / 32768) * scaleFactor;
      this.ctx.fillRect(x - barWidth / 2, yMax, barWidth, yMin - yMax);
    }
  }

  renderPointStyle(samples, visibleStart, visibleEnd, height, centerY, color) {
    const pointRadius = Math.max(2, Math.floor(this.zoom / 2));
    this.ctx.fillStyle = color;
    for (let i = visibleStart; i < visibleEnd; i++) {
      const x = (i - this.offset) * this.zoom;
      const minY = samples[i * 2];
      const maxY = samples[i * 2 + 1];
      const scaleFactor = (height / 2) * 0.8;
      const yMin = centerY - (minY / 32768) * scaleFactor;
      const yMax = centerY - (maxY / 32768) * scaleFactor;
      this.ctx.beginPath();
      this.ctx.arc(x, yMin, pointRadius, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.beginPath();
      this.ctx.arc(x, yMax, pointRadius, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  renderEmptyState(message = "Upload a waveform JSON file to visualize") {
    const { width, height } = this.canvas;
    this.ctx.fillStyle = this.colors.background;
    this.ctx.fillRect(0, 0, width, height);
    this.ctx.fillStyle = "#666";
    this.ctx.font = "14px Arial";
    this.ctx.textAlign = "center";
    this.ctx.fillText(message, width / 2, height / 2);
  }

  updateTimeInfo() {
    if (!this.waveformData) return;

    const { samples_per_pixel, sample_rate } = this.waveformData;
    const totalPoints = this.getFullWidth();
    const totalDuration = (totalPoints * samples_per_pixel) / sample_rate;
    const startTime = (this.offset * samples_per_pixel) / sample_rate;
    const endTime = Math.min(
      totalDuration,
      ((this.offset + this.getVisibleWidth()) * samples_per_pixel) / sample_rate
    );

    document.getElementById(
      "time-info"
    ).textContent = `Duration: ${totalDuration.toFixed(
      2
    )}s | View: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`;
  }

  updateZoomLabel() {
    document.getElementById("zoom-value").textContent = `${this.zoom.toFixed(
      1
    )}x`;
  }

  enableControls() {
    [
      "zoom-in",
      "zoom-out",
      "scroll-bar",
      "waveform-style",
      "display-mode",
    ].forEach((id) => (document.getElementById(id).disabled = false));
  }

  updateScrollBar() {
    const scrollBar = document.getElementById("scroll-bar");
    if (!this.waveformData) return;

    const totalWidth = this.getFullWidth();
    const visibleWidth = this.getVisibleWidth();
    scrollBar.max = Math.max(0, totalWidth - visibleWidth);
    scrollBar.value = this.offset;
    scrollBar.disabled = totalWidth <= visibleWidth;
  }

  setDisplayMode(mode) {
    this.displayMode = mode;
    this.render();
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const renderer = new WaveformRenderer("waveform");

  document
    .getElementById("zoom-in")
    .addEventListener("click", () => renderer.zoomIn());
  document
    .getElementById("zoom-out")
    .addEventListener("click", () => renderer.zoomOut());
  document.getElementById("scroll-bar").addEventListener("input", (e) => {
    renderer.offset = parseFloat(e.target.value);
    renderer.render();
    renderer.updateTimeInfo();
  });
  document.getElementById("waveform-style").addEventListener("change", (e) => {
    renderer.renderStyle = e.target.value;
    renderer.render();
  });
  document.getElementById("display-mode").addEventListener("change", (e) => {
    renderer.setDisplayMode(e.target.value);
  });

  document
    .getElementById("file-input")
    .addEventListener("change", async function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const loading = document.getElementById("loading");
      loading.style.display = "inline";

      try {
        const text = await file.text();
        const jsonData = JSON.parse(text);
        await renderer.loadData(jsonData);
      } catch (err) {
        console.error("Error processing file:", err);
        alert("Error loading waveform file: " + err.message);
      } finally {
        loading.style.display = "none";
      }
    });
});
