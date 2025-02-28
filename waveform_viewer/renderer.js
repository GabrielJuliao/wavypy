class WaveformRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.waveformData = null;
    this.zoom = 1;
    this.offset = 0;
    this.selectedBands = new Set();
    this.displayMode = "stacked"; // 'stacked' or 'overlay'

    // Color palette for different bands (default fallback)
    this.defaultColors = [
      "#4CAF50", // Green
      "#2196F3", // Blue
      "#F44336", // Red
      "#9C27B0", // Purple
      "#FF9800", // Orange
      "#795548", // Brown
      "#607D8B", // Blue Grey
      "#E91E63", // Pink
      "#FFEB3B", // Yellow
      "#009688", // Teal
      "#673AB7", // Deep Purple
      "#CDDC39", // Lime
    ];

    // Specific color scheme for Rekordbox-style visualization
    this.colorScheme = {
      low: "#0055e2", // Blue
      low_mid: "#aa6d28", // Gold
      mid: "#f2aa3c", // Orange
      high: "#ffffff", // White
      ultra: "#ffffff", // White (same as high)
    };

    this.bandColors = {}; // Will be populated with actual colors when data is loaded

    this.colors = {
      background: "#000000",
      waveform: "#4CAF50",
      centerLine: "#cccccc",
    };
    this.renderStyle = "line"; // Default render style

    this.resizeCanvas();

    window.addEventListener("resize", () => this.resizeCanvas());
  }

  resizeCanvas() {
    const container = this.canvas.parentElement;
    this.canvas.width = container.clientWidth;
    this.canvas.height = container.clientHeight;
    this.render();
  }

  loadData(data) {
    console.log("Loading data into renderer:", data);
    this.waveformData = data;
    this.zoom = 1;
    this.offset = 0;

    // Initialize selected bands - select all by default
    this.selectedBands = new Set(data.bands || ["fullrange"]);

    // Set up color mapping for bands
    this.setupBandColors();

    // Create band checkboxes
    this.createBandCheckboxes();

    // Update file info panel
    this.updateFileInfo();

    // Render the data
    this.render();
    this.updateTimeInfo();
    this.enableControls();
    this.updateScrollBar();
  }

  setupBandColors() {
    // Reset band colors
    this.bandColors = {};

    if (!this.waveformData) return;

    // First check if the data includes a color scheme
    if (this.waveformData.freq && Array.isArray(this.waveformData.freq)) {
      console.log("Using color scheme from file");
      this.waveformData.freq.forEach((item) => {
        if (item.name && item.color && item.color.hex) {
          this.bandColors[item.name] = item.color.hex;
        }
      });
    }

    // For any bands without specified colors, use the predefined scheme if applicable
    // or fall back to default colors
    if (this.waveformData.bands) {
      this.waveformData.bands.forEach((band, index) => {
        if (!this.bandColors[band]) {
          if (this.colorScheme[band]) {
            this.bandColors[band] = this.colorScheme[band];
          } else {
            this.bandColors[band] =
              this.defaultColors[index % this.defaultColors.length];
          }
        }
      });
    }

    console.log("Band colors set up:", this.bandColors);
  }

  createBandCheckboxes() {
    const container = document.getElementById("band-checkboxes");
    container.innerHTML = ""; // Clear existing checkboxes

    if (!this.waveformData || !this.waveformData.bands) return;

    this.waveformData.bands.forEach((band) => {
      const color = this.bandColors[band] || "#4CAF50";

      const checkbox = document.createElement("label");
      checkbox.className = "band-checkbox";

      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = this.selectedBands.has(band);
      input.dataset.band = band;
      input.addEventListener("change", () => {
        if (input.checked) {
          this.selectedBands.add(band);
        } else {
          this.selectedBands.delete(band);
        }
        this.render();
      });

      const colorIndicator = document.createElement("span");
      colorIndicator.className = "color-indicator";
      colorIndicator.style.backgroundColor = color;

      checkbox.appendChild(input);
      checkbox.appendChild(document.createTextNode(band));
      checkbox.appendChild(colorIndicator);

      container.appendChild(checkbox);
    });
  }

  updateFileInfo() {
    if (!this.waveformData) return;

    const infoPanel = document.getElementById("file-info");
    const channels = this.waveformData.channels || 1;
    const sampleRate = this.waveformData.sample_rate || 44100;
    const samplesPerPixel = this.waveformData.samples_per_pixel || 256;
    const bits = this.waveformData.bits || 16;
    const bands = this.waveformData.bands
      ? this.waveformData.bands.join(", ")
      : "fullrange";
    const version = this.waveformData.version || "unknown";

    infoPanel.innerHTML = `
                  <strong>File Info:</strong>
                  Sample Rate: ${sampleRate}Hz |
                  Channels: ${channels} |
                  Resolution: ${bits}bit |
                  Samples per Pixel: ${samplesPerPixel} |
                  Version: ${version} |
                  Bands: ${bands}
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

    const bands = this.waveformData.bands || ["fullrange"];
    if (bands.length === 0) return 0;

    // Use the first available band to determine length
    const firstBand = bands[0];
    const bandData = this.waveformData.data[firstBand];

    return this.waveformData.length || (bandData ? bandData.length / 2 : 0);
  }

  render() {
    console.log("Rendering waveform...");

    if (!this.waveformData) {
      console.log("No waveform data to render");
      this.renderEmptyState();
      return;
    }

    console.log("Waveform data available, rendering...");
    const { width, height } = this.canvas;

    // Clear canvas
    this.ctx.fillStyle = this.colors.background;
    this.ctx.fillRect(0, 0, width, height);

    // Get selected bands
    const selectedBands = Array.from(this.selectedBands);
    if (selectedBands.length === 0) {
      this.renderEmptyState(
        "No bands selected. Select at least one band to display."
      );
      return;
    }

    if (this.displayMode === "stacked") {
      this.renderStackedBands(selectedBands, width, height);
    } else {
      this.renderOverlayBands(selectedBands, width, height);
    }
  }

  renderStackedBands(selectedBands, width, height) {
    // Calculate the height for each band
    const bandHeight = height / selectedBands.length;

    selectedBands.forEach((band, index) => {
      // Calculate the vertical position for this band
      const bandTop = index * bandHeight;
      const bandBottom = (index + 1) * bandHeight;
      const bandCenterY = (bandTop + bandBottom) / 2;

      // Draw center line for this band
      this.ctx.beginPath();
      this.ctx.moveTo(0, bandCenterY);
      this.ctx.lineTo(width, bandCenterY);
      this.ctx.strokeStyle = this.colors.centerLine;
      this.ctx.stroke();

      // Get the band color
      const bandColor =
        this.bandColors[band] ||
        this.defaultColors[index % this.defaultColors.length];

      // Calculate visible range
      const visibleStart = Math.floor(this.offset);
      const visibleEnd = Math.min(
        Math.ceil(this.offset + this.getVisibleWidth()),
        this.getFullWidth()
      );

      // Draw the band label
      this.ctx.fillStyle = "#333";
      this.ctx.font = "12px Arial";
      this.ctx.fillText(band, 5, bandTop + 15);

      // Render the waveform for this band
      this.renderBandWaveform(
        band,
        visibleStart,
        visibleEnd,
        width,
        bandHeight,
        bandCenterY,
        bandColor
      );
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

    // Calculate visible range
    const visibleStart = Math.floor(this.offset);
    const visibleEnd = Math.min(
      Math.ceil(this.offset + this.getVisibleWidth()),
      this.getFullWidth()
    );

    // Draw each band
    selectedBands.forEach((band, index) => {
      const bandColor =
        this.bandColors[band] ||
        this.defaultColors[index % this.defaultColors.length];

      // Render the waveform for this band
      this.renderBandWaveform(
        band,
        visibleStart,
        visibleEnd,
        width,
        height,
        centerY,
        bandColor
      );

      // Draw band label
      this.ctx.fillStyle = bandColor;
      this.ctx.font = "12px Arial";
      this.ctx.fillText(band, 5 + index * 60, 15);
    });
  }

  renderBandWaveform(
    band,
    visibleStart,
    visibleEnd,
    width,
    height,
    centerY,
    color
  ) {
    if (!this.waveformData.data[band]) {
      console.warn(`Band data not found for: ${band}`);
      return;
    }

    // Choose the appropriate rendering method based on the style
    switch (this.renderStyle) {
      case "solid":
        this.renderSolidStyle(
          band,
          visibleStart,
          visibleEnd,
          width,
          height,
          centerY,
          color
        );
        break;
      case "bars":
        this.renderBarStyle(
          band,
          visibleStart,
          visibleEnd,
          width,
          height,
          centerY,
          color
        );
        break;
      case "points":
        this.renderPointStyle(
          band,
          visibleStart,
          visibleEnd,
          width,
          height,
          centerY,
          color
        );
        break;
      case "line":
      default:
        this.renderLineStyle(
          band,
          visibleStart,
          visibleEnd,
          width,
          height,
          centerY,
          color
        );
        break;
    }
  }

  renderLineStyle(
    band,
    visibleStart,
    visibleEnd,
    width,
    height,
    centerY,
    color
  ) {
    // Draw waveform as connected lines between min/max points
    this.ctx.beginPath();
    let pointsDrawn = 0;
    const bandData = this.waveformData.data[band];

    if (!bandData || bandData.length === 0) {
      console.warn(`No data for band: ${band}`);
      return;
    }

    for (let i = visibleStart; i < visibleEnd; i++) {
      if (i < 0 || i >= this.getFullWidth()) continue;

      const channels = this.waveformData.channels || 1;

      // Calculate the x-coordinate on the canvas
      const x = (i - this.offset) * this.zoom;

      // Direct access to the data - our format is just [min1, max1, min2, max2, ...]
      const pointIndex = i * 2; // Each point has min and max

      if (pointIndex + 1 >= bandData.length) {
        continue; // Skip if data is out of bounds
      }

      const minY = bandData[pointIndex]; // min value
      const maxY = bandData[pointIndex + 1]; // max value

      // Normalize values to the available height
      const scaleFactor = (height / 2) * 0.8; // 80% of half height for better visibility

      // Convert to y-coordinates (invert because canvas y increases downward)
      const yMin = centerY - (minY / 32768) * scaleFactor;
      const yMax = centerY - (maxY / 32768) * scaleFactor;

      // Draw vertical line for this point
      this.ctx.moveTo(x, yMin);
      this.ctx.lineTo(x, yMax);

      pointsDrawn++;
    }

    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    console.log(`Drew ${pointsDrawn} points for band ${band} (line style)`);
  }

  renderSolidStyle(
    band,
    visibleStart,
    visibleEnd,
    width,
    height,
    centerY,
    color
  ) {
    // Draw waveform as a filled shape
    this.ctx.beginPath();
    let pointsDrawn = 0;
    const bandData = this.waveformData.data[band];

    if (!bandData || bandData.length === 0) return;

    // Start at the center line
    this.ctx.moveTo((visibleStart - this.offset) * this.zoom, centerY);

    // Draw the top edge (max values)
    for (let i = visibleStart; i < visibleEnd; i++) {
      if (i < 0 || i >= this.getFullWidth()) continue;

      const pointIndex = i * 2;

      if (pointIndex + 1 >= bandData.length) continue;

      const maxY = bandData[pointIndex + 1]; // max value
      const x = (i - this.offset) * this.zoom;
      const scaleFactor = (height / 2) * 0.8;
      const y = centerY - (maxY / 32768) * scaleFactor;

      this.ctx.lineTo(x, y);
      pointsDrawn++;
    }

    // Draw the bottom edge (min values) in reverse
    for (let i = visibleEnd - 1; i >= visibleStart; i--) {
      if (i < 0 || i >= this.getFullWidth()) continue;

      const pointIndex = i * 2;

      if (pointIndex >= bandData.length) continue;

      const minY = bandData[pointIndex]; // min value
      const x = (i - this.offset) * this.zoom;
      const scaleFactor = (height / 2) * 0.8;
      const y = centerY - (minY / 32768) * scaleFactor;

      this.ctx.lineTo(x, y);
      pointsDrawn++;
    }

    // Close the path back to the start
    this.ctx.closePath();

    // Fill the shape
    this.ctx.fillStyle = color;
    this.ctx.globalAlpha = 0.5; // Semi-transparent
    this.ctx.fill();
    this.ctx.globalAlpha = 1.0; // Reset alpha

    console.log(`Drew ${pointsDrawn} points for band ${band} (solid style)`);
  }

  renderBarStyle(
    band,
    visibleStart,
    visibleEnd,
    width,
    height,
    centerY,
    color
  ) {
    // Draw waveform as bars
    let pointsDrawn = 0;
    const bandData = this.waveformData.data[band];

    if (!bandData || bandData.length === 0) return;

    const barWidth = Math.max(2, Math.floor(this.zoom - 1)); // Adjust bar width based on zoom

    this.ctx.fillStyle = color;

    for (let i = visibleStart; i < visibleEnd; i++) {
      if (i < 0 || i >= this.getFullWidth()) continue;

      const pointIndex = i * 2;

      if (pointIndex + 1 >= bandData.length) continue;

      const minY = bandData[pointIndex]; // min value
      const maxY = bandData[pointIndex + 1]; // max value

      const x = (i - this.offset) * this.zoom;
      const scaleFactor = (height / 2) * 0.8;

      const yMin = centerY - (minY / 32768) * scaleFactor;
      const yMax = centerY - (maxY / 32768) * scaleFactor;

      // Draw a rectangle for this bar
      this.ctx.fillRect(
        x - barWidth / 2, // center the bar on the x-coordinate
        yMax, // top of the bar (max value)
        barWidth, // width of the bar
        yMin - yMax // height of the bar
      );

      pointsDrawn++;
    }

    console.log(`Drew ${pointsDrawn} points for band ${band} (bar style)`);
  }

  renderPointStyle(
    band,
    visibleStart,
    visibleEnd,
    width,
    height,
    centerY,
    color
  ) {
    // Draw waveform as points
    let pointsDrawn = 0;
    const bandData = this.waveformData.data[band];

    if (!bandData || bandData.length === 0) return;

    const pointRadius = Math.max(2, Math.floor(this.zoom / 2)); // Adjust point size based on zoom

    this.ctx.fillStyle = color;

    for (let i = visibleStart; i < visibleEnd; i++) {
      if (i < 0 || i >= this.getFullWidth()) continue;

      const pointIndex = i * 2;

      if (pointIndex + 1 >= bandData.length) continue;

      const minY = bandData[pointIndex]; // min value
      const maxY = bandData[pointIndex + 1]; // max value

      const x = (i - this.offset) * this.zoom;
      const scaleFactor = (height / 2) * 0.8;

      const yMin = centerY - (minY / 32768) * scaleFactor;
      const yMax = centerY - (maxY / 32768) * scaleFactor;

      // Draw points at min and max
      this.ctx.beginPath();
      this.ctx.arc(x, yMin, pointRadius, 0, Math.PI * 2);
      this.ctx.fill();

      this.ctx.beginPath();
      this.ctx.arc(x, yMax, pointRadius, 0, Math.PI * 2);
      this.ctx.fill();

      pointsDrawn += 2; // Count both min and max points
    }

    console.log(`Drew ${pointsDrawn} points for band ${band} (point style)`);
  }

  renderEmptyState(message = "Upload a waveform JSON file to visualize") {
    const { width, height } = this.canvas;

    // Clear canvas
    this.ctx.fillStyle = this.colors.background;
    this.ctx.fillRect(0, 0, width, height);

    // Draw message
    this.ctx.fillStyle = "#666";
    this.ctx.font = "14px Arial";
    this.ctx.textAlign = "center";
    this.ctx.fillText(message, width / 2, height / 2);
  }

  updateTimeInfo() {
    if (!this.waveformData) return;

    const samplesPerPixel = this.waveformData.samples_per_pixel || 256; // Default if not provided
    const sampleRate = this.waveformData.sample_rate || 44100; // Default if not provided

    const totalPoints = this.getFullWidth();
    const totalDuration = (totalPoints * samplesPerPixel) / sampleRate;
    const startTime = (this.offset * samplesPerPixel) / sampleRate;
    const endTime =
      ((this.offset + this.getVisibleWidth()) * samplesPerPixel) / sampleRate;

    document.getElementById(
      "time-info"
    ).textContent = `Duration: ${totalDuration.toFixed(
      2
    )} seconds | View: ${startTime.toFixed(2)}s - ${Math.min(
      endTime,
      totalDuration
    ).toFixed(2)}s`;
  }

  updateZoomLabel() {
    document.getElementById("zoom-value").textContent = `${this.zoom.toFixed(
      1
    )}x`;
  }

  enableControls() {
    document.getElementById("zoom-in").disabled = false;
    document.getElementById("zoom-out").disabled = false;
    document.getElementById("scroll-bar").disabled = false;
    document.getElementById("waveform-style").disabled = false;
    document.getElementById("display-mode").disabled = false;
  }

  updateScrollBar() {
    const scrollBar = document.getElementById("scroll-bar");
    if (!this.waveformData) return;

    const totalWidth = this.getFullWidth();
    const visibleWidth = this.getVisibleWidth();

    // Update range
    scrollBar.max = Math.max(0, totalWidth - visibleWidth);
    scrollBar.value = this.offset;

    // Disable scrollbar if everything is visible
    scrollBar.disabled = totalWidth <= visibleWidth;
  }

  setDisplayMode(mode) {
    this.displayMode = mode;
    this.render();
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const renderer = new WaveformRenderer("waveform");

  // Initialize with disabled controls
  document.getElementById("zoom-in").disabled = true;
  document.getElementById("zoom-out").disabled = true;
  document.getElementById("scroll-bar").disabled = true;
  document.getElementById("waveform-style").disabled = true;
  document.getElementById("display-mode").disabled = true;

  // Attach event listeners
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
    .addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const loading = document.getElementById("loading");
      loading.style.display = "inline";

      const reader = new FileReader();
      reader.onload = function (e) {
        try {
          const jsonData = JSON.parse(e.target.result);
          console.log("Loaded JSON data:", jsonData);

          // Handle different versions of waveform data
          if (jsonData.version === 3 || jsonData.bands) {
            // New multi-band format
            console.log("Detected multi-band waveform data (version 3)");

            // If data is not in the expected format, try to adapt it
            if (!jsonData.data || typeof jsonData.data !== "object") {
              alert("Error: Invalid waveform data format");
              loading.style.display = "none";
              return;
            }

            renderer.loadData(jsonData);
          } else {
            // Legacy format (version 1 or 2) - convert to new format
            console.log(
              "Detected legacy waveform data, converting to multi-band format"
            );

            const convertedData = {
              version: jsonData.version || 2,
              length: jsonData.length,
              channels: jsonData.channels || 1,
              samples_per_pixel: jsonData.samples_per_pixel || 256,
              sample_rate: jsonData.sample_rate || 44100,
              bits: jsonData.bits || 16,
              bands: ["fullrange"],
              data: {
                fullrange: jsonData.data,
              },
            };

            renderer.loadData(convertedData);
          }
        } catch (err) {
          console.error("JSON parsing error:", err);
          alert("Error parsing JSON file: " + err.message);
        } finally {
          loading.style.display = "none";
        }
      };
      reader.onerror = function () {
        alert("Error reading file");
        loading.style.display = "none";
      };
      reader.readAsText(file);
    });
});
